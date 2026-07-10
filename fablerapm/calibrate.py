"""Null calibration for interaction curvature signs.

The free-signed interaction estimator is biased: ridge shrinkage
under-predicts talented lineups, so talent-derived concentration features
pick up spurious positive curvature even when the world is perfectly
additive (demonstrated in tests/test_interactions.py). Sign constraints are
the pipeline's day-to-day defense; this module is the principled one — it
measures the estimator's bias directly and asks whether the real data's
curvature exceeds it:

1. Fit the linear (two-phase) model on the real stints.
2. Simulate synthetic copies of the dataset from that fit — identical
   games, lineups, and possession counts; points drawn Poisson around the
   linear model's predicted scoring rates. Additive truth by construction,
   so every nonzero gamma the estimator finds on a simulation is bias.
3. Re-run the *entire* free-signed interaction pipeline on each simulation
   (including its internal phase-1/phase-2 refits, whose estimated talent
   feeds the concentration features) to get the null distribution of each
   gamma under "no curvature + this estimator on this exact dataset".
4. Report the real-data curvature against its null band: the percentile, a
   bias-corrected estimate (real minus null mean), and a flag for values
   outside the central ``1 - alpha`` null interval. Only flagged values are
   findings; everything else is indistinguishable from estimator bias.

The headline quantity is a per-side *summary functional*, not the raw
gammas: the fitted interaction contribution (points/100) at
high-concentration lineups (top decile of the side's top2 feature).
The three basis features are collinear, so individual gamma coordinates
trade off against each other and their null bands are wide and jumpy; the
contribution the model actually assigns to stacked lineups is what the
coordinates jointly determine, and it is stable. Diminishing returns =
negative offensive score (stacked offenses score less than additive),
positive defensive score (in points-allowed terms). Per-term rows are
still reported for inspection.

Attribution caveat: a flag outside the null band is evidence of real
non-additivity, but the *side* it lands on is only as trustworthy as the
free estimator's credit assignment. The offense and defense gamma blocks
are solved jointly, and when the two sides' concentration features are
correlated across stints, curvature injected on one side can surface on
the other (synthetic demonstration in tests/test_calibrate.py). Treat a
flag as "the additive model is missing something real", and arbitrate the
attribution with replication across seasons and held-out evaluation.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import results_dir, season_type_slug
from .evaluate import predict_stints
from .model import (
    INTERACTION_FEATURES,
    _concentration_features,
    build_design,
    cross_validate_lambda,
    fit_interaction_rapm,
    fit_rapm,
)

logger = logging.getLogger(__name__)

TOP_CONCENTRATION_QUANTILE = 0.9


def _gammas(result) -> dict[str, float]:
    """Flatten a fit's interaction gammas to {'off:top2': g, 'def:top2': g, ...}."""
    inter = result.meta["interaction"]
    out = {}
    for side in ("off", "def"):
        for feat, g in inter[f"gamma_{side}"].items():
            out[f"{side}:{feat}"] = float(g)
    return out


def _curvature_scores(result, stints: pd.DataFrame) -> dict[str, float]:
    """Per-side interaction contribution at high-concentration lineups.

    For each side, evaluate the fit's interaction term F @ gamma on every
    stint row and average it (possession-weighted) over the rows in the top
    decile of that side's standardized top2 feature — the lineups whose two
    best talents are largest together. Units are points/100. Offense < 0
    and defense > 0 (points-allowed terms) mean stacked talent produces
    less than the additive model predicts.
    """
    inter = result.meta["interaction"]
    talent = {int(k): tuple(v) for k, v in inter["talent"].items()}
    w = stints["poss"].to_numpy(dtype=float)
    scores = {}
    for side, idx, col in (("off", 0, "off_lineup"), ("def", 1, "def_lineup")):
        F, _ = _concentration_features(
            stints[col].to_numpy(), talent, idx, w, inter["info"][side]
        )
        gamma = np.array([inter[f"gamma_{side}"][f] for f in inter["features"]])
        contribution = F @ gamma
        top2 = F[:, INTERACTION_FEATURES.index("top2")]
        sel = top2 >= np.quantile(top2, TOP_CONCENTRATION_QUANTILE)
        scores[side] = float(np.average(contribution[sel], weights=w[sel]))
    return scores


def simulate_null_points(
    stints: pd.DataFrame,
    linear_result,
    rng: np.random.Generator,
    null: str = "poisson",
) -> pd.DataFrame:
    """One synthetic copy of the dataset under the fitted additive model.

    Lineups, games, and possession counts are the real ones; only points
    are replaced. Two generators:

    ``poisson``: points ~ Poisson(predicted points/100 x poss / 100).
    Slightly understates real per-possession scoring variance (2s and 3s),
    which makes the null band conservative in the safe direction: real
    noise is wider, so a gamma that clears this band clears an
    easier-than-life null only in dispersion, not in bias — and bias is
    what this calibration exists to measure.

    ``resample``: redraw studentized residuals with replacement. Each row's
    residual (y - mu, in per-100 units) is scaled by sqrt(poss) so rows are
    exchangeable despite different possession counts, pooled, resampled,
    and scaled back — matching the real data's dispersion exactly at the
    cost of assuming residuals are exchangeable after studentization.
    Simulated points are continuous (the fit never needs integers) and
    clipped at zero.
    """
    poss = stints["poss"].to_numpy(dtype=float)
    mu_100 = np.clip(predict_stints(linear_result, stints), 1e-6, None)
    sim = stints.copy()
    if null == "poisson":
        sim["points"] = rng.poisson(mu_100 * poss / 100.0)
    elif null == "resample":
        y = 100.0 * stints["points"].to_numpy(dtype=float) / poss
        studentized = (y - mu_100) * np.sqrt(poss)
        drawn = rng.choice(studentized, size=len(studentized), replace=True)
        y_sim = mu_100 + drawn / np.sqrt(poss)
        sim["points"] = np.clip(y_sim * poss / 100.0, 0.0, None)
    else:
        raise ValueError(f"Unknown null generator {null!r}")
    return sim


def calibrate_curvature(
    stints: pd.DataFrame,
    lam: float | str = "cv",
    n_sims: int = 100,
    seed: int = 0,
    alpha: float = 0.05,
    null: str = "poisson",
    offense_curvature: str = "free",
    defense_curvature: str = "free",
    decay: float = 1.0,
    playoff_weight: float = 1.0,
    garbage_weight: float = 1.0,
) -> dict:
    """Run the null calibration on one set of stints. Returns a report dict.

    ``lam="cv"`` picks lambda once by game-grouped CV and then holds it
    fixed for the real fit and every simulation, so the null distribution
    reflects the estimator actually being calibrated (and the sims don't
    each pay for a CV sweep).
    """
    # zero-possession rows can't be fit, predicted, or scored
    stints = stints[stints["poss"] > 0].reset_index(drop=True)
    # the alpha/2 and 1-alpha/2 quantiles are meaningless with fewer draws
    # than ~2/alpha; findings from a degenerate band would all be spurious
    min_sims = int(np.ceil(2.0 / alpha))
    band_reliable = n_sims >= min_sims
    if not band_reliable:
        logger.warning(
            "n_sims=%d is below ~%d needed for a reliable alpha=%g null "
            "band; outside-null flags are suppressed (bias correction is "
            "still reported)", n_sims, min_sims, alpha,
        )
    fit_kwargs = dict(
        decay=decay, playoff_weight=playoff_weight, garbage_weight=garbage_weight
    )
    if lam == "cv":
        lam, _ = cross_validate_lambda(build_design(stints))
        logger.info("calibration lambda fixed at %g (game-grouped CV)", lam)
    lam = float(lam)

    real = fit_interaction_rapm(
        stints, lam=lam,
        offense_curvature=offense_curvature,
        defense_curvature=defense_curvature,
        **fit_kwargs,
    )
    real_gammas = _gammas(real)
    real_scores = _curvature_scores(real, stints)

    # the additive generator: same linear two-phase fit the interaction
    # model nests, so the null is "this exact model, minus the curvature"
    phase1 = fit_rapm(stints, lam=lam, **fit_kwargs)
    phase1_prior = {
        int(r.player_id): (float(r.orapm), float(r.drapm))
        for r in phase1.players.itertuples()
    }
    generator = fit_rapm(stints, lam=lam, prior=phase1_prior, **fit_kwargs)

    rng = np.random.default_rng(seed)
    null_draws: dict[str, list[float]] = {k: [] for k in real_gammas}
    null_scores: dict[str, list[float]] = {"off": [], "def": []}
    for i in range(n_sims):
        sim = simulate_null_points(stints, generator, rng, null=null)
        sim_fit = fit_interaction_rapm(
            sim, lam=lam,
            offense_curvature=offense_curvature,
            defense_curvature=defense_curvature,
            **fit_kwargs,
        )
        for k, g in _gammas(sim_fit).items():
            null_draws[k].append(g)
        for side, s in _curvature_scores(sim_fit, sim).items():
            null_scores[side].append(s)
        if (i + 1) % 10 == 0 or i + 1 == n_sims:
            logger.info("null simulation %d/%d done", i + 1, n_sims)

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def summarize(real_value: float, draws: np.ndarray, name: str) -> dict:
        lo, hi = float(np.percentile(draws, lo_q)), float(np.percentile(draws, hi_q))
        # mid-p empirical percentile of the real value within the null draws
        pct = float(
            100.0
            * ((draws < real_value).sum() + 0.5 * (draws == real_value).sum())
            / len(draws)
        )
        return {
            name: real_value,
            "null_mean": float(draws.mean()),
            "null_sd": float(draws.std()),
            "null_lo": lo,
            "null_hi": hi,
            "percentile": pct,
            "bias_corrected": real_value - float(draws.mean()),
            "outside_null": bool(
                band_reliable and (real_value < lo or real_value > hi)
            ),
        }

    sides = {
        side: summarize(real_scores[side], np.array(null_scores[side]), "score")
        for side in ("off", "def")
    }
    terms = {
        key: summarize(g_real, np.array(null_draws[key]), "gamma")
        for key, g_real in real_gammas.items()
    }

    return {
        "lambda": lam,
        "n_sims": n_sims,
        "null": null,
        "band_reliable": band_reliable,
        "seed": seed,
        "alpha": alpha,
        "offense_curvature": offense_curvature,
        "defense_curvature": defense_curvature,
        "features": INTERACTION_FEATURES,
        "weights": fit_kwargs,
        "n_games": int(stints["game_id"].nunique()),
        "n_stint_rows": int(len(stints)),
        "sides": sides,
        "terms": terms,
    }


def format_report(report: dict) -> str:
    def row(label, t, value_key):
        band = f"[{t['null_lo']:+.3f}, {t['null_hi']:+.3f}]"
        return (
            f"{label:<14}{t[value_key]:>+9.3f}{t['null_mean']:>+11.3f}{band:>20}"
            f"{t['percentile']:>8.1f}{t['bias_corrected']:>+11.3f}  "
            + ("OUTSIDE NULL" if t["outside_null"] else "within null")
        )

    header = (
        f"{'':<14}{'real':>9}{'null mean':>11}{'null band':>20}"
        f"{'pctile':>8}{'corrected':>11}  finding?"
    )
    lines = [
        f"null calibration: {report['n_sims']} {report.get('null', 'poisson')} "
        f"sims, lambda={report['lambda']:g}, {report['n_games']} games "
        f"(curvature off={report['offense_curvature']} "
        f"def={report['defense_curvature']})",
        "",
        "curvature at stacked lineups (pts/100 at top-decile concentration;"
        " diminishing returns = off < 0, def > 0):",
        header,
        row("off", report["sides"]["off"], "score"),
        row("def", report["sides"]["def"], "score"),
        "",
        "per-term gammas (collinear basis — coordinates trade off; read the"
        " side scores above for the finding):",
        header,
    ]
    for key, t in report["terms"].items():
        lines.append(row(key, t, "gamma"))
    lines.append(
        "only OUTSIDE NULL values are evidence of real curvature beyond "
        "this estimator's own bias on this dataset"
    )
    if not report.get("band_reliable", True):
        lines.append(
            f"WARNING: {report['n_sims']} sims is too few for a reliable "
            f"alpha={report['alpha']:g} null band — outside-null flags are "
            "suppressed; run with more --sims"
        )
    return "\n".join(lines)


def run_calibration(
    data_dir: Path,
    seasons: list[str],
    season_types: list[str],
    out_dir: Path | None = None,
    **kwargs,
) -> tuple[dict, Path]:
    """Load stints for the scope, calibrate, and write the report JSON."""
    from .model import load_stints

    stints = load_stints(data_dir, seasons, season_types)
    report = calibrate_curvature(stints, **kwargs)
    report["seasons"] = seasons
    report["season_types"] = season_types

    out_dir = out_dir or results_dir(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    season_label = (
        seasons[0] if len(seasons) == 1 else f"{seasons[0]}_to_{seasons[-1]}"
    )
    type_label = "_".join(season_type_slug(t) for t in season_types)
    path = (
        out_dir
        / f"curvature_calibration_{season_label.replace('-', '_')}_{type_label}.json"
    )
    path.write_text(json.dumps(report, indent=1))
    logger.info("wrote %s", path)
    return report, path
