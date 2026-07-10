"""Learn every tunable free parameter from data — with an honest scorecard.

Greedy coordinate descent over the pipeline's free parameters, scored by
possession-weighted MSE on held-out games averaged over several inner
splits (multi-split averaging is what keeps the search from chasing stint
noise — gaps between reasonable settings are small). The winner is written
to ``data/results/tuned_config.json``; ``fablerapm rapm --tuned`` fits with
it.

Nested evaluation: before any tuning, an *outer* test block of games is
set aside and never touched by the search. The inner splits (which the
descent selects on) come from the remaining development games only, as
does the lambda CV. After the search, the winning config and the untouched
default config are each fit on the full development set and scored once on
the outer block — ``outer_mse`` vs ``outer_default_mse`` is the unbiased
report of what tuning bought; the inner selection MSE is only a ranking
signal and is optimistic as an estimate (it chose the winner).

Split modes: ``chrono`` (default) orders games by season/type/game-id —
approximate schedule order — takes the latest block as the outer test and
forward-chains the inner splits (train strictly earlier than test), which
is the right protocol when the model's job is current ratings or upcoming
games. ``random`` reproduces the old seeded game shuffles.

Parameters searched (grids skipped automatically when not applicable):
    garbage weight x rule (parse-time flag or fit-time linear threshold,
    the latter only when the stint schema carries margin/secs_left); decay
    (multi-season data only); playoff_weight (mixed season types only);
    prior kind + scale (none / two-phase / last-season, last-season only
    when the previous season is built); interactions on/off and the
    defense curvature mode; ridge lambda over a multiplier grid around the
    CV pick, searched *last* so it re-tunes for whichever model family the
    descent has settled on (use ``--passes 2`` to let the family choice
    react to the re-tuned lambda in turn).

Not searched, by design: parse-time definitions (possession attribution),
the interaction feature basis, and the offensive sign constraint — see
README.
"""

import json
import logging
from pathlib import Path

import numpy as np

from .evaluate import _weighted_mse, chrono_split, holdout_split, predict_stints
from .model import cross_validate_lambda, build_design, fit_interaction_rapm, fit_rapm

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "lambda": None,
    "garbage_weight": 1.0,
    "garbage_rule": None,
    "decay": 1.0,
    "playoff_weight": 1.0,
    "prior": "none",
    "prior_scale": 1.0,
    "interactions": False,
    "defense_curvature": "diminishing",
    "offense_curvature": "diminishing",
}

LAMBDA_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]

# fit-time garbage thresholds |margin| >= base + per_minute * minutes_left,
# searched only when the stint schema carries margin/secs_left (see
# model.garbage_flags); None = the parse-time GARBAGE_TIERS flag
GARBAGE_RULES = [[8.0, 1.0], [12.0, 1.5], [16.0, 2.0]]


def tuned_config_path(data_dir: Path) -> Path:
    return data_dir / "results" / "tuned_config.json"


def _fit(train, cfg, data_dir, seasons, season_types):
    from .prior import last_season_prior, two_phase_prior

    lam = cfg["lambda"]
    rule = cfg.get("garbage_rule")
    kw = dict(
        lam=lam,
        decay=cfg["decay"],
        playoff_weight=cfg["playoff_weight"],
        garbage_weight=cfg["garbage_weight"],
        garbage_rule=tuple(rule) if rule else None,
    )
    prior = None
    if cfg["prior"] == "two-phase":
        prior = two_phase_prior(train, lam=lam, scale=cfg["prior_scale"])
    elif cfg["prior"] == "last-season":
        prior = last_season_prior(
            data_dir, seasons, season_types, lam=lam, scale=cfg["prior_scale"]
        )
    if cfg["interactions"]:
        return fit_interaction_rapm(
            train, prior=prior,
            offense_curvature=cfg["offense_curvature"],
            defense_curvature=cfg["defense_curvature"],
            **kw,
        )
    return fit_rapm(train, prior=prior, **kw)


def _score_split(train, test, cfg, data_dir, seasons, season_types) -> float:
    result = _fit(train, cfg, data_dir, seasons, season_types)
    return _weighted_mse(test, predict_stints(result, test))


def _score(cfg, splits, data_dir, seasons, season_types) -> float:
    return float(np.mean([
        _score_split(train, test, cfg, data_dir, seasons, season_types)
        for train, test in splits
    ]))


def _candidate_grids(stints, data_dir, seasons, season_types, base_lambda) -> list[tuple]:
    """(param updates applied together) grouped per coordinate."""
    grids = []
    if "garbage" in stints:
        garbage_options = [
            {"garbage_weight": v, "garbage_rule": None} for v in (1.0, 0.5, 0.0)
        ]
        # fit-time thresholds only matter at weight != 1, and only when the
        # schema carries the late-game context
        if {"margin", "secs_left"} <= set(stints.columns) and (
            stints["secs_left"].notna().any()
        ):
            garbage_options += [
                {"garbage_weight": w, "garbage_rule": rule}
                for w in (0.5, 0.0)
                for rule in GARBAGE_RULES
            ]
        grids.append(("garbage", garbage_options))
    if "season" in stints and stints["season"].nunique() > 1:
        grids.append(("decay", [{"decay": v} for v in (1.0, 0.95, 0.9, 0.8)]))
    if "season_type" in stints and stints["season_type"].nunique() > 1:
        grids.append(("playoff_weight", [{"playoff_weight": v} for v in (1.0, 1.5, 2.0)]))
    prior_options = [
        {"prior": "none", "prior_scale": 1.0},
        {"prior": "two-phase", "prior_scale": 0.5},
        {"prior": "two-phase", "prior_scale": 1.0},
    ]
    try:  # last-season prior needs the previous season's stints on disk
        from .prior import last_season_prior

        last_season_prior(data_dir, seasons, season_types, lam=1000.0, scale=0.7)
        prior_options += [
            {"prior": "last-season", "prior_scale": 0.5},
            {"prior": "last-season", "prior_scale": 0.7},
        ]
    except FileNotFoundError:
        logger.info("previous season not built; skipping last-season prior")
    grids.append(("prior", prior_options))
    grids.append((
        "interactions",
        [
            {"interactions": False},
            {"interactions": True, "defense_curvature": "diminishing"},
            {"interactions": True, "defense_curvature": "free"},
        ],
    ))
    # lambda last, so it re-tunes for the model family chosen above — a
    # two-phase or interaction winner must beat plain ridge at *its* best
    # lambda too, not just at the baseline's
    grids.append((
        "lambda",
        [{"lambda": base_lambda * m} for m in LAMBDA_MULTIPLIERS],
    ))
    return grids


def _make_splits(dev, split: str, n_seeds: int, test_frac: float) -> list[tuple]:
    if split == "chrono":
        raw = [
            chrono_split(dev, test_frac=test_frac, fold=i, n_folds=n_seeds)
            for i in range(n_seeds)
        ]
    else:
        raw = [holdout_split(dev, test_frac, seed) for seed in range(n_seeds)]
    # zero-possession rows (orphaned technical FTs) can't be scored
    return [
        (train, test[test["poss"] > 0].reset_index(drop=True))
        for train, test in raw
    ]


def tune(
    data_dir: Path,
    seasons: list[str],
    season_types: list[str],
    n_seeds: int = 3,
    test_frac: float = 0.2,
    passes: int = 1,
    lam: float | str = "cv",
    split: str = "chrono",
    outer_frac: float = 0.2,
) -> dict:
    from .model import load_stints

    stints = load_stints(data_dir, seasons, season_types)

    # outer split first: the search never sees these games
    if outer_frac > 0:
        if split == "chrono":
            dev, outer_test = chrono_split(stints, test_frac=outer_frac)
        else:
            dev, outer_test = holdout_split(stints, outer_frac, seed=10_000)
        outer_test = outer_test[outer_test["poss"] > 0].reset_index(drop=True)
        logger.info(
            "outer test: %d games held out untouched (%s split); "
            "tuning on the remaining %d",
            outer_test["game_id"].nunique(), split, dev["game_id"].nunique(),
        )
    else:
        dev, outer_test = stints, None

    if lam == "cv":
        lam, _ = cross_validate_lambda(build_design(dev))
    cfg = dict(DEFAULT_CONFIG, **{"lambda": float(lam)})
    base_lambda = float(lam)
    splits = _make_splits(dev, split, n_seeds, test_frac)

    history = []
    best = _score(cfg, splits, data_dir, seasons, season_types)
    history.append({"config": dict(cfg), "mse": best, "note": "baseline"})
    logger.info("baseline inner MSE %.4f (lambda=%g)", best, lam)

    for _ in range(passes):
        for name, options in _candidate_grids(
            data_dir=data_dir, stints=dev, seasons=seasons,
            season_types=season_types, base_lambda=base_lambda,
        ):
            for update in options:
                candidate = dict(cfg, **update)
                if candidate == cfg:
                    continue
                try:
                    mse = _score(candidate, splits, data_dir, seasons, season_types)
                except Exception as exc:
                    logger.warning("%s candidate %s failed: %s", name, update, exc)
                    continue
                history.append({"config": dict(candidate), "mse": mse, "note": name})
                logger.info("%s %s: %.4f (best %.4f)", name, update, mse, best)
                if mse < best:
                    best, cfg = mse, candidate

    out = dict(cfg)
    out["inner_mse"] = best
    out["holdout_mse"] = best  # legacy name for the inner selection score
    out["split"] = split
    out["outer_frac"] = outer_frac
    out["seasons"] = seasons
    out["season_types"] = season_types
    out["n_seeds"] = n_seeds

    if outer_test is not None:
        # one shot each on the untouched games: the winner and the default,
        # both fit on the full development set. outer_mse is the unbiased
        # estimate; outer_mse - outer_default_mse is what tuning bought.
        default_cfg = dict(DEFAULT_CONFIG, **{"lambda": base_lambda})
        out["outer_mse"] = _score_split(
            dev, outer_test, cfg, data_dir, seasons, season_types
        )
        out["outer_default_mse"] = _score_split(
            dev, outer_test, default_cfg, data_dir, seasons, season_types
        )
        dev_nz = dev[dev["poss"] > 0]
        y_dev = 100.0 * dev_nz["points"].to_numpy() / dev_nz["poss"].to_numpy()
        intercept = np.full(
            len(outer_test), np.average(y_dev, weights=dev_nz["poss"])
        )
        out["outer_intercept_mse"] = _weighted_mse(outer_test, intercept)
        logger.info(
            "outer (untouched) MSE: tuned %.4f vs default %.4f vs "
            "intercept-only %.4f",
            out["outer_mse"], out["outer_default_mse"], out["outer_intercept_mse"],
        )

    out["history"] = history
    path = tuned_config_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    logger.info("wrote %s (inner MSE %.4f)", path, best)
    return out


def load_tuned_config(data_dir: Path) -> dict:
    path = tuned_config_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"No tuned config at {path}; run `fablerapm tune` first")
    cfg = json.loads(path.read_text())
    # .get: configs written before a key existed fall back to its default
    return {k: cfg.get(k, default) for k, default in DEFAULT_CONFIG.items()}
