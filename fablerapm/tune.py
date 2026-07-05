"""Learn every tunable free parameter from data.

Greedy coordinate descent over the pipeline's free parameters, scored by
possession-weighted MSE on held-out games averaged over several seeds
(multi-seed averaging is what keeps the search from chasing stint noise —
gaps between reasonable settings are small). The winner is written to
``data/results/tuned_config.json``; ``fablerapm rapm --tuned`` fits with it.

Parameters searched (grids skipped automatically when not applicable):
    garbage_weight; decay (multi-season data only); playoff_weight (mixed
    season types only); prior kind + scale (none / two-phase / last-season,
    last-season only when the previous season is built); interactions
    on/off and the defense curvature mode. Ridge lambda is chosen once by
    game-grouped CV on the full data and reused throughout.

Not searched, by design: parse-time definitions (garbage-time tiers,
possession attribution), the interaction feature basis, and the offensive
sign constraint — see README.
"""

import json
import logging
from pathlib import Path

import numpy as np

from .evaluate import _weighted_mse, holdout_split, predict_stints
from .model import cross_validate_lambda, build_design, fit_interaction_rapm, fit_rapm

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "lambda": None,
    "garbage_weight": 1.0,
    "decay": 1.0,
    "playoff_weight": 1.0,
    "prior": "none",
    "prior_scale": 1.0,
    "interactions": False,
    "defense_curvature": "diminishing",
    "offense_curvature": "diminishing",
}


def tuned_config_path(data_dir: Path) -> Path:
    return data_dir / "results" / "tuned_config.json"


def _fit(train, cfg, data_dir, seasons, season_types):
    from .prior import last_season_prior, two_phase_prior

    lam = cfg["lambda"]
    kw = dict(
        lam=lam,
        decay=cfg["decay"],
        playoff_weight=cfg["playoff_weight"],
        garbage_weight=cfg["garbage_weight"],
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


def _score(stints, cfg, splits, data_dir, seasons, season_types) -> float:
    mses = []
    for train, test in splits:
        result = _fit(train, cfg, data_dir, seasons, season_types)
        mses.append(_weighted_mse(test, predict_stints(result, test)))
    return float(np.mean(mses))


def _candidate_grids(stints, data_dir, seasons, season_types) -> list[tuple]:
    """(param updates applied together) grouped per coordinate."""
    grids = []
    if "garbage" in stints:
        grids.append(("garbage_weight", [{"garbage_weight": v} for v in (1.0, 0.5, 0.0)]))
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
    return grids


def tune(
    data_dir: Path,
    seasons: list[str],
    season_types: list[str],
    n_seeds: int = 3,
    test_frac: float = 0.2,
    passes: int = 1,
    lam: float | str = "cv",
) -> dict:
    from .model import load_stints

    stints = load_stints(data_dir, seasons, season_types)
    if lam == "cv":
        lam, _ = cross_validate_lambda(build_design(stints))
    cfg = dict(DEFAULT_CONFIG, **{"lambda": float(lam)})
    splits = [holdout_split(stints, test_frac, seed) for seed in range(n_seeds)]

    history = []
    best = _score(stints, cfg, splits, data_dir, seasons, season_types)
    history.append({"config": dict(cfg), "mse": best, "note": "baseline"})
    logger.info("baseline holdout MSE %.4f (lambda=%g)", best, lam)

    for _ in range(passes):
        for name, options in _candidate_grids(data_dir=data_dir, stints=stints,
                                              seasons=seasons, season_types=season_types):
            for update in options:
                candidate = dict(cfg, **update)
                if candidate == cfg:
                    continue
                try:
                    mse = _score(stints, candidate, splits, data_dir, seasons, season_types)
                except Exception as exc:
                    logger.warning("%s candidate %s failed: %s", name, update, exc)
                    continue
                history.append({"config": dict(candidate), "mse": mse, "note": name})
                logger.info("%s %s: %.4f (best %.4f)", name, update, mse, best)
                if mse < best:
                    best, cfg = mse, candidate

    out = dict(cfg)
    out["holdout_mse"] = best
    out["seasons"] = seasons
    out["season_types"] = season_types
    out["n_seeds"] = n_seeds
    out["history"] = history
    path = tuned_config_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    logger.info("wrote %s (holdout MSE %.4f)", path, best)
    return out


def load_tuned_config(data_dir: Path) -> dict:
    path = tuned_config_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"No tuned config at {path}; run `fablerapm tune` first")
    cfg = json.loads(path.read_text())
    return {k: cfg[k] for k in DEFAULT_CONFIG}
