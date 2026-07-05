"""Holdout evaluation: compare RAPM variants on games the fit never saw.

Games (not stints) are split into train/holdout so all of a game's rows stay
on one side. Each variant fits on the train games — priors are computed from
train data only, so there is no leakage — and is scored by
possession-weighted MSE predicting held-out stint scoring rates.
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

from .model import RapmResult, fit_rapm

logger = logging.getLogger(__name__)

PriorFn = Callable[[pd.DataFrame], dict[int, tuple[float, float]] | None]


def holdout_split(
    stints: pd.DataFrame, test_frac: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = np.array(sorted(stints["game_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_test = max(1, int(len(games) * test_frac))
    test_games = set(games[:n_test])
    mask = stints["game_id"].isin(test_games)
    return stints[~mask].reset_index(drop=True), stints[mask].reset_index(drop=True)


def predict_stints(result: RapmResult, stints: pd.DataFrame) -> np.ndarray:
    """Predicted points/100 for each stint row; unseen players contribute 0."""
    o = dict(zip(result.players["player_id"], result.players["orapm"]))
    d = dict(zip(result.players["player_id"], result.players["drapm"]))
    intercept = result.meta["intercept"]
    preds = np.empty(len(stints))
    off_lineups = stints["off_lineup"].to_numpy()
    def_lineups = stints["def_lineup"].to_numpy()
    for i in range(len(stints)):
        preds[i] = (
            intercept
            + sum(o.get(int(p), 0.0) for p in off_lineups[i].split("-"))
            - sum(d.get(int(p), 0.0) for p in def_lineups[i].split("-"))
        )
    return preds


def _weighted_mse(stints: pd.DataFrame, preds: np.ndarray) -> float:
    y = 100.0 * stints["points"].to_numpy() / stints["poss"].to_numpy()
    return float(np.average((y - preds) ** 2, weights=stints["poss"]))


def evaluate_variants(
    stints: pd.DataFrame,
    variants: dict[str, PriorFn | None],
    lam: float | str = "cv",
    test_frac: float = 0.2,
    seed: int = 0,
    decay: float = 1.0,
    playoff_weight: float = 1.0,
    garbage_weight: float = 1.0,
    score_garbage: bool = True,
) -> pd.DataFrame:
    """Fit each variant on train games, score on holdout games.

    Includes an intercept-only baseline. Lower holdout_mse is better; the
    interesting quantity is the gap each variant closes vs the baseline.

    ``score_garbage=False`` drops garbage-time rows from the holdout metric
    (use when tuning garbage_weight, so the target is non-garbage scoring).
    """
    train, test = holdout_split(stints, test_frac=test_frac, seed=seed)
    test = test[test["poss"] > 0]
    if not score_garbage and "garbage" in test:
        test = test[~test["garbage"].astype(bool)]
    logger.info(
        "Evaluation: %d train games, %d holdout games (%d holdout rows)",
        train["game_id"].nunique(), test["game_id"].nunique(), len(test),
    )

    y = 100.0 * test["points"].to_numpy() / test["poss"].to_numpy()
    baseline_pred = np.full(len(test), np.average(y, weights=test["poss"]))
    rows = [{
        "variant": "intercept-only",
        "lambda": np.nan,
        "holdout_mse": _weighted_mse(test, baseline_pred),
    }]

    for name, prior_fn in variants.items():
        prior = prior_fn(train) if prior_fn is not None else None
        result = fit_rapm(
            train, lam=lam, prior=prior,
            decay=decay, playoff_weight=playoff_weight,
            garbage_weight=garbage_weight,
        )
        rows.append({
            "variant": name,
            "lambda": result.meta["lambda"],
            "holdout_mse": _weighted_mse(test, predict_stints(result, test)),
        })
        logger.info("%s: holdout MSE %.4f", name, rows[-1]["holdout_mse"])

    table = pd.DataFrame(rows)
    base = table.loc[0, "holdout_mse"]
    table["vs_baseline"] = base - table["holdout_mse"]
    return table
