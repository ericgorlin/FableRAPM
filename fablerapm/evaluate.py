"""Holdout evaluation: compare RAPM variants on games the fit never saw.

Games (not stints) are split into train/holdout so all of a game's rows stay
on one side. Each variant fits on the train games — priors are computed from
train data only, so there is no leakage — and is scored by
possession-weighted MSE predicting held-out stint scoring rates.

Two split modes:
    random  independent shuffled game holdouts (seeded)
    chrono  forward-chaining: hold out the *latest* games, train only on
            earlier ones — the honest setup when the model's job is to rate
            players now or predict upcoming games
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

from .config import season_end_year
from .model import RapmResult, fit_interaction_rapm, fit_rapm

logger = logging.getLogger(__name__)

PriorFn = Callable[[pd.DataFrame], dict[int, tuple[float, float]] | None]
# a variant is either a prior fn (or None), or a dict:
#   {"prior_fn": PriorFn | None, "interactions": bool}

_TYPE_ORDER = {"Regular Season": 0, "Play In": 1, "Playoffs": 2}


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


def chrono_game_order(stints: pd.DataFrame) -> list:
    """Game ids in (approximate) chronological order.

    NBA game ids are assigned in schedule order within a season and season
    type, so sorting by (season end year, regular < play-in < playoffs,
    game_id) recovers the season's timeline without needing game dates.
    Approximation: within a season type, ids follow the *scheduled* order,
    so postponed/rescheduled games can be slightly out of true date order.
    """
    per_game = stints.drop_duplicates("game_id")
    keys = {}
    for row in per_game.itertuples():
        year = season_end_year(row.season) if hasattr(row, "season") else 0
        type_rank = (
            _TYPE_ORDER.get(row.season_type, 3)
            if hasattr(row, "season_type") else 0
        )
        keys[row.game_id] = (year, type_rank, str(row.game_id))
    return sorted(keys, key=keys.__getitem__)


def chrono_split(
    stints: pd.DataFrame,
    test_frac: float = 0.2,
    fold: int = 0,
    n_folds: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward-chaining split: test on a late block, train on everything
    strictly before it.

    With ``n_folds`` > 1, fold ``i`` (0-based) tests the i-th of the last
    ``n_folds`` blocks of ``test_frac`` games each — e.g. n_folds=3,
    test_frac=0.1 gives train [0,70%) / test [70,80%), train [0,80%) /
    test [80,90%), train [0,90%) / test [90,100%). Later rows never leak
    into training.
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold {fold} out of range for n_folds={n_folds}")
    games = chrono_game_order(stints)
    n = len(games)
    block = max(1, int(n * test_frac))
    test_end = n - (n_folds - 1 - fold) * block
    test_start = test_end - block
    if test_start <= 0:
        raise ValueError(
            f"chrono split needs more games: {n} games, {n_folds} folds "
            f"of {block} test games each leave no training data"
        )
    test_games = set(games[test_start:test_end])
    train_games = set(games[:test_start])
    return (
        stints[stints["game_id"].isin(train_games)].reset_index(drop=True),
        stints[stints["game_id"].isin(test_games)].reset_index(drop=True),
    )


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
    if "interaction" in result.meta:
        from .model import _concentration_features

        inter = result.meta["interaction"]
        talent = {int(k): tuple(v) for k, v in inter["talent"].items()}
        w = stints["poss"].to_numpy(dtype=float)
        F_off, _ = _concentration_features(
            off_lineups, talent, 0, w, inter["info"]["off"]
        )
        F_def, _ = _concentration_features(
            def_lineups, talent, 1, w, inter["info"]["def"]
        )
        g_off = np.array([inter["gamma_off"][f] for f in inter["features"]])
        g_def = np.array([inter["gamma_def"][f] for f in inter["features"]])
        preds += F_off @ g_off + F_def @ g_def
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
    split: str = "random",
) -> pd.DataFrame:
    """Fit each variant on train games, score on holdout games.

    Includes an intercept-only baseline. Lower holdout_mse is better; the
    interesting quantity is the gap each variant closes vs the baseline.

    ``score_garbage=False`` drops garbage-time rows from the holdout metric
    (use when tuning garbage_weight, so the target is non-garbage scoring).

    ``split="chrono"`` holds out the latest ``test_frac`` games instead of a
    random sample (``seed`` is then ignored) — see ``chrono_split``.
    """
    if split == "chrono":
        train, test = chrono_split(stints, test_frac=test_frac)
    else:
        train, test = holdout_split(stints, test_frac=test_frac, seed=seed)
    test = test[test["poss"] > 0]
    if not score_garbage and "garbage" in test:
        test = test[~test["garbage"].astype(bool)]
    logger.info(
        "Evaluation: %d train games, %d holdout games (%d holdout rows)",
        train["game_id"].nunique(), test["game_id"].nunique(), len(test),
    )

    # baseline mean from TRAIN games, like every other variant
    train_nz = train[train["poss"] > 0]
    y_train = 100.0 * train_nz["points"].to_numpy() / train_nz["poss"].to_numpy()
    baseline_pred = np.full(len(test), np.average(y_train, weights=train_nz["poss"]))
    rows = [{
        "variant": "intercept-only",
        "lambda": np.nan,
        "holdout_mse": _weighted_mse(test, baseline_pred),
    }]

    for name, spec in variants.items():
        if not isinstance(spec, dict):
            spec = {"prior_fn": spec, "interactions": False}
        prior_fn = spec.get("prior_fn")
        prior = prior_fn(train) if prior_fn is not None else None
        fit = fit_interaction_rapm if spec.get("interactions") else fit_rapm
        result = fit(
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
