"""RAPM model tests on synthetic data with known player effects (offline)."""

import numpy as np
import pandas as pd
import pytest

from fablerapm.model import build_design, fit_rapm

rng = np.random.default_rng(7)

N_PLAYERS = 60
BASE_PER_POSS = 1.10  # league average points per possession


def make_league():
    """Players with true offensive/defensive effects (per 1 possession)."""
    true_off = rng.normal(0, 0.02, N_PLAYERS)
    true_def = rng.normal(0, 0.015, N_PLAYERS)  # positive = allows more
    return true_off, true_def


def simulate_stints(true_off, true_def, n_games=400, stints_per_game=25, seed=1):
    """Random lineups; points drawn from the true model, so RAPM should
    recover the effects up to shrinkage."""
    r = np.random.default_rng(seed)
    rows = []
    players = np.arange(N_PLAYERS)
    for g in range(n_games):
        for _ in range(stints_per_game):
            ten = r.choice(players, size=10, replace=False)
            off, dfn = ten[:5], ten[5:]
            poss = int(r.integers(2, 15))
            rate = BASE_PER_POSS + true_off[off].sum() + true_def[dfn].sum()
            points = r.poisson(max(rate, 0.05) * poss)
            rows.append(
                {
                    "game_id": f"G{g:05d}",
                    "off_team_id": 1,
                    "def_team_id": 2,
                    "off_lineup": "-".join(str(p + 1) for p in sorted(off)),
                    "def_lineup": "-".join(str(p + 1) for p in sorted(dfn)),
                    "poss": poss,
                    "points": int(points),
                }
            )
    return pd.DataFrame(rows)


def test_design_matrix_shape_and_counts():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=5)
    design = build_design(stints)
    assert design.X.shape == (len(stints), 2 * len(design.player_ids))
    assert (np.asarray(design.X.sum(axis=1)).ravel() == 10).all()
    # possession counts match direct aggregation for a random player
    pid = design.player_ids[0]
    tag = str(pid)
    expected_off = stints[
        stints["off_lineup"].str.split("-").apply(lambda l: tag in l)
    ]["poss"].sum()
    assert design.off_poss[0] == expected_off


def test_recovers_true_effects():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=1000)
    result = fit_rapm(stints, lam=200.0)
    players = result.players.set_index("player_id").sort_index()
    est_o = players["orapm"].to_numpy()
    est_d = players["drapm"].to_numpy()
    # per-100 truth; DRAPM output flips sign so positive = good defense
    corr_o = np.corrcoef(est_o, 100 * true_off)[0, 1]
    corr_d = np.corrcoef(est_d, -100 * true_def)[0, 1]
    assert corr_o > 0.85, f"ORAPM correlation too low: {corr_o:.3f}"
    assert corr_d > 0.70, f"DRAPM correlation too low: {corr_d:.3f}"
    assert np.allclose(
        players["rapm"], players["orapm"] + players["drapm"]
    )
    assert 100 < result.meta["avg_points_per_100"] < 120


def test_cv_lambda_selection_runs():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=60)
    result = fit_rapm(stints, lam="cv", lambdas=[50, 500, 5000], n_folds=3)
    assert result.meta["lambda"] in (50, 500, 5000)
    assert len(result.meta["cv"]) == 3
    assert all(np.isfinite(r["cv_mse"]) for r in result.meta["cv"])


def test_prior_shifts_estimates_toward_prior():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=40)
    base = fit_rapm(stints, lam=1000.0).players.set_index("player_id")
    pid = int(base.index[0])
    prior = {pid: (5.0, 5.0)}
    shifted = fit_rapm(stints, lam=1000.0, prior=prior).players.set_index("player_id")
    assert shifted.loc[pid, "orapm"] > base.loc[pid, "orapm"]
    assert shifted.loc[pid, "drapm"] > base.loc[pid, "drapm"]
    # zero prior is a no-op
    zero = fit_rapm(
        stints, lam=1000.0, prior={pid: (0.0, 0.0)}
    ).players.set_index("player_id")
    assert np.allclose(zero["rapm"], base["rapm"], atol=1e-6)


def test_zero_possession_rows_dropped():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=5)
    orphan = stints.iloc[[0]].copy()
    orphan["poss"] = 0
    orphan["points"] = 1
    design = build_design(pd.concat([stints, orphan], ignore_index=True))
    assert design.dropped_points == 1
    assert design.X.shape[0] == len(stints)
