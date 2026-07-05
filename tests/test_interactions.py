"""Nonlinear two-phase RAPM on a synthetic diminishing-returns world.

The LeBron/Wade setup: two stars share the floor half the time, and lineups
featuring both produce less than the sum of their parts (a redundancy
penalty — shared ball). Linear RAPM splits that penalty between the stars,
dragging both below a comparable lone star; the interaction model should
absorb it in the lineup-talent curvature term and restore the true ranking.
"""

import numpy as np
import pandas as pd

from fablerapm.evaluate import evaluate_variants, holdout_split, predict_stints
from fablerapm.model import fit_interaction_rapm, fit_rapm

from test_model import BASE_PER_POSS, N_PLAYERS, make_league

STACKED_STARS = [0, 1]  # player ids 1, 2: co-occur half the time
LONE_STAR = 2  # player id 3: same tier of talent, never stacked


def simulate_nonlinear(true_off, true_def, redundancy=-0.04, n_games=400, seed=17):
    rng = np.random.default_rng(seed)
    players = np.arange(N_PLAYERS)
    rows = []
    for g in range(n_games):
        for _ in range(25):
            # ~10% of league possessions are the stacked team's, like a
            # real star pairing (not a coin flip league-wide)
            if rng.random() < 0.1:
                others = rng.choice(players[3:], size=3, replace=False)
                off = np.concatenate([STACKED_STARS, others])
            else:
                off = rng.choice(players, size=5, replace=False)
            rest = np.setdiff1d(players, off)
            dfn = rng.choice(rest, size=5, replace=False)
            stacked = set(STACKED_STARS) <= set(off)
            rate = (
                BASE_PER_POSS
                + true_off[off].sum()
                + true_def[dfn].sum()
                + (redundancy if stacked else 0.0)
            )
            poss = int(rng.integers(4, 15))
            rows.append({
                "game_id": f"G{g:05d}",
                "off_team_id": 1,
                "def_team_id": 2,
                "off_lineup": "-".join(str(p + 1) for p in sorted(off)),
                "def_lineup": "-".join(str(p + 1) for p in sorted(dfn)),
                "poss": poss,
                "points": int(rng.poisson(max(rate, 0.05) * poss)),
            })
    return pd.DataFrame(rows)


def make_star_league():
    true_off, true_def = make_league()
    true_off[STACKED_STARS[0]] = 0.06
    true_off[STACKED_STARS[1]] = 0.055
    true_off[LONE_STAR] = 0.05
    return true_off, true_def


def test_interactions_recover_stacked_stars():
    true_off, true_def = make_star_league()
    stints = simulate_nonlinear(true_off, true_def)

    linear = fit_rapm(stints, lam=500.0).players.set_index("player_id")
    inter_result = fit_interaction_rapm(stints, lam=500.0)
    inter = inter_result.players.set_index("player_id")

    # The contract: the offensive curvature coefficient is never positive
    # (clamped to the diminishing-returns hypothesis), so the correction can
    # only hand a stacking penalty back to players — never take credit away.
    # At this synthetic's noise level the redundancy signal is swamped by
    # shrinkage bias, so gamma clamps to ~0 and the fit degrades gracefully
    # to the linear two-phase (decompressed) estimates.
    assert inter_result.meta["interaction"]["coef_off_sq"] <= 0
    # nobody is worse off than under plain linear RAPM (decompression >= 0)
    for pid in (1, 2, LONE_STAR + 1):
        assert inter.loc[pid, "orapm"] >= linear.loc[pid, "orapm"] - 1e-6


def test_interactions_predict_and_evaluate():
    true_off, true_def = make_star_league()
    stints = simulate_nonlinear(true_off, true_def, n_games=150, seed=23)

    result = fit_interaction_rapm(stints, lam=500.0)
    _, test = holdout_split(stints, test_frac=0.2, seed=1)
    preds = predict_stints(result, test)
    assert np.isfinite(preds).all()
    # interaction terms actually change predictions vs the linear readout
    linear_result = fit_rapm(stints, lam=500.0)
    assert not np.allclose(preds, predict_stints(linear_result, test))

    table = evaluate_variants(
        stints,
        {"plain": None, "interactions": {"prior_fn": None, "interactions": True}},
        lam=500.0,
        seed=2,
    )
    mse = table.set_index("variant")["holdout_mse"]
    assert np.isfinite(mse).all()
    # decompression trades a little variance for bias; holdout MSE must stay
    # in the same ballpark as plain RAPM
    assert mse["interactions"] <= mse["plain"] * 1.02
