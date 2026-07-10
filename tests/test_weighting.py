"""Decay, playoff, and garbage-time weighting in fit_rapm (offline)."""

import numpy as np
import pandas as pd
import pytest

from fablerapm.model import fit_rapm

from test_model import make_league, simulate_stints


def two_season_stints(true_off_old, true_off_new, true_def, n_games=150):
    old = simulate_stints(true_off_old, true_def, n_games=n_games, seed=21)
    old["game_id"] = "A" + old["game_id"]
    old["season"] = "2023-24"
    new = simulate_stints(true_off_new, true_def, n_games=n_games, seed=22)
    new["game_id"] = "B" + new["game_id"]
    new["season"] = "2024-25"
    return pd.concat([old, new], ignore_index=True)


def test_decay_pulls_toward_recent_season():
    true_off, true_def = make_league()
    true_off_new = true_off.copy()
    true_off[0], true_off_new[0] = -0.02, 0.06  # player 1 improved a lot
    stints = two_season_stints(true_off, true_off_new, true_def)

    flat = fit_rapm(stints, lam=500.0).players.set_index("player_id")
    decayed = fit_rapm(stints, lam=500.0, decay=0.1).players.set_index("player_id")
    recent = fit_rapm(
        stints[stints["season"] == "2024-25"], lam=500.0
    ).players.set_index("player_id")
    # with heavy decay the pooled fit approaches the recent-season-only fit
    assert abs(decayed.loc[1, "orapm"] - recent.loc[1, "orapm"]) < abs(
        flat.loc[1, "orapm"] - recent.loc[1, "orapm"]
    )
    # raw possession counts are not affected by weighting
    assert np.allclose(
        decayed["off_poss"].sort_index(), flat["off_poss"].sort_index()
    )
    assert fit_rapm(stints, lam=500.0, decay=1.0).meta["decay"] == 1.0


def test_decay_requires_season_column():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=10)
    with pytest.raises(ValueError, match="season"):
        fit_rapm(stints, lam=500.0, decay=0.5)


def test_playoff_weight_zero_equals_dropping_playoffs():
    true_off, true_def = make_league()
    stints = two_season_stints(true_off, true_off, true_def, n_games=60)
    stints["season_type"] = np.where(
        stints["season"] == "2024-25", "Playoffs", "Regular Season"
    )
    weighted = fit_rapm(stints, lam=500.0, playoff_weight=0.0).players
    rs_only = fit_rapm(
        stints[stints["season_type"] == "Regular Season"], lam=500.0
    ).players
    merged = weighted.merge(rs_only, on="player_id", suffixes=("_w", "_rs"))
    assert np.allclose(merged["rapm_w"], merged["rapm_rs"], atol=1e-6)


def test_playoff_weight_noop_on_single_type_fit():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=30)
    stints["season_type"] = "Playoffs"
    plain = fit_rapm(stints, lam=500.0).players
    upweighted = fit_rapm(stints, lam=500.0, playoff_weight=2.0).players
    assert np.allclose(plain["rapm"], upweighted["rapm"], atol=1e-6)


def test_garbage_weight_zero_equals_filtering():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=80)
    rng = np.random.default_rng(0)
    stints["garbage"] = rng.random(len(stints)) < 0.1

    dropped = fit_rapm(stints, lam=500.0, garbage_weight=0.0).players
    filtered = fit_rapm(stints[~stints["garbage"]], lam=500.0).players
    merged = dropped.merge(filtered, on="player_id", suffixes=("_d", "_f"))
    assert np.allclose(merged["rapm_d"], merged["rapm_f"], atol=1e-6)

    # downweighting lands between keeping and dropping
    half = fit_rapm(stints, lam=500.0, garbage_weight=0.5)
    assert half.meta["garbage_weight"] == 0.5

    plain_stints = stints.drop(columns=["garbage"])
    with pytest.raises(ValueError, match="garbage"):
        fit_rapm(plain_stints, lam=500.0, garbage_weight=0.5)


def with_late_context(stints, seed=0, late_frac=0.25):
    """Attach margin/secs_left to a random subset of rows (Q4-like)."""
    rng = np.random.default_rng(seed)
    late = rng.random(len(stints)) < late_frac
    stints = stints.copy()
    stints["margin"] = np.where(
        late, rng.integers(-35, 36, len(stints)).astype(float), np.nan
    )
    stints["secs_left"] = np.where(
        late, rng.integers(0, 721, len(stints)).astype(float), np.nan
    )
    return stints


def test_garbage_rule_flags_match_threshold():
    from fablerapm.model import garbage_flags

    true_off, true_def = make_league()
    stints = with_late_context(simulate_stints(true_off, true_def, n_games=20))
    flags = garbage_flags(stints, (12.0, 1.5))
    margin = stints["margin"].to_numpy()
    secs = stints["secs_left"].to_numpy()
    expected = (
        np.isfinite(margin)
        & np.isfinite(secs)
        & (np.abs(margin) >= 12.0 + 1.5 * secs / 60.0)
    )
    assert (flags == expected).all()
    assert flags.any() and not flags.all()
    # early-game rows (no context) are never garbage under a rule
    assert not flags[~np.isfinite(margin)].any()


def test_garbage_rule_weight_zero_equals_filtering():
    from fablerapm.model import garbage_flags

    true_off, true_def = make_league()
    stints = with_late_context(simulate_stints(true_off, true_def, n_games=60))
    stints["garbage"] = False  # stored flag disagrees with the rule on purpose
    rule = (10.0, 1.0)

    dropped = fit_rapm(
        stints, lam=500.0, garbage_weight=0.0, garbage_rule=rule
    )
    filtered = fit_rapm(stints[~garbage_flags(stints, rule)], lam=500.0)
    merged = dropped.players.merge(
        filtered.players, on="player_id", suffixes=("_d", "_f")
    )
    assert np.allclose(merged["rapm_d"], merged["rapm_f"], atol=1e-6)
    assert dropped.meta["garbage_rule"] == [10.0, 1.0]

    # rule requested but schema lacks the context columns
    with pytest.raises(ValueError, match="margin"):
        fit_rapm(
            stints.drop(columns=["margin", "secs_left"]),
            lam=500.0, garbage_weight=0.0, garbage_rule=rule,
        )
