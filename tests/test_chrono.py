"""Chronological (forward-chaining) splits for evaluate and tune (offline)."""

import numpy as np
import pandas as pd
import pytest

from fablerapm.evaluate import chrono_game_order, chrono_split, evaluate_variants

from test_model import make_league, simulate_stints


def two_season_stints():
    true_off, true_def = make_league()
    frames = []
    for season, season_type, seed, tag in [
        ("2023-24", "Regular Season", 41, "A"),
        ("2023-24", "Playoffs", 42, "B"),
        ("2024-25", "Regular Season", 43, "C"),
    ]:
        df = simulate_stints(true_off, true_def, n_games=10, seed=seed)
        df["game_id"] = f"{tag}_" + df["game_id"]
        df["season"] = season
        df["season_type"] = season_type
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_chrono_order_respects_season_and_type():
    stints = two_season_stints()
    order = chrono_game_order(stints)
    # 2023-24 regular < 2023-24 playoffs < 2024-25 regular, and game ids
    # ascend within each block
    tags = [g.split("_")[0] for g in order]
    assert tags == ["A"] * 10 + ["B"] * 10 + ["C"] * 10
    for tag in "ABC":
        block = [g for g in order if g.startswith(tag)]
        assert block == sorted(block)


def test_chrono_split_trains_strictly_earlier():
    stints = two_season_stints()
    order = chrono_game_order(stints)
    rank = {g: i for i, g in enumerate(order)}
    for fold in range(3):
        train, test = chrono_split(stints, test_frac=0.1, fold=fold, n_folds=3)
        max_train = max(rank[g] for g in train["game_id"].unique())
        min_test = min(rank[g] for g in test["game_id"].unique())
        assert max_train < min_test
    # later folds test later blocks, and blocks don't overlap
    blocks = [
        set(chrono_split(stints, 0.1, fold, 3)[1]["game_id"].unique())
        for fold in range(3)
    ]
    assert not (blocks[0] & blocks[1]) and not (blocks[1] & blocks[2])
    # final fold ends at the latest game
    assert order[-1] in blocks[2]

    with pytest.raises(ValueError):
        chrono_split(stints, test_frac=0.5, fold=0, n_folds=3)


def test_chrono_split_without_season_columns_falls_back_to_game_id():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=20, seed=44)
    train, test = chrono_split(stints, test_frac=0.2)
    assert max(train["game_id"]) < min(test["game_id"])


def test_evaluate_variants_chrono_split():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=40, seed=45)
    table = evaluate_variants(
        stints, {"plain": None}, lam=1000.0, split="chrono", test_frac=0.2
    )
    assert np.isfinite(table["holdout_mse"]).all()
    assert set(table["variant"]) == {"intercept-only", "plain"}
