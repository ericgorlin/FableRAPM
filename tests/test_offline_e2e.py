"""Offline end-to-end: synthetic stint parquet -> CLI rapm -> CSV output.

Exercises storage layout, season/type resolution, the model, and CSV writing
without any network access (names lookup is skipped with --no-names).
"""

import json

import numpy as np
import pandas as pd

from fablerapm.cli import main
from fablerapm.config import stints_path

from test_model import make_league, simulate_stints


def seed_stints(data_dir, season, season_type, n_games, seed):
    true_off, true_def = make_league()
    df = simulate_stints(true_off, true_def, n_games=n_games, seed=seed)
    df["game_id"] = f"S{seed}_" + df["game_id"]
    path = stints_path(data_dir, season, season_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def test_cli_rapm_end_to_end(tmp_path):
    data_dir = tmp_path / "data"
    seed_stints(data_dir, "2023-24", "Regular Season", n_games=40, seed=1)
    seed_stints(data_dir, "2023-24", "Playoffs", n_games=10, seed=2)
    seed_stints(data_dir, "2024-25", "Regular Season", n_games=40, seed=3)
    seed_stints(data_dir, "2024-25", "Playoffs", n_games=10, seed=4)

    rc = main([
        "rapm",
        "--data-dir", str(data_dir),
        "--seasons", "2023-24:2024-25",
        "--season-types", "regular,playoffs",
        "--lambda", "500",
        "--no-names",
    ])
    assert rc == 0

    results = sorted((data_dir / "results").glob("rapm_*.csv"))
    # one output per (season, season type) by default
    assert len(results) == 4
    for csv_path in results:
        out = pd.read_csv(csv_path)
        assert list(out.columns) == [
            "player_id", "player_name", "orapm", "drapm", "rapm",
            "off_poss", "def_poss",
        ]
        assert len(out) > 0
        assert np.isfinite(out["rapm"]).all()
        meta_path = csv_path.parent / (csv_path.stem + ".meta.json")
        meta = json.loads(meta_path.read_text())
        assert meta["lambda"] == 500.0
        assert meta["n_games"] in (10, 40)

    # pooled run across seasons and types -> single CSV
    rc = main([
        "rapm",
        "--data-dir", str(data_dir),
        "--seasons", "2023-24:2024-25",
        "--season-types", "regular,playoffs",
        "--lambda", "500",
        "--pool", "--combine-types",
        "--no-names",
        "--out-dir", str(tmp_path / "pooled"),
    ])
    assert rc == 0
    pooled = list((tmp_path / "pooled").glob("rapm_*.csv"))
    assert len(pooled) == 1
    meta = json.loads(
        (pooled[0].parent / (pooled[0].stem + ".meta.json")).read_text()
    )
    assert meta["n_games"] == 100
    assert meta["seasons"] == ["2023-24", "2024-25"]
