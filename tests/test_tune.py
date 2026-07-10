"""Parameter tuning end-to-end on small synthetic data (offline)."""

import numpy as np

from fablerapm.cli import main
from fablerapm.config import stints_path
from fablerapm.tune import load_tuned_config, tune, tuned_config_path

from test_model import make_league, simulate_stints


def seed_two_seasons(data_dir):
    true_off, true_def = make_league()
    rng = np.random.default_rng(5)
    for season, seed in [("2023-24", 31), ("2024-25", 32)]:
        df = simulate_stints(true_off, true_def, n_games=25, seed=seed)
        df["game_id"] = f"S{seed}_" + df["game_id"]
        df["garbage"] = rng.random(len(df)) < 0.08
        late = rng.random(len(df)) < 0.25
        df["margin"] = np.where(
            late, rng.integers(-35, 36, len(df)).astype(float), np.nan
        )
        df["secs_left"] = np.where(
            late, rng.integers(0, 721, len(df)).astype(float), np.nan
        )
        path = stints_path(data_dir, season, "Regular Season")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


def test_tune_writes_config_and_rapm_uses_it(tmp_path):
    data_dir = tmp_path / "data"
    seed_two_seasons(data_dir)

    result = tune(
        data_dir, ["2024-25"], ["Regular Season"],
        n_seeds=2, lam=1000.0,
    )
    assert tuned_config_path(data_dir).exists()
    # searched coordinates present, incl. last-season prior (2023-24 exists)
    # and the lambda multiplier grid
    notes = {h["note"] for h in result["history"]}
    assert {"baseline", "garbage", "prior", "interactions", "lambda"} <= notes
    assert any(
        h["config"]["prior"] == "last-season" for h in result["history"]
    )
    # fit-time garbage rules were searched (schema carries margin/secs_left)
    assert any(
        h["config"].get("garbage_rule") for h in result["history"]
    )
    assert np.isfinite(result["inner_mse"])
    assert result["holdout_mse"] == result["inner_mse"]  # legacy alias

    # nested evaluation: the outer block was never used for selection, and
    # its scores are reported for tuned vs default vs intercept-only
    assert result["split"] == "chrono"
    for key in ("outer_mse", "outer_default_mse", "outer_intercept_mse"):
        assert np.isfinite(result[key])

    cfg = load_tuned_config(data_dir)
    # lambda is searched over multipliers of the base CV/fixed value
    assert cfg["lambda"] in {1000.0 * m for m in (0.25, 0.5, 1.0, 2.0, 4.0)}
    assert set(cfg) >= {"garbage_weight", "decay", "prior", "interactions"}

    rc = main([
        "rapm", "--data-dir", str(data_dir),
        "--seasons", "2024-25", "--season-types", "regular",
        "--tuned", "--no-names",
    ])
    assert rc == 0
    assert list((data_dir / "results").glob("rapm_2024_25_*.csv"))


def test_tune_random_split_without_outer_holdout(tmp_path):
    data_dir = tmp_path / "data"
    seed_two_seasons(data_dir)
    result = tune(
        data_dir, ["2024-25"], ["Regular Season"],
        n_seeds=2, lam=1000.0, split="random", outer_frac=0.0,
    )
    assert result["split"] == "random"
    assert "outer_mse" not in result
    assert np.isfinite(result["inner_mse"])
