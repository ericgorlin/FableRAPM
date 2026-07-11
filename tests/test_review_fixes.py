"""Regression tests for external-review findings."""

import numpy as np
import pandas as pd

from fablerapm.build import build_season
from fablerapm.config import stints_path
from fablerapm.model import build_design, cross_validate_lambda, fit_rapm

from test_build import SEASON, STYPE, seed
from test_fixture_game import GAME_ID
from test_model import make_league, simulate_stints


def test_cv_uses_prior_adjusted_target():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=60, seed=41)
    design = build_design(stints)
    # a strong prior changes the residual problem, so CV tables must differ
    prior_y = design.y - design.X[:, :1].toarray().ravel() * 50.0
    _, plain_table = cross_validate_lambda(design, lambdas=[100, 1000], n_folds=3)
    _, prior_table = cross_validate_lambda(
        design, lambdas=[100, 1000], n_folds=3, y=prior_y
    )
    assert plain_table != prior_table
    # and fit_rapm with a prior runs CV end-to-end without error
    pid = int(stints["off_lineup"].iloc[0].split("-")[0])
    result = fit_rapm(stints, lam="cv", lambdas=[100, 1000], n_folds=3,
                      prior={pid: (30.0, 0.0)})
    assert np.isfinite(result.players["rapm"]).all()


def test_tune_handles_zero_possession_rows(tmp_path):
    from fablerapm.tune import tune

    data_dir = tmp_path / "data"
    true_off, true_def = make_league()
    df = simulate_stints(true_off, true_def, n_games=25, seed=42)
    orphan = df.iloc[[0]].copy()
    orphan["poss"], orphan["points"] = 0, 1
    df = pd.concat([df, orphan], ignore_index=True)
    path = stints_path(data_dir, "2024-25", "Regular Season")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    result = tune(data_dir, ["2024-25"], ["Regular Season"], n_seeds=2, lam=1000.0)
    assert np.isfinite(result["inner_mse"])
    assert all(np.isfinite(h["mse"]) for h in result["history"])


def test_build_reconciles_parquet_ahead_of_manifest(tmp_path, monkeypatch):
    data_dir = seed(tmp_path, monkeypatch, [GAME_ID])
    build_season(data_dir, SEASON, STYPE)
    baseline = pd.read_parquet(stints_path(data_dir, SEASON, STYPE))

    # simulate a crash between parquet write and manifest write: parquet has
    # a game the manifest doesn't know about
    from fablerapm.config import manifest_path
    import json

    mpath = manifest_path(data_dir, SEASON, STYPE)
    manifest = json.loads(mpath.read_text())
    del manifest["processed"][GAME_ID]
    mpath.write_text(json.dumps(manifest))

    summary = build_season(data_dir, SEASON, STYPE)
    assert summary["processed"] == 1
    after = pd.read_parquet(stints_path(data_dir, SEASON, STYPE))
    # re-processing did not duplicate: same totals as the clean build
    assert len(after) == len(baseline)
    assert after["poss"].sum() == baseline["poss"].sum()
