"""SPM prior, two-phase prior, and holdout evaluation on synthetic data."""

import numpy as np
import pandas as pd

from fablerapm.evaluate import evaluate_variants, holdout_split, predict_stints
from fablerapm.model import fit_rapm
from fablerapm.prior import SpmModel, fit_spm, predict_spm, two_phase_prior

from test_model import N_PLAYERS, make_league, simulate_stints

rng = np.random.default_rng(11)


def make_features(true_off, true_def, noise=0.3):
    """Synthetic box-score-ish features correlated with true impact."""
    n = len(true_off)
    return pd.DataFrame({
        "player_id": np.arange(1, n + 1),
        "minutes": rng.uniform(500, 2500, n),
        "age": rng.uniform(19, 40, n).round(),
        "box_PTS": 100 * true_off + rng.normal(0, noise, n) + 20,
        "box_AST": 50 * true_off + rng.normal(0, noise, n) + 5,
        "box_STL": -60 * true_def + rng.normal(0, noise, n) + 2,
        "pt_defense_CONTESTED_SHOTS": -80 * true_def + rng.normal(0, noise, n) + 10,
        "box_NOISE": rng.normal(0, 1, n),
    })


def test_spm_fit_predict_roundtrip():
    true_off, true_def = make_league()
    features = make_features(true_off, true_def)
    targets = pd.DataFrame({
        "player_id": np.arange(1, N_PLAYERS + 1),
        "orapm": 100 * true_off + rng.normal(0, 0.5, N_PLAYERS),
        "drapm": -100 * true_def + rng.normal(0, 0.5, N_PLAYERS),
    })
    model = fit_spm(features, targets)
    # metadata columns feed other machinery (minutes-weighting, age curve),
    # never the SPM regression itself
    assert "age" not in model.feature_cols
    assert "minutes" not in model.feature_cols
    prior = predict_spm(model, features)
    assert set(prior) == set(range(1, N_PLAYERS + 1))
    o_pred = np.array([prior[i + 1][0] for i in range(N_PLAYERS)])
    d_pred = np.array([prior[i + 1][1] for i in range(N_PLAYERS)])
    assert np.corrcoef(o_pred, 100 * true_off)[0, 1] > 0.9
    assert np.corrcoef(d_pred, -100 * true_def)[0, 1] > 0.9
    # serialization roundtrip, incl. provenance fields
    model.train_seasons = ["2021-22", "2022-23"]
    model.train_season_types = ["Regular Season"]
    restored = SpmModel.from_json(model.to_json())
    assert predict_spm(restored, features)[1] == prior[1]
    assert restored.train_seasons == ["2021-22", "2022-23"]
    # artifacts saved before provenance existed still load
    import json as _json

    legacy = _json.loads(model.to_json())
    del legacy["train_seasons"], legacy["train_season_types"]
    old = SpmModel.from_json(_json.dumps(legacy))
    assert old.train_seasons is None
    # predicting with a missing feature column still works (treated as mean)
    slim = features.drop(columns=["pt_defense_CONTESTED_SHOTS"])
    assert np.isfinite(predict_spm(model, slim)[1]).all()


def test_spm_prior_improves_estimates_on_thin_data():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=40, seed=5)
    features = make_features(true_off, true_def, noise=0.1)
    targets = pd.DataFrame({
        "player_id": np.arange(1, N_PLAYERS + 1),
        "orapm": 100 * true_off,
        "drapm": -100 * true_def,
    })
    prior = predict_spm(fit_spm(features, targets), features)

    plain = fit_rapm(stints, lam=2000.0).players.set_index("player_id").sort_index()
    with_prior = fit_rapm(stints, lam=2000.0, prior=prior).players.set_index(
        "player_id"
    ).sort_index()
    truth = 100 * (true_off - true_def)
    corr_plain = np.corrcoef(plain["rapm"], truth)[0, 1]
    corr_prior = np.corrcoef(with_prior["rapm"], truth)[0, 1]
    assert corr_prior > corr_plain


def test_two_phase_decompresses_star_estimates():
    true_off, true_def = make_league()
    # make player 1 a clear star on offense
    true_off[0] = 0.08
    stints = simulate_stints(true_off, true_def, n_games=300, seed=9)
    phase1 = fit_rapm(stints, lam=2000.0).players.set_index("player_id")
    prior = two_phase_prior(stints, lam=2000.0)
    phase2 = fit_rapm(stints, lam=2000.0, prior=prior).players.set_index("player_id")
    # heavy flat shrinkage compresses the star; phase two moves him back up
    assert phase2.loc[1, "orapm"] > phase1.loc[1, "orapm"]
    assert phase2.loc[1, "orapm"] <= 100 * true_off[0] * 1.5


def test_holdout_split_partitions_games():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=50, seed=3)
    train, test = holdout_split(stints, test_frac=0.2, seed=1)
    train_games = set(train["game_id"])
    test_games = set(test["game_id"])
    assert not train_games & test_games
    assert len(test_games) == 10
    assert len(train) + len(test) == len(stints)


def test_evaluate_variants_truth_prior_wins():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=120, seed=4)
    truth_prior = {
        i + 1: (100 * true_off[i], -100 * true_def[i]) for i in range(N_PLAYERS)
    }
    table = evaluate_variants(
        stints,
        {"plain": None, "truth-prior": lambda train: truth_prior},
        lam=2000.0,
        seed=2,
    )
    assert list(table["variant"]) == ["intercept-only", "plain", "truth-prior"]
    assert np.isfinite(table["holdout_mse"]).all()
    mse = table.set_index("variant")["holdout_mse"]
    # the informative comparison is between variants; intercept-only can win
    # outright when true effects are small relative to stint noise
    assert mse["truth-prior"] < mse["plain"]
    assert "vs_baseline" in table


def test_last_season_prior_scales_previous_fit(tmp_path):
    from fablerapm.config import stints_path
    from fablerapm.prior import last_season_prior

    data_dir = tmp_path / "data"
    true_off, true_def = make_league()
    prev = simulate_stints(true_off, true_def, n_games=30, seed=13)
    path = stints_path(data_dir, "2023-24", "Regular Season")
    path.parent.mkdir(parents=True, exist_ok=True)
    prev.to_parquet(path, index=False)

    prior = last_season_prior(
        data_dir, ["2024-25"], ["Regular Season"], lam=500.0, scale=0.5
    )
    direct = fit_rapm(prev, lam=500.0).players.set_index("player_id")
    pid = int(direct.index[0])
    assert prior[pid][0] == 0.5 * direct.loc[pid, "orapm"]
    assert prior[pid][1] == 0.5 * direct.loc[pid, "drapm"]


def test_predict_stints_handles_unseen_players():
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=30, seed=6)
    result = fit_rapm(stints, lam=1000.0)
    unseen = stints.head(3).copy()
    unseen["off_lineup"] = "901-902-903-904-905"
    preds = predict_stints(result, unseen)
    # all-unseen offense vs known defense: intercept minus defense effects
    assert np.isfinite(preds).all()
