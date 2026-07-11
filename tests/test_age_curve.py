"""Aging curve for the last-season prior, on synthetic seasons where young
players' impact persists/improves and old players' decays (offline)."""

import json

import numpy as np
import pandas as pd
import pytest

from fablerapm.cli import main
from fablerapm.config import stints_path
from fablerapm.features import features_path
from fablerapm.model import fit_rapm
from fablerapm.prior import (
    _age_bucket,
    last_season_prior,
    learn_age_curve,
    load_age_curve,
    train_age_curve,
)

from test_model import N_PLAYERS, make_league, simulate_stints

SEASONS = ["2022-23", "2023-24", "2024-25"]
STYPE = "Regular Season"
# player index -> age in the FIRST season; +1 each season after
BASE_AGES = {i: (21 if i < 20 else 27 if i < 40 else 34) for i in range(N_PLAYERS)}
PERSISTENCE = {21: 1.2, 27: 1.0, 34: 0.6}  # by base age group


def seed_aging_world(data_dir, n_games=250, lam_noise_seed=7):
    true_off, true_def = make_league()
    # inflate effects a bit so slopes are measurable through ridge noise
    true_off, true_def = true_off * 1.5, true_def * 1.5
    for si, season in enumerate(SEASONS):
        if si > 0:
            g = np.array([PERSISTENCE[BASE_AGES[i]] for i in range(N_PLAYERS)])
            true_off, true_def = true_off * g, true_def * g
        df = simulate_stints(
            true_off, true_def, n_games=n_games, seed=lam_noise_seed + si
        )
        df["game_id"] = f"S{si}_" + df["game_id"]
        path = stints_path(data_dir, season, STYPE)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

        features = pd.DataFrame({
            "player_id": np.arange(1, N_PLAYERS + 1),
            "minutes": 2000.0,
            "age": [float(BASE_AGES[i] + si) for i in range(N_PLAYERS)],
        })
        fpath = features_path(data_dir, season, STYPE)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(fpath, index=False)


def test_age_buckets_cover_and_label():
    assert _age_bucket(19) == "<=23" and _age_bucket(23) == "<=23"
    assert _age_bucket(25) == "24-26"
    assert _age_bucket(33) == "33+" and _age_bucket(40) == "33+"
    with pytest.raises(ValueError):
        _age_bucket(-1)


def test_learn_age_curve_orders_buckets_by_persistence(tmp_path):
    data_dir = tmp_path / "data"
    seed_aging_world(data_dir)
    curve = learn_age_curve(data_dir, SEASONS, [STYPE], lam=500.0)

    young = curve["buckets"]["<=23"]
    old = curve["buckets"]["33+"]
    assert young["players"] > 0 and old["players"] > 0
    # the world's truth: young talent persists (x1.2/yr), old decays (x0.6)
    assert young["scale"] > curve["global_scale"] > old["scale"]
    assert 0.0 <= old["scale"] and young["scale"] <= 1.5
    # buckets with no observations shrink fully to the global slope
    assert curve["buckets"]["30-32"]["scale"] == pytest.approx(
        curve["global_scale"]
    )

    with pytest.raises(ValueError, match="consecutive"):
        learn_age_curve(data_dir, ["2022-23"], [STYPE], lam=500.0)


def test_last_season_prior_age_scale_applies_curve(tmp_path):
    data_dir = tmp_path / "data"
    seed_aging_world(data_dir, n_games=120)
    train_age_curve(data_dir, SEASONS[:2], [STYPE], lam=500.0)
    curve = load_age_curve(data_dir)

    prior = last_season_prior(
        data_dir, ["2024-25"], [STYPE], lam=500.0, scale="age"
    )
    prev = fit_rapm(
        pd.read_parquet(stints_path(data_dir, "2023-24", STYPE)), lam=500.0
    ).players.set_index("player_id")

    young_pid, old_pid = 1, N_PLAYERS  # base ages 21 and 34
    s_young = curve["buckets"]["<=23"]["scale"]
    s_old = curve["buckets"]["33+"]["scale"]
    assert prior[young_pid][0] == pytest.approx(
        s_young * prev.loc[young_pid, "orapm"]
    )
    assert prior[old_pid][1] == pytest.approx(s_old * prev.loc[old_pid, "drapm"])

    # without the artifact, the age scale is a clear, named error
    (data_dir / "results" / "age_curve.json").unlink()
    with pytest.raises(FileNotFoundError, match="age-curve"):
        last_season_prior(data_dir, ["2024-25"], [STYPE], lam=500.0, scale="age")


def test_cli_age_curve_and_rapm(tmp_path):
    data_dir = tmp_path / "data"
    seed_aging_world(data_dir, n_games=120)

    rc = main([
        "age-curve", "--data-dir", str(data_dir),
        "--seasons", "2022-23:2024-25", "--season-types", "regular",
        "--lambda", "1000",
    ])
    assert rc == 0
    assert (data_dir / "results" / "age_curve.json").exists()

    rc = main([
        "rapm", "--data-dir", str(data_dir),
        "--seasons", "2024-25", "--season-types", "regular",
        "--prior", "last-season", "--prior-scale", "age",
        "--lambda", "1000", "--no-names",
    ])
    assert rc == 0
    meta_files = list((data_dir / "results").glob("rapm_*lastseason*.meta.json"))
    assert meta_files
    meta = json.loads(meta_files[0].read_text())
    assert meta["prior_scale"] == "age"

    # age scale is last-season-only
    with pytest.raises(ValueError, match="last-season"):
        main([
            "rapm", "--data-dir", str(data_dir),
            "--seasons", "2024-25", "--season-types", "regular",
            "--prior", "two-phase", "--prior-scale", "age",
            "--lambda", "1000", "--no-names",
        ])

    # malformed --prior-scale is a standard argparse error
    with pytest.raises(SystemExit):
        main([
            "rapm", "--data-dir", str(data_dir), "--seasons", "2024-25",
            "--prior-scale", "young", "--no-names",
        ])
