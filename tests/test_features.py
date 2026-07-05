"""Feature endpoint parsing from canned payloads (offline)."""

import json

import pandas as pd

import fablerapm.features as feat
from fablerapm.features import build_season_features, load_features


def dash_payload(measure):
    headers = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE",
        "GP", "W", "L", "W_PCT", "MIN", "PTS", "AST", "STL", "PTS_RANK",
        "CFID", "CFPARAMS",
    ]
    rows = [
        [1, "Player One", 10, "AAA", 25, 70, 40, 30, 0.57, 34.0, 28.5, 8.1, 1.4, 1, 5, "x"],
        [2, "Player Two", 11, "BBB", 30, 60, 20, 40, 0.33, 20.0, 12.0, 2.0, 0.7, 2, 5, "x"],
    ]
    return {"resultSets": [{"name": measure, "headers": headers, "rowSet": rows}]}


def pt_payload():
    headers = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP",
        "W", "L", "MIN", "DRIVES", "DRIVE_PTS", "DRIVE_FG_PCT",
    ]
    rows = [
        [1, "Player One", 10, "AAA", 70, 40, 30, 2380.0, 700.0, 840.0, 0.51],
        [2, "Player Two", 11, "BBB", 60, 20, 40, 1200.0, 120.0, 100.0, 0.44],
    ]
    return {"resultSets": [{"name": "pt", "headers": headers, "rowSet": rows}]}


def seed_cache(data_dir, season="2023-24", stype="Regular Season"):
    slug = f"{season.replace('-', '_')}_{stype.replace(' ', '').lower()}"
    cache = data_dir / "raw" / "features"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"dash_base_{slug}.json").write_text(json.dumps(dash_payload("Base")))
    (cache / f"dash_advanced_{slug}.json").write_text(
        json.dumps(dash_payload("Advanced"))
    )
    for m in feat.PT_MEASURE_TYPES:
        (cache / f"pt_{m.lower()}_{slug}.json").write_text(json.dumps(pt_payload()))


def test_build_and_load_features_from_cache(tmp_path):
    data_dir = tmp_path / "data"
    seed_cache(data_dir)
    build_season_features(data_dir, "2023-24", "Regular Season")

    df = load_features(data_dir, ["2023-24"], ["Regular Season"])
    assert set(df["player_id"]) == {1, 2}
    # id/rank/meta columns excluded, stat columns prefixed
    assert "box_PTS" in df and "adv_PTS" in df
    assert not any("RANK" in c or "CFID" in c for c in df.columns)
    assert df["minutes"].iloc[0] == 34.0 * 70
    # tracking counts are per-36: 700 drives in 2380 min -> 10.588...
    drives = df.set_index("player_id")["pt_drives_DRIVES"]
    assert abs(drives[1] - 700.0 / 2380.0 * 36) < 1e-9
    # percentage columns not rate-converted
    assert df.set_index("player_id")["pt_drives_DRIVE_FG_PCT"][1] == 0.51


def test_old_seasons_skip_tracking(tmp_path):
    data_dir = tmp_path / "data"
    seed_cache(data_dir, season="2005-06")
    build_season_features(data_dir, "2005-06", "Regular Season")
    df = load_features(data_dir, ["2005-06"], ["Regular Season"])
    assert not any(c.startswith("pt_") for c in df.columns)
    assert "box_PTS" in df
