"""Validation against official game-log scores, offline via the fixture game."""

import json

import pandas as pd

import fablerapm.build as build_mod
from fablerapm.build import build_season
from fablerapm.config import stints_path
from fablerapm.validate import validate_season

from test_build import SEASON, STYPE, seed
from test_fixture_game import AWAY, GAME_ID, HOME


def write_schedule(data_dir, rows):
    name = (
        f"stats_leaguegamelog_nba_{SEASON.replace('-', '_')}_"
        f"{STYPE.replace(' ', '_')}.json"
    )
    payload = {"resultSets": [{
        "headers": ["GAME_ID", "TEAM_ID", "PTS", "MATCHUP"],
        "rowSet": rows,
    }]}
    (data_dir / "raw" / "schedule" / name).write_text(json.dumps(payload))


def build_fixture(tmp_path, monkeypatch):
    data_dir = seed(tmp_path, monkeypatch, [GAME_ID])
    build_season(data_dir, SEASON, STYPE)
    return data_dir


def test_validate_clean_data_passes(tmp_path, monkeypatch):
    data_dir = build_fixture(tmp_path, monkeypatch)
    write_schedule(data_dir, [
        [GAME_ID, HOME, 40, "AAA vs. BBB"],
        [GAME_ID, AWAY, 40, "BBB @ AAA"],
    ])
    report = validate_season(data_dir, SEASON, STYPE)
    assert report["problems"] == []
    assert report["score_checked"] == 1
    assert report["score_mismatches"] == []
    assert report["games_missing"] == []


def test_validate_detects_score_mismatch(tmp_path, monkeypatch):
    data_dir = build_fixture(tmp_path, monkeypatch)
    write_schedule(data_dir, [
        [GAME_ID, HOME, 42, "AAA vs. BBB"],  # official says 42, stints say 40
        [GAME_ID, AWAY, 40, "BBB @ AAA"],
    ])
    report = validate_season(data_dir, SEASON, STYPE)
    assert len(report["score_mismatches"]) == 1
    assert report["score_mismatches"][0]["team_id"] == HOME
    assert report["problems"]


def test_validate_detects_missing_games(tmp_path, monkeypatch):
    data_dir = build_fixture(tmp_path, monkeypatch)
    write_schedule(data_dir, [
        [GAME_ID, HOME, 40, "AAA vs. BBB"],
        [GAME_ID, AWAY, 40, "BBB @ AAA"],
        ["0022300777", HOME, 100, "AAA vs. BBB"],  # in schedule, never built
        ["0022300777", AWAY, 90, "BBB @ AAA"],
    ])
    report = validate_season(data_dir, SEASON, STYPE)
    assert report["games_missing"] == ["0022300777"]
    assert report["problems"]


def test_validate_detects_corrupted_stints(tmp_path, monkeypatch):
    data_dir = build_fixture(tmp_path, monkeypatch)
    write_schedule(data_dir, [
        [GAME_ID, HOME, 40, "AAA vs. BBB"],
        [GAME_ID, AWAY, 40, "BBB @ AAA"],
    ])
    path = stints_path(data_dir, SEASON, STYPE)
    df = pd.read_parquet(path)
    df.loc[df["off_team_id"] == HOME, "points"] += 1  # corrupt attribution
    df.to_parquet(path, index=False)
    report = validate_season(data_dir, SEASON, STYPE)
    assert report["score_mismatches"]
