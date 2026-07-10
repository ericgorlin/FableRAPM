"""Incremental build/resume logic, offline via the fixture game."""

import json

import pandas as pd

import fablerapm.build as build_mod
from fablerapm.build import build_season
from fablerapm.config import ensure_raw_dirs, manifest_path, stints_path

from test_fixture_game import GAME_ID, make_fixture_pbp

SEASON, STYPE = "2023-24", "Regular Season"


def seed(tmp_path, monkeypatch, game_ids):
    data_dir = tmp_path / "data"
    raw = ensure_raw_dirs(data_dir)
    (raw / "pbp" / f"stats_{GAME_ID}.json").write_text(
        json.dumps(make_fixture_pbp())
    )
    monkeypatch.setattr(
        build_mod, "season_game_ids", lambda *a, **k: list(game_ids)
    )
    return data_dir


def test_build_processes_and_resumes(tmp_path, monkeypatch):
    data_dir = seed(tmp_path, monkeypatch, [GAME_ID])

    summary = build_season(data_dir, SEASON, STYPE)
    assert summary["new"] == 1 and summary["processed"] == 1
    assert summary["failed"] == 0

    df = pd.read_parquet(stints_path(data_dir, SEASON, STYPE))
    assert set(df["game_id"]) == {GAME_ID}
    assert df["poss"].sum() == 44 and df["points"].sum() == 80

    manifest = json.loads(manifest_path(data_dir, SEASON, STYPE).read_text())
    assert manifest["processed"][GAME_ID]["poss"] == 44

    # second run is a no-op: nothing new, data unchanged
    summary2 = build_season(data_dir, SEASON, STYPE)
    assert summary2["new"] == 0
    df2 = pd.read_parquet(stints_path(data_dir, SEASON, STYPE))
    assert len(df2) == len(df)


def test_build_parallel_workers_match_serial(tmp_path, monkeypatch):
    # three cached copies of the fixture game under different ids, one
    # broken game: parallel re-parse must produce exactly the serial result
    ids = ["0022300901", "0022300902", "0022300903"]
    bad_id = "0022300904"
    data_dir = seed(tmp_path, monkeypatch, ids + [bad_id])
    raw = data_dir / "raw"
    fixture = json.dumps(make_fixture_pbp())
    for gid in ids:
        (raw / "pbp" / f"stats_{gid}.json").write_text(
            fixture.replace(GAME_ID, gid)
        )
    (raw / "pbp" / f"stats_{bad_id}.json").write_text("{}")

    summary = build_season(data_dir, SEASON, STYPE, workers=2)
    assert summary["processed"] == 3 and summary["failed"] == 1

    df = pd.read_parquet(stints_path(data_dir, SEASON, STYPE))
    assert set(df["game_id"]) == set(ids)
    per_game = df.groupby("game_id")[["poss", "points"]].sum()
    assert (per_game["poss"] == 44).all() and (per_game["points"] == 80).all()

    manifest = json.loads(manifest_path(data_dir, SEASON, STYPE).read_text())
    assert bad_id in manifest["failed"]
    # a parallel re-run over the same manifest is a no-op
    assert build_season(data_dir, SEASON, STYPE, workers=2)["new"] == 0


def test_build_records_failures(tmp_path, monkeypatch):
    bad_id = "0022309998"
    data_dir = seed(tmp_path, monkeypatch, [GAME_ID, bad_id])
    raw = data_dir / "raw"
    (raw / "pbp" / f"stats_{bad_id}.json").write_text("{}")  # unparseable

    summary = build_season(data_dir, SEASON, STYPE)
    assert summary["processed"] == 1 and summary["failed"] == 1

    manifest = json.loads(manifest_path(data_dir, SEASON, STYPE).read_text())
    assert bad_id in manifest["failed"]

    # failed games are not retried by default, but are with --retry-failed
    (raw / "pbp" / f"stats_{bad_id}.json").write_text(
        json.dumps(make_fixture_pbp())
    )
    assert build_season(data_dir, SEASON, STYPE)["new"] == 0
    summary3 = build_season(data_dir, SEASON, STYPE, retry_failed=True)
    assert summary3["processed"] == 2 and summary3["failed"] == 0
