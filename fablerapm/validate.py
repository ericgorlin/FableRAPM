"""Cross-check stored stint data against independent ground truth.

Per-game reconciliation at build time catches internal inconsistencies; this
checks against a *separate* source — the cached league game log, which
carries official final scores per team — so systematic attribution bugs
can't hide. Zero network: everything comes from data/ on disk.

Checks per (season, season type):
  1. Coverage: every final game in the schedule is processed or has a
     recorded failure.
  2. Exactness: for every processed game, each team's points summed from
     stint rows equal the official game-log score.
  3. Plausibility: league points per 100 possessions within a sane band.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from .build import _load_manifest
from .config import manifest_path, raw_dir, stints_path

logger = logging.getLogger(__name__)

PTS_PER_100_BAND = (85.0, 130.0)  # league averages span ~92 (98-99) to ~117


def _schedule_team_points(
    data_dir: Path, season: str, season_type: str
) -> dict[tuple[str, int], int] | None:
    """(game_id, team_id) -> official final points, from the cached game log."""
    name = (
        f"stats_leaguegamelog_nba_{season.replace('-', '_')}_"
        f"{season_type.replace(' ', '_')}.json"
    )
    path = raw_dir(data_dir) / "schedule" / name
    if not path.exists():
        return None
    rs = json.loads(path.read_text())["resultSets"][0]
    headers = rs["headers"]
    g, t, p = headers.index("GAME_ID"), headers.index("TEAM_ID"), headers.index("PTS")
    return {(row[g], row[t]): row[p] for row in rs["rowSet"]}


def validate_season(data_dir: Path, season: str, season_type: str) -> dict:
    report = {"season": season, "season_type": season_type, "problems": []}

    spath = stints_path(data_dir, season, season_type)
    if not spath.exists():
        report["problems"].append(f"no stint data at {spath}")
        return report
    stints = pd.read_parquet(spath)
    manifest = _load_manifest(manifest_path(data_dir, season, season_type))
    processed = set(manifest["processed"])
    failed = set(manifest["failed"])
    report["games_processed"] = len(processed)
    report["games_failed"] = len(failed)
    report["build_warnings"] = sum(
        1 for g in manifest["processed"].values() if g.get("warnings")
    )

    official = _schedule_team_points(data_dir, season, season_type)
    if official is None:
        report["problems"].append("no cached schedule; coverage/score checks skipped")
    else:
        schedule_games = {g for g, _ in official}
        missing = schedule_games - processed - failed
        report["games_in_schedule"] = len(schedule_games)
        report["games_missing"] = sorted(missing)
        if missing:
            report["problems"].append(
                f"{len(missing)} final games neither processed nor failed "
                "(re-run `fablerapm build`)"
            )

        # stint rows must reproduce every official team score exactly
        team_pts = (
            stints.groupby(["game_id", "off_team_id"])["points"].sum().to_dict()
        )
        mismatches = []
        for (game_id, team_id), pts in official.items():
            if game_id not in processed:
                continue
            got = int(team_pts.get((game_id, team_id), 0))
            if got != pts:
                mismatches.append(
                    {"game_id": game_id, "team_id": team_id,
                     "stints": got, "official": pts}
                )
        report["score_checked"] = len(processed & schedule_games)
        report["score_mismatches"] = mismatches
        if mismatches:
            report["problems"].append(
                f"{len(mismatches)} team-game scores disagree with the game log"
            )

    total_poss = int(stints["poss"].sum())
    total_pts = int(stints["points"].sum())
    if total_poss:
        per100 = 100.0 * total_pts / total_poss
        report["pts_per_100"] = round(per100, 2)
        lo, hi = PTS_PER_100_BAND
        # only meaningful with a real sample (a quarter season or so)
        if total_poss >= 20000 and not lo <= per100 <= hi:
            report["problems"].append(
                f"league {per100:.1f} pts/100 outside plausible band [{lo}, {hi}]"
            )
    return report


def validate_many(
    data_dir: Path, seasons: list[str], season_types: list[str]
) -> tuple[list[dict], bool]:
    """Validate each scope; returns (reports, all_ok)."""
    reports = []
    ok = True
    for season in seasons:
        for season_type in season_types:
            report = validate_season(data_dir, season, season_type)
            reports.append(report)
            if report["problems"]:
                ok = False
    return reports, ok
