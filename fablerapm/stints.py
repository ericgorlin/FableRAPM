"""Extract stint rows (offense lineup vs defense lineup) from play-by-play.

Possession parsing, lineup tracking, and stat attribution are delegated to
pbpstats, which resolves period starters, substitutions during free throws,
and possession-counting conventions exactly from the raw play-by-play.

For each game we aggregate, per (offense lineup, defense lineup) pair:
  - poss:   offensive possessions (pbpstats 'OffPoss', attributed to the
            lineup on the floor for the possession's efficiency event)
  - points: points scored by the offense while that matchup was on the floor
            (pbpstats 'OpponentPoints' recorded against the defense)

Both stats are emitted by pbpstats once per player on the floor (5 per team),
so lineup-level values are the per-player sums divided by 5, which must be
exact.
"""

import logging
from pathlib import Path

from pbpstats import DEFENSIVE_POSSESSION_STRING  # noqa: F401 (documentation)
from pbpstats import OFFENSIVE_POSSESSION_STRING, OPPONENT_POINTS
from pbpstats.data_loader import (
    StatsNbaPossessionFileLoader,
    StatsNbaPossessionLoader,
    StatsNbaPossessionWebLoader,
)

from .config import ensure_raw_dirs, raw_dir
from .http import install_throttle

logger = logging.getLogger(__name__)

STINT_COLUMNS = [
    "game_id",
    "off_team_id",
    "def_team_id",
    "off_lineup",
    "def_lineup",
    "poss",
    "points",
]

_EMPTY_RESULT_SETS = {"resultSets": [{"headers": [], "rowSet": []}]}


class _NoShotsSourceLoader:
    """Stand-in for the shot-chart source loader.

    pbpstats fetches home/away shot charts (2 extra API requests per game)
    solely to attach x/y coordinates to field goal events. Coordinates play
    no part in possession, lineup, or point attribution, so RAPM skips them.
    """

    def load_data(self, game_id):
        return _EMPTY_RESULT_SETS, _EMPTY_RESULT_SETS


def _pbp_cached(data_dir: Path, game_id: str) -> bool:
    return (raw_dir(data_dir) / "pbp" / f"stats_{game_id}.json").exists()


def load_game_possessions(data_dir: Path, game_id: str):
    """Load pbpstats possessions for a game, from disk cache if available."""
    raw = ensure_raw_dirs(data_dir)
    # Throttle even when reading cached files: resolving period starters can
    # fall back to a live boxscore request for a small share of games.
    install_throttle()
    if _pbp_cached(data_dir, game_id):
        source_loader = StatsNbaPossessionFileLoader(str(raw))
    else:
        source_loader = StatsNbaPossessionWebLoader(str(raw))
    source_loader.enhanced_pbp_source_loader.shots_source_loader = (
        _NoShotsSourceLoader()
    )
    return StatsNbaPossessionLoader(game_id, source_loader).items


def possessions_to_stint_rows(possessions, game_id: str) -> list[dict]:
    """Aggregate pbpstats possessions into per-game stint rows.

    Pure function of possession objects (each exposing ``possession_stats``),
    so it is testable without network or files.
    """
    poss_sums: dict[tuple, int] = {}
    point_sums: dict[tuple, int] = {}
    for possession in possessions:
        for stat in possession.possession_stats:
            key_off = None
            if stat["stat_key"] == OFFENSIVE_POSSESSION_STRING:
                # team_id is the offense
                key_off = (
                    stat["team_id"],
                    stat["lineup_id"],
                    stat["opponent_team_id"],
                    stat["opponent_lineup_id"],
                )
                poss_sums[key_off] = poss_sums.get(key_off, 0) + stat["stat_value"]
            elif stat["stat_key"] == OPPONENT_POINTS:
                # team_id is the team scored against; the offense is the opponent
                key_off = (
                    stat["opponent_team_id"],
                    stat["opponent_lineup_id"],
                    stat["team_id"],
                    stat["lineup_id"],
                )
                point_sums[key_off] = point_sums.get(key_off, 0) + stat["stat_value"]

    rows = []
    for key in poss_sums.keys() | point_sums.keys():
        off_team_id, off_lineup, def_team_id, def_lineup = key
        poss_x5 = poss_sums.get(key, 0)
        points_x5 = point_sums.get(key, 0)
        if poss_x5 % 5 != 0 or points_x5 % 5 != 0:
            raise ValueError(
                f"Game {game_id}: lineup stats not divisible by 5 players "
                f"for {key}: poss={poss_x5}, points={points_x5}"
            )
        if off_lineup.count("-") != 4 or def_lineup.count("-") != 4:
            raise ValueError(
                f"Game {game_id}: lineup without exactly 5 players: "
                f"off={off_lineup!r} def={def_lineup!r}"
            )
        rows.append(
            {
                "game_id": game_id,
                "off_team_id": off_team_id,
                "def_team_id": def_team_id,
                "off_lineup": off_lineup,
                "def_lineup": def_lineup,
                "poss": poss_x5 // 5,
                "points": points_x5 // 5,
            }
        )
    return rows


def check_stint_rows(possessions, rows: list[dict]) -> list[str]:
    """Cross-check stint totals against the final score in the play-by-play.

    Returns a list of warning strings (empty when everything reconciles).
    """
    warnings = []
    final_score = possessions[-1].events[-1].score  # {team_id: points}
    stint_points = sum(r["points"] for r in rows)
    pbp_points = sum(final_score.values())
    if stint_points != pbp_points:
        warnings.append(
            f"stint points {stint_points} != final score total {pbp_points}"
        )
    return warnings


def game_stint_rows(data_dir: Path, game_id: str) -> tuple[list[dict], list[str]]:
    """Fetch/parse one game and return (stint rows, warnings)."""
    possessions = load_game_possessions(data_dir, game_id)
    rows = possessions_to_stint_rows(possessions, game_id)
    warnings = check_stint_rows(possessions, rows)
    return rows, warnings
