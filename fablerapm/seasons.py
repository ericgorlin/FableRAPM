"""Enumerate final game ids for a season/season-type via the league game log."""

import logging
from pathlib import Path

from pbpstats.client import Client

from .config import current_season_end_year, ensure_raw_dirs, raw_dir, season_end_year, season_str
from .http import install_throttle

logger = logging.getLogger(__name__)


def _schedule_cached(data_dir: Path, season: str, season_type: str) -> bool:
    name = (
        f"stats_leaguegamelog_nba_{season.replace('-', '_')}_"
        f"{season_type.replace(' ', '_')}.json"
    )
    return (raw_dir(data_dir) / "schedule" / name).exists()


def season_game_ids(
    data_dir: Path, season: str, season_type: str, refresh: bool = False
) -> list[str]:
    """Return sorted final game ids. Uses the cached schedule for past
    seasons; the current season is refetched so new games show up."""
    ensure_raw_dirs(data_dir)
    is_current = season_end_year(season) >= current_season_end_year()
    use_file = _schedule_cached(data_dir, season, season_type) and not refresh and not is_current
    source = "file" if use_file else "web"
    if source == "web":
        install_throttle()
    client = Client(
        {
            "dir": str(raw_dir(data_dir)),
            "Games": {"source": source, "data_provider": "stats_nba"},
        }
    )
    games = client.Season("nba", season, season_type).games.items
    game_ids = sorted({g.game_id for g in games if g.is_final})
    logger.info("%s %s: %d final games", season, season_type, len(game_ids))
    return game_ids


def latest_completed_season() -> str:
    return season_str(current_season_end_year())
