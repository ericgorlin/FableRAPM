"""Player id -> display name, via stats.nba.com commonallplayers (cached)."""

import json
import logging
import time
from pathlib import Path

from .http import get_throttled

logger = logging.getLogger(__name__)

COMMON_ALL_PLAYERS_URL = "https://stats.nba.com/stats/commonallplayers"
CACHE_MAX_AGE_SECONDS = 7 * 24 * 3600


def _cache_path(data_dir: Path) -> Path:
    return data_dir / "raw" / "commonallplayers.json"


def _fetch(season: str) -> dict:
    response = get_throttled().get(
        COMMON_ALL_PLAYERS_URL,
        params={"IsOnlyCurrentSeason": 0, "LeagueID": "00", "Season": season},
    )
    response.raise_for_status()
    return response.json()


def _parse(payload: dict) -> dict[int, str]:
    result_set = payload["resultSets"][0]
    headers = result_set["headers"]
    id_idx = headers.index("PERSON_ID")
    name_idx = headers.index("DISPLAY_FIRST_LAST")
    return {row[id_idx]: row[name_idx] for row in result_set["rowSet"]}


def get_player_names(
    data_dir: Path,
    player_ids: set[int],
    season: str,
    allow_fetch: bool = True,
) -> dict[int, str]:
    """Return a name for every requested player id.

    Uses a cached commonallplayers response; refetches when ids are missing
    and the cache is stale. Falls back to 'Player <id>' so name lookups can
    never break a run (e.g. offline).
    """
    cache = _cache_path(data_dir)
    names: dict[int, str] = {}
    if cache.exists():
        try:
            names = _parse(json.loads(cache.read_text()))
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Ignoring unreadable name cache %s: %s", cache, exc)

    missing = player_ids - names.keys()
    cache_stale = (
        not cache.exists()
        or time.time() - cache.stat().st_mtime > CACHE_MAX_AGE_SECONDS
    )
    if missing and cache_stale and allow_fetch:
        try:
            payload = _fetch(season)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(payload))
            names = _parse(payload)
        except Exception as exc:
            logger.warning("Could not fetch player names: %s", exc)

    return {pid: names.get(pid, f"Player {pid}") for pid in player_ids}
