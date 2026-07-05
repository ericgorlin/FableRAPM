"""Build and maintain the on-disk stint dataset, one file per season/type.

Everything is incremental and resumable: raw API responses are cached by
pbpstats under ``data/raw`` (delete nothing to \"regenerate fresh\" without
re-scraping; delete ``data/raw`` to truly re-scrape), and each
season/season-type gets a parquet of stint rows plus a manifest recording
processed and failed games. Re-running only touches games not yet processed,
so a nightly ``fablerapm build`` keeps the dataset up to date during a season.
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd

from .config import manifest_path, stints_path
from .seasons import season_game_ids
from .stints import STINT_COLUMNS, game_stint_rows

logger = logging.getLogger(__name__)

FLUSH_EVERY = 25


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"processed": {}, "failed": {}}


def _save_manifest(path: Path, manifest: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True))


def _load_stints(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=STINT_COLUMNS)


def build_season(
    data_dir: Path,
    season: str,
    season_type: str,
    game_limit: int | None = None,
    refresh_schedule: bool = False,
    retry_failed: bool = False,
) -> dict:
    """Scrape/parse all missing games for one season & season type.

    Returns a summary dict. ``game_limit`` caps how many games end up in the
    dataset (used by the smoke test to keep API usage tiny).
    """
    spath = stints_path(data_dir, season, season_type)
    mpath = manifest_path(data_dir, season, season_type)
    manifest = _load_manifest(mpath)
    if retry_failed:
        manifest["failed"] = {}

    game_ids = season_game_ids(data_dir, season, season_type, refresh=refresh_schedule)
    if game_limit is not None:
        game_ids = game_ids[:game_limit]

    todo = [
        g for g in game_ids
        if g not in manifest["processed"] and g not in manifest["failed"]
    ]
    logger.info(
        "%s %s: %d games total, %d already processed, %d failed previously, %d to do",
        season, season_type, len(game_ids),
        len(manifest["processed"]), len(manifest["failed"]), len(todo),
    )

    stints = _load_stints(spath)
    new_rows: list[dict] = []
    started = time.monotonic()

    def flush():
        nonlocal stints, new_rows
        if new_rows:
            stints = pd.concat(
                [stints, pd.DataFrame(new_rows, columns=STINT_COLUMNS)],
                ignore_index=True,
            )
            new_rows = []
            spath.parent.mkdir(parents=True, exist_ok=True)
            stints.to_parquet(spath, index=False)
        _save_manifest(mpath, manifest)

    for i, game_id in enumerate(todo, 1):
        try:
            rows, warnings = game_stint_rows(data_dir, game_id)
        except KeyboardInterrupt:
            flush()
            raise
        except Exception as exc:
            logger.warning("%s: failed (%s: %s)", game_id, type(exc).__name__, exc)
            manifest["failed"][game_id] = f"{type(exc).__name__}: {exc}"
            continue
        for warning in warnings:
            logger.warning("%s: %s", game_id, warning)
        manifest["processed"][game_id] = {
            "rows": len(rows),
            "poss": int(sum(r["poss"] for r in rows)),
            "points": int(sum(r["points"] for r in rows)),
            "warnings": warnings,
        }
        new_rows.extend(rows)
        if i % FLUSH_EVERY == 0:
            flush()
            rate = i / (time.monotonic() - started)
            logger.info(
                "%s %s: %d/%d games (%.0f games/min)",
                season, season_type, i, len(todo), rate * 60,
            )
    flush()

    return {
        "season": season,
        "season_type": season_type,
        "games": len(game_ids),
        "processed": len(manifest["processed"]),
        "failed": len(manifest["failed"]),
        "new": len(todo),
        "stint_rows": len(stints),
    }


def build_many(
    data_dir: Path,
    seasons: list[str],
    season_types: list[str],
    game_limit: int | None = None,
    refresh_schedule: bool = False,
    retry_failed: bool = False,
) -> list[dict]:
    summaries = []
    for season in seasons:
        for season_type in season_types:
            summaries.append(
                build_season(
                    data_dir, season, season_type,
                    game_limit=game_limit,
                    refresh_schedule=refresh_schedule,
                    retry_failed=retry_failed,
                )
            )
    return summaries
