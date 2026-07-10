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
import multiprocessing
import time
from pathlib import Path

import pandas as pd

from .config import manifest_path, stints_path
from .seasons import season_game_ids
from .stints import STINT_COLUMNS, _pbp_cached, game_stint_rows

logger = logging.getLogger(__name__)

FLUSH_EVERY = 25


def _parse_game_worker(args: tuple) -> tuple:
    """Parse one cached game in a worker process.

    Returns (game_id, rows, warnings, error): rows is None when parsing
    failed and error carries the message. Exceptions never propagate — a
    raised exception would poison the pool and lose the other results.
    """
    data_dir, game_id = args
    try:
        rows, warnings = game_stint_rows(Path(data_dir), game_id)
        return game_id, rows, warnings, None
    except Exception as exc:  # noqa: BLE001 - recorded in the manifest
        return game_id, None, [], f"{type(exc).__name__}: {exc}"


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"processed": {}, "failed": {}}


def _save_manifest(path: Path, manifest: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    tmp.replace(path)


def _load_stints(path: Path, manifest: dict) -> pd.DataFrame:
    """Load stored stints, reconciled against the manifest.

    The parquet is written before the manifest, so after a hard crash the
    parquet may contain games the manifest doesn't record as processed.
    Dropping those rows here makes re-processing them safe (no duplicates).
    """
    if not path.exists():
        return pd.DataFrame(columns=STINT_COLUMNS)
    stints = pd.read_parquet(path)
    missing_cols = [c for c in STINT_COLUMNS if c not in stints.columns]
    if missing_cols:
        logger.warning(
            "%s was written by an older schema (missing %s); new games get "
            "the new columns but old rows stay NaN. Delete data/stints and "
            "re-run build (offline re-parse from the raw cache) for a "
            "uniform dataset — required before using fit-time garbage "
            "rules on these seasons.",
            path.name, ", ".join(missing_cols),
        )
    known = stints["game_id"].isin(manifest["processed"])
    if not known.all():
        logger.warning(
            "Dropping %d stint rows from %d game(s) not in the manifest "
            "(likely an interrupted run); they will be re-processed",
            int((~known).sum()), stints.loc[~known, "game_id"].nunique(),
        )
        stints = stints.loc[known].reset_index(drop=True)
    return stints


def build_season(
    data_dir: Path,
    season: str,
    season_type: str,
    game_limit: int | None = None,
    refresh_schedule: bool = False,
    retry_failed: bool = False,
    workers: int = 1,
) -> dict:
    """Scrape/parse all missing games for one season & season type.

    Returns a summary dict. ``game_limit`` caps how many games end up in the
    dataset (used by the smoke test to keep API usage tiny).

    ``workers > 1`` parallelizes games whose play-by-play is already in the
    raw cache (pure-Python re-parsing dominates there — the whole point of
    a cache rebuild); uncached games always go through the serial throttled
    path, since the API rate limit is global, not per-process. Caveat: a
    small share of cached games fall back to a live boxscore request to
    resolve period starters, so up to ``workers`` requests can occasionally
    fire concurrently.
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

    stints = _load_stints(spath, manifest)
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
            tmp = spath.with_suffix(".tmp")
            stints.to_parquet(tmp, index=False)
            tmp.replace(spath)
        _save_manifest(mpath, manifest)

    def record(game_id, rows, warnings, error):
        if error is not None:
            logger.warning("%s: failed (%s)", game_id, error)
            manifest["failed"][game_id] = error
            return
        for warning in warnings:
            logger.warning("%s: %s", game_id, warning)
        manifest["processed"][game_id] = {
            "rows": len(rows),
            "poss": int(sum(r["poss"] for r in rows)),
            "points": int(sum(r["points"] for r in rows)),
            "warnings": warnings,
        }
        new_rows.extend(rows)

    def progress(i, total, label):
        rate = i / (time.monotonic() - started)
        logger.info(
            "%s %s: %d/%d %s games (%.0f games/min)",
            season, season_type, i, total, label, rate * 60,
        )

    cached = [g for g in todo if _pbp_cached(data_dir, g)] if workers > 1 else []
    cached_set = set(cached)
    serial = [g for g in todo if g not in cached_set]

    if cached:
        logger.info(
            "%s %s: re-parsing %d cached games with %d workers "
            "(%d uncached games follow serially)",
            season, season_type, len(cached), workers, len(serial),
        )
        with multiprocessing.Pool(workers) as pool:
            results = pool.imap_unordered(
                _parse_game_worker, [(str(data_dir), g) for g in cached]
            )
            try:
                for i, (game_id, rows, warnings, error) in enumerate(results, 1):
                    record(game_id, rows, warnings, error)
                    if i % FLUSH_EVERY == 0:
                        flush()
                        progress(i, len(cached), "cached")
            except KeyboardInterrupt:
                pool.terminate()
                flush()
                raise
        flush()

    for i, game_id in enumerate(serial, 1):
        try:
            rows, warnings = game_stint_rows(data_dir, game_id)
        except KeyboardInterrupt:
            flush()
            raise
        except Exception as exc:
            record(game_id, None, [], f"{type(exc).__name__}: {exc}")
            continue
        record(game_id, rows, warnings, None)
        if i % FLUSH_EVERY == 0:
            flush()
            progress(i, len(serial), "serial")
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
    workers: int = 1,
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
                    workers=workers,
                )
            )
    return summaries
