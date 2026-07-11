"""Per-player, per-season feature scraping: box score + player tracking.

Sources (stats.nba.com, cached raw under data/raw/features/):
  - leaguedashplayerstats MeasureType=Base, PerMode=Per100Possessions
  - leaguedashplayerstats MeasureType=Advanced
  - leaguedashptstats (player tracking: drives, touches, passing, defense,
    rebounding, speed/distance) — available from 2013-14 on; skipped
    automatically for earlier seasons

Output: one wide parquet per season/type under data/features/ with columns
prefixed box_/adv_/pt_<measure>_. Tracking counting stats are converted to
per-minute rates. These feed the SPM prior in prior.py.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from .config import season_end_year, season_type_slug
from .http import get_throttled

logger = logging.getLogger(__name__)

FIRST_TRACKING_SEASON_END_YEAR = 2014
BASE_URL = "https://stats.nba.com/stats/"

# every filter param must be present (even if empty) or the API 400s
_DASH_COMMON = {
    "College": "", "Conference": "", "Country": "", "DateFrom": "",
    "DateTo": "", "Division": "", "DraftPick": "", "DraftYear": "",
    "GameScope": "", "Height": "", "LastNGames": 0, "LeagueID": "00",
    "Location": "", "Month": 0, "OpponentTeamID": 0, "Outcome": "",
    "PORound": 0, "PlayerExperience": "", "PlayerPosition": "",
    "SeasonSegment": "", "StarterBench": "", "TeamID": 0,
    "VsConference": "", "VsDivision": "", "Weight": "",
}

PT_MEASURE_TYPES = [
    "Drives", "Possessions", "Passing", "Defense", "Rebounding",
    "SpeedDistance",
]

# non-feature columns in the dash responses
_DROP_TOKENS = ("_RANK", "FANTASY", "CFID", "CFPARAMS", "WNBA")
_ID_COLS = {
    "PLAYER_ID", "PLAYER_NAME", "NICKNAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "TEAM_NAME", "AGE", "GP", "W", "L", "W_PCT", "MIN",
}


def features_path(data_dir: Path, season: str, season_type: str) -> Path:
    return (
        data_dir / "features"
        / f"{season.replace('-', '_')}_{season_type_slug(season_type)}.parquet"
    )


def _fetch_cached(data_dir: Path, name: str, endpoint: str, params: dict) -> dict:
    cache = data_dir / "raw" / "features" / f"{name}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    response = get_throttled().get(BASE_URL + endpoint, params=params)
    response.raise_for_status()
    payload = response.json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload))
    return payload


def _result_frame(payload: dict) -> pd.DataFrame:
    rs = payload["resultSets"][0]
    return pd.DataFrame(rs["rowSet"], columns=rs["headers"])


def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in _ID_COLS or any(tok in c for tok in _DROP_TOKENS):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _prefixed(df: pd.DataFrame, prefix: str, per_minute: bool = False) -> pd.DataFrame:
    """Reduce a dash response frame to player_id + prefixed feature columns."""
    cols = _feature_cols(df)
    out = df[["PLAYER_ID"] + cols].copy()
    if per_minute:
        minutes = df["MIN"].astype(float).clip(lower=1.0)
        for c in cols:
            # averages/percentages/speeds are already rates
            if not any(tok in c for tok in ("PCT", "AVG", "SPEED", "PER_")):
                out[c] = out[c].astype(float) / minutes * 36.0
    out.columns = ["player_id"] + [f"{prefix}{c}" for c in cols]
    return out


def build_season_features(
    data_dir: Path, season: str, season_type: str, tracking: bool = True
) -> Path:
    """Fetch (or reuse cached) feature endpoints and write the season parquet."""
    slug = f"{season.replace('-', '_')}_{season_type_slug(season_type)}"

    frames = []
    for measure, prefix, per_mode in [
        ("Base", "box_", "Per100Possessions"),
        ("Advanced", "adv_", "Totals"),
    ]:
        params = dict(
            _DASH_COMMON, MeasureType=measure, PerMode=per_mode,
            PlusMinus="N", PaceAdjust="N", Rank="N", Period=0,
            GameSegment="", ShotClockRange="",
            Season=season, SeasonType=season_type,
        )
        payload = _fetch_cached(
            data_dir, f"dash_{measure.lower()}_{slug}",
            "leaguedashplayerstats", params,
        )
        df = _result_frame(payload)
        keep = _prefixed(df, prefix)
        if measure == "Base":
            keep.insert(1, "minutes", df["MIN"].astype(float) * df["GP"].astype(float))
            # metadata, not an SPM feature (fit_spm excludes it): feeds the
            # aging curve for the last-season prior
            keep.insert(2, "age", df["AGE"].astype(float))
        frames.append(keep)

    if tracking and season_end_year(season) >= FIRST_TRACKING_SEASON_END_YEAR:
        for measure in PT_MEASURE_TYPES:
            params = dict(
                _DASH_COMMON, PerMode="Totals", PlayerOrTeam="Player",
                PtMeasureType=measure, Season=season, SeasonType=season_type,
            )
            payload = _fetch_cached(
                data_dir, f"pt_{measure.lower()}_{slug}",
                "leaguedashptstats", params,
            )
            frames.append(
                _prefixed(
                    _result_frame(payload), f"pt_{measure.lower()}_",
                    per_minute=True,
                )
            )
    elif tracking:
        logger.info("%s predates tracking data (2013-14+); box/advanced only", season)

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="player_id", how="outer")

    path = features_path(data_dir, season, season_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    logger.info("Wrote %s (%d players, %d features)", path, len(merged), merged.shape[1] - 1)
    return path


def load_features(
    data_dir: Path, seasons: list[str], season_types: list[str]
) -> pd.DataFrame:
    """Load stored feature parquets; adds season/season_type columns."""
    frames = []
    for season in seasons:
        for season_type in season_types:
            path = features_path(data_dir, season, season_type)
            if not path.exists():
                raise FileNotFoundError(
                    f"No features for {season} {season_type} ({path}). "
                    "Run `fablerapm features` first."
                )
            df = pd.read_parquet(path)
            df["season"] = season
            df["season_type"] = season_type
            frames.append(df)
    return pd.concat(frames, ignore_index=True)
