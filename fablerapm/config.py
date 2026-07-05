"""Paths, season utilities, and defaults."""

import datetime
import os
from pathlib import Path

# stats.nba.com play-by-play (with substitutions) exists from 1996-97 onward.
# A handful of mid/late-90s games have broken pbp; those are skipped and
# recorded in the per-season manifest.
FIRST_PBP_SEASON_END_YEAR = 1997

SEASON_TYPES = {
    "regular": "Regular Season",
    "playoffs": "Playoffs",
    "playin": "Play In",
}
DEFAULT_SEASON_TYPES = ["regular", "playoffs"]

# New season's games start in October; before that, "current" is the season
# that ended in the spring.
SEASON_ROLLOVER_MONTH = 10


def default_data_dir() -> Path:
    env = os.environ.get("FABLERAPM_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data"


def current_season_end_year(today: datetime.date | None = None) -> int:
    today = today or datetime.date.today()
    return today.year + 1 if today.month >= SEASON_ROLLOVER_MONTH else today.year


def season_str(end_year: int) -> str:
    """1997 -> '1996-97'"""
    return f"{end_year - 1}-{str(end_year)[-2:]}"


def season_end_year(season: str) -> int:
    """'1996-97' -> 1997"""
    return int(season.split("-")[0]) + 1


def parse_seasons(spec: str) -> list[str]:
    """Parse a seasons spec into a list of season strings.

    Accepts 'all' (1996-97 through current), a single season '2019-20',
    a range '2005-06:2010-11', or a comma-separated mix.
    """
    seasons: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower() == "all":
            start, end = FIRST_PBP_SEASON_END_YEAR, current_season_end_year()
        elif ":" in part:
            lo, hi = part.split(":")
            start, end = season_end_year(lo.strip()), season_end_year(hi.strip())
        else:
            start = end = season_end_year(part)
        for y in range(start, end + 1):
            s = season_str(y)
            if s not in seasons:
                seasons.append(s)
    return seasons


def parse_season_types(spec: str) -> list[str]:
    """Parse comma-separated season type keys into pbpstats season type strings."""
    out = []
    for part in spec.split(","):
        key = part.strip().lower().replace(" ", "").replace("-", "")
        if key in ("regularseason",):
            key = "regular"
        if key in ("playin",):
            key = "playin"
        if key not in SEASON_TYPES:
            raise ValueError(
                f"Unknown season type {part!r}; options: {', '.join(SEASON_TYPES)}"
            )
        if SEASON_TYPES[key] not in out:
            out.append(SEASON_TYPES[key])
    return out


def season_type_slug(season_type: str) -> str:
    return season_type.replace(" ", "").lower()


def stints_path(data_dir: Path, season: str, season_type: str) -> Path:
    return (
        data_dir
        / "stints"
        / f"{season.replace('-', '_')}_{season_type_slug(season_type)}.parquet"
    )


def manifest_path(data_dir: Path, season: str, season_type: str) -> Path:
    return (
        data_dir
        / "stints"
        / f"{season.replace('-', '_')}_{season_type_slug(season_type)}.manifest.json"
    )


def raw_dir(data_dir: Path) -> Path:
    """Directory where pbpstats caches raw API responses."""
    return data_dir / "raw"


def results_dir(data_dir: Path) -> Path:
    return data_dir / "results"


def ensure_raw_dirs(data_dir: Path) -> Path:
    """Create the subdirectory layout pbpstats expects for its file cache."""
    raw = raw_dir(data_dir)
    for sub in ("pbp", "schedule", "overrides", "game_details"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    return raw
