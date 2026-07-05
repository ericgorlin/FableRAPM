"""Small end-to-end smoke test against the live stats.nba.com API.

Fetches a handful of games (default: 3 regular season + 2 playoff games of
one season, ~8 API requests total including schedules and player names),
runs the entire pipeline — scrape, possession parsing, stint extraction,
storage, RAPM fit — and asserts basic invariants at each stage.
"""

import logging
from pathlib import Path

import pandas as pd

from .build import build_many
from .config import SEASON_TYPES, stints_path
from .model import fit_rapm, load_stints, run_rapm

logger = logging.getLogger(__name__)


def run_smoke_test(
    data_dir: Path,
    season: str = "2023-24",
    regular_games: int = 3,
    playoff_games: int = 2,
) -> int:
    checks: list[str] = []

    def check(condition: bool, description: str):
        status = "ok" if condition else "FAIL"
        checks.append(f"[{status}] {description}")
        print(checks[-1])
        if not condition:
            raise AssertionError(description)

    print(f"Smoke test: {season}, {regular_games} regular season "
          f"+ {playoff_games} playoff games -> {data_dir}\n")

    plan = []
    if regular_games:
        plan.append((SEASON_TYPES["regular"], regular_games))
    if playoff_games:
        plan.append((SEASON_TYPES["playoffs"], playoff_games))
    season_types = [t for t, _ in plan]

    # 1. Build (scrape + parse + store), tiny game limits
    for season_type, limit in plan:
        summaries = build_many(data_dir, [season], [season_type], game_limit=limit)
        s = summaries[0]
        check(s["failed"] == 0, f"{season_type}: no games failed to parse")
        check(
            s["processed"] >= limit,
            f"{season_type}: processed {s['processed']}/{limit} games",
        )
        path = stints_path(data_dir, season, season_type)
        check(path.exists(), f"{season_type}: stint parquet written ({path.name})")
        df = pd.read_parquet(path)
        check(len(df) > 0, f"{season_type}: {len(df)} stint rows extracted")
        check(
            (df["off_lineup"].str.count("-") == 4).all()
            and (df["def_lineup"].str.count("-") == 4).all(),
            f"{season_type}: every stint has 5-player lineups",
        )
        per_game = df.groupby("game_id")[["poss", "points"]].sum()
        check(
            bool(per_game["poss"].between(150, 300).all()),
            f"{season_type}: total possessions per game in a sane range "
            f"({per_game['poss'].min()}-{per_game['poss'].max()})",
        )
        check(
            bool(per_game["points"].between(120, 350).all()),
            f"{season_type}: total points per game in a sane range "
            f"({per_game['points'].min()}-{per_game['points'].max()})",
        )

    # 2. Fit RAPM directly (fixed lambda; CV is meaningless on a few games)
    stints = load_stints(data_dir, [season], season_types)
    result = fit_rapm(stints, lam=500.0)
    players = result.players
    check(len(players) >= 60, f"RAPM covers {len(players)} players")
    check(
        players[["orapm", "drapm", "rapm"]].notna().all().all()
        and float(players["rapm"].abs().max()) < 50,
        "RAPM estimates are finite and bounded",
    )
    check(
        60 < result.meta["avg_points_per_100"] < 160,
        f"League average ORtg plausible "
        f"({result.meta['avg_points_per_100']:.1f} pts/100)",
    )

    # 3. Full run_rapm path (pooled across types), including name lookup
    written = run_rapm(
        data_dir, [season], season_types,
        pool_seasons=True, combine_types=True, lam=500.0,
    )
    check(len(written) == 1, "run_rapm wrote a combined CSV")
    out = pd.read_csv(written[0])
    expected_cols = [
        "player_id", "player_name", "orapm", "drapm", "rapm", "off_poss", "def_poss",
    ]
    check(list(out.columns) == expected_cols, f"CSV columns are {expected_cols}")
    named = (out["player_name"].fillna("").str.len() > 0).mean()
    check(named > 0.9, f"{named:.0%} of players have names resolved")

    print(f"\nTop 10 by possessions ({written[0]}):")
    top = out.sort_values("off_poss", ascending=False).head(10)
    print(
        top.to_string(
            index=False,
            formatters={c: "{:.2f}".format for c in ("orapm", "drapm", "rapm")},
        )
    )
    print(f"\nSMOKE TEST PASSED ({len(checks)} checks)")
    return 0
