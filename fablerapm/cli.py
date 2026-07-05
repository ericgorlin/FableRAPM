"""Command-line interface.

    fablerapm build       scrape & store stint data (incremental, resumable)
    fablerapm rapm        fit RAPM from stored stints and write CSVs
    fablerapm smoke-test  tiny end-to-end run against the live API
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    DEFAULT_SEASON_TYPES,
    default_data_dir,
    parse_season_types,
    parse_seasons,
)

logger = logging.getLogger(__name__)


def _add_common(parser):
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Data directory (default: ./data, or $FABLERAPM_DATA_DIR)",
    )
    parser.add_argument(
        "--seasons", default="all",
        help="'all', '2019-20', '1996-97:2005-06', or comma-separated mix",
    )
    parser.add_argument(
        "--season-types", default=",".join(DEFAULT_SEASON_TYPES),
        help="Comma-separated: regular, playoffs, playin (default: regular,playoffs)",
    )


def _resolve_common(args):
    data_dir = args.data_dir or default_data_dir()
    return data_dir, parse_seasons(args.seasons), parse_season_types(args.season_types)


def cmd_build(args) -> int:
    from .build import build_many

    data_dir, seasons, season_types = _resolve_common(args)
    logger.info(
        "Building stint data for %d season(s) x %s into %s "
        "(a full 30-season build makes ~1 API request/sec and takes hours; "
        "it is resumable, so interrupting is safe)",
        len(seasons), season_types, data_dir,
    )
    summaries = build_many(
        data_dir, seasons, season_types,
        game_limit=args.game_limit,
        refresh_schedule=args.refresh_schedule,
        retry_failed=args.retry_failed,
    )
    failed_total = 0
    for s in summaries:
        print(
            f"{s['season']} {s['season_type']}: {s['processed']}/{s['games']} games, "
            f"{s['stint_rows']} stint rows, {s['failed']} failed"
        )
        failed_total += s["failed"]
    if failed_total:
        print(
            f"\n{failed_total} game(s) failed to parse; see the .manifest.json files "
            "under data/stints for details. Some can be fixed with pbpstats override "
            "files in data/raw/overrides (see README), then re-run with --retry-failed."
        )
    return 0


def cmd_rapm(args) -> int:
    from .model import run_rapm

    data_dir, seasons, season_types = _resolve_common(args)
    lam = "cv" if args.ridge_lambda == "cv" else float(args.ridge_lambda)
    written = run_rapm(
        data_dir, seasons, season_types,
        pool_seasons=args.pool,
        combine_types=args.combine_types,
        lam=lam,
        min_poss=args.min_poss,
        fetch_names=not args.no_names,
        out_dir=args.out_dir,
        prior_kind=args.prior,
        prior_scale=args.prior_scale,
    )
    for path in written:
        print(f"wrote {path}")
    return 0


def cmd_features(args) -> int:
    from .features import build_season_features

    data_dir, seasons, season_types = _resolve_common(args)
    for season in seasons:
        for season_type in season_types:
            path = build_season_features(
                data_dir, season, season_type, tracking=not args.no_tracking
            )
            print(f"wrote {path}")
    return 0


def cmd_spm_train(args) -> int:
    from .prior import train_spm

    data_dir, seasons, season_types = _resolve_common(args)
    lam = "cv" if args.ridge_lambda == "cv" else float(args.ridge_lambda)
    path = train_spm(data_dir, seasons, season_types, lam=lam)
    print(f"wrote {path}")
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import evaluate_variants
    from .model import load_stints
    from .prior import spm_model_path, spm_prior, two_phase_prior

    data_dir, seasons, season_types = _resolve_common(args)
    stints = load_stints(data_dir, seasons, season_types)
    lam = "cv" if args.ridge_lambda == "cv" else float(args.ridge_lambda)

    variants = {"plain": None}
    for scale in args.two_phase_scales:
        variants[f"two-phase(scale={scale})"] = (
            lambda train, s=scale: two_phase_prior(train, lam=lam, scale=s)
        )
    if args.spm:
        if not spm_model_path(data_dir).exists():
            print("No SPM model found; run `fablerapm spm-train` first")
            return 1
        # SPM prior comes from features + a saved model, independent of the
        # train/holdout stint split, so it can't leak holdout outcomes
        variants["spm"] = lambda train: spm_prior(data_dir, seasons, season_types)

    table = evaluate_variants(
        stints, variants, lam=lam, test_frac=args.test_frac, seed=args.seed
    )
    print(table.to_string(index=False, float_format="%.4f"))
    return 0


def cmd_smoke_test(args) -> int:
    from .smoke import run_smoke_test

    data_dir = args.data_dir or (default_data_dir() / "smoke")
    return run_smoke_test(
        data_dir=data_dir,
        season=args.season,
        regular_games=args.regular_games,
        playoff_games=args.playoff_games,
    )


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        prog="fablerapm",
        description="NBA RAPM pipeline (exact possession stints + ridge regression)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Scrape & store stint data (incremental)")
    _add_common(p_build)
    p_build.add_argument(
        "--game-limit", type=int, default=None,
        help="Only process the first N games per season/type (for testing)",
    )
    p_build.add_argument(
        "--refresh-schedule", action="store_true",
        help="Refetch season schedules even if cached",
    )
    p_build.add_argument(
        "--retry-failed", action="store_true",
        help="Retry games that failed to parse on previous runs",
    )
    p_build.set_defaults(func=cmd_build)

    p_rapm = sub.add_parser("rapm", help="Fit RAPM from stored stints, write CSVs")
    _add_common(p_rapm)
    p_rapm.add_argument(
        "--pool", action="store_true",
        help="Pool all requested seasons into one regression "
        "(default: one output per season)",
    )
    p_rapm.add_argument(
        "--combine-types", action="store_true",
        help="Pool regular season + playoffs together "
        "(default: separate outputs per season type)",
    )
    p_rapm.add_argument(
        "--lambda", dest="ridge_lambda", default="cv",
        help="Ridge strength: 'cv' (game-grouped cross-validation, default) "
        "or a fixed number",
    )
    p_rapm.add_argument(
        "--min-poss", type=float, default=0,
        help="Drop players below this many possessions from the OUTPUT "
        "(they always stay in the regression)",
    )
    p_rapm.add_argument(
        "--no-names", action="store_true",
        help="Skip fetching player names (fully offline)",
    )
    p_rapm.add_argument("--out-dir", type=Path, default=None)
    p_rapm.add_argument(
        "--prior", choices=["none", "two-phase", "spm"], default="none",
        help="Shrinkage target: 'two-phase' uses a first-pass RAPM (helps "
        "star compression), 'spm' uses the trained box/tracking model",
    )
    p_rapm.add_argument(
        "--prior-scale", type=float, default=1.0,
        help="Scale applied to the two-phase prior (default 1.0)",
    )
    p_rapm.set_defaults(func=cmd_rapm)

    p_feat = sub.add_parser(
        "features",
        help="Scrape per-player box score + tracking features (for SPM prior)",
    )
    _add_common(p_feat)
    p_feat.add_argument(
        "--no-tracking", action="store_true",
        help="Skip player-tracking endpoints (tracking exists 2013-14+)",
    )
    p_feat.set_defaults(func=cmd_features)

    p_spm = sub.add_parser(
        "spm-train",
        help="Train the SPM prior model from stored features + stint data",
    )
    _add_common(p_spm)
    p_spm.add_argument("--lambda", dest="ridge_lambda", default="cv")
    p_spm.set_defaults(func=cmd_spm_train)

    p_eval = sub.add_parser(
        "evaluate",
        help="Compare RAPM variants by weighted MSE on held-out games",
    )
    _add_common(p_eval)
    p_eval.add_argument("--lambda", dest="ridge_lambda", default="cv")
    p_eval.add_argument(
        "--two-phase-scales", type=float, nargs="*", default=[1.0],
        help="Two-phase prior scales to evaluate (default: 1.0)",
    )
    p_eval.add_argument(
        "--spm", action="store_true",
        help="Also evaluate the trained SPM prior (needs spm-train first)",
    )
    p_eval.add_argument("--test-frac", type=float, default=0.2)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.set_defaults(func=cmd_evaluate)

    p_smoke = sub.add_parser(
        "smoke-test",
        help="End-to-end test on a handful of live-API games (~8 requests)",
    )
    p_smoke.add_argument("--data-dir", type=Path, default=None)
    p_smoke.add_argument("--season", default="2023-24")
    p_smoke.add_argument("--regular-games", type=int, default=3)
    p_smoke.add_argument("--playoff-games", type=int, default=2)
    p_smoke.set_defaults(func=cmd_smoke_test)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
