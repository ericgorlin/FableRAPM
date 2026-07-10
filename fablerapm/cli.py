"""Command-line interface.

    fablerapm build       scrape & store stint data (incremental, resumable)
    fablerapm rapm        fit RAPM from stored stints and write CSVs
    fablerapm smoke-test  tiny end-to-end run against the live API
"""

import argparse
import json
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
        workers=args.workers,
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


def _parse_garbage_rule(spec: str | None) -> tuple[float, float] | None:
    if spec is None:
        return None
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        raise SystemExit(
            f"--garbage-rule expects 'BASE,PER_MINUTE' (e.g. '12,1.5'), "
            f"got {spec!r}"
        )
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        raise SystemExit(
            f"--garbage-rule expects two numbers 'BASE,PER_MINUTE', got {spec!r}"
        )


def cmd_rapm(args) -> int:
    from .model import run_rapm

    data_dir, seasons, season_types = _resolve_common(args)
    lam = "cv" if args.ridge_lambda == "cv" else float(args.ridge_lambda)
    kwargs = dict(
        lam=lam,
        prior_kind=args.prior,
        prior_scale=args.prior_scale,
        decay=args.decay,
        playoff_weight=args.playoff_weight,
        garbage_weight=args.garbage_weight,
        garbage_rule=_parse_garbage_rule(args.garbage_rule),
        interactions=args.interactions,
        offense_curvature=args.offense_curvature,
        defense_curvature=args.defense_curvature,
    )
    if args.tuned:
        from .tune import load_tuned_config

        cfg = load_tuned_config(data_dir)
        rule = cfg.get("garbage_rule")
        kwargs = dict(
            lam=cfg["lambda"],
            prior_kind=cfg["prior"],
            prior_scale=cfg["prior_scale"],
            decay=cfg["decay"],
            playoff_weight=cfg["playoff_weight"],
            garbage_weight=cfg["garbage_weight"],
            garbage_rule=tuple(rule) if rule else None,
            interactions=cfg["interactions"],
            offense_curvature=cfg["offense_curvature"],
            defense_curvature=cfg["defense_curvature"],
        )
        print(f"using tuned config: {kwargs}")
    written = run_rapm(
        data_dir, seasons, season_types,
        pool_seasons=args.pool,
        combine_types=args.combine_types,
        min_poss=args.min_poss,
        fetch_names=not args.no_names,
        out_dir=args.out_dir,
        **kwargs,
    )
    for path in written:
        print(f"wrote {path}")
    return 0


def cmd_tune(args) -> int:
    from .tune import tune

    data_dir, seasons, season_types = _resolve_common(args)
    lam = "cv" if args.ridge_lambda == "cv" else float(args.ridge_lambda)
    result = tune(
        data_dir, seasons, season_types,
        n_seeds=args.seeds, test_frac=args.test_frac,
        passes=args.passes, lam=lam,
        split=args.split, outer_frac=args.outer_frac,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=1))
    if "outer_mse" in result:
        print(
            f"untouched outer games: tuned {result['outer_mse']:.4f} vs "
            f"default {result['outer_default_mse']:.4f} vs intercept-only "
            f"{result['outer_intercept_mse']:.4f} (lower is better; this is "
            "the unbiased number — the inner MSE selected the winner)"
        )
    print("use it with: fablerapm rapm --tuned")
    return 0


def cmd_calibrate_curvature(args) -> int:
    from .calibrate import format_report, run_calibration

    data_dir, seasons, season_types = _resolve_common(args)
    lam = "cv" if args.ridge_lambda == "cv" else float(args.ridge_lambda)
    report, path = run_calibration(
        data_dir, seasons, season_types,
        lam=lam, n_sims=args.sims, seed=args.seed, alpha=args.alpha,
        null=args.null,
        offense_curvature=args.offense_curvature,
        defense_curvature=args.defense_curvature,
        decay=args.decay, playoff_weight=args.playoff_weight,
        garbage_weight=args.garbage_weight,
    )
    print(format_report(report))
    print(f"wrote {path}")
    return 0


def cmd_validate(args) -> int:
    from .validate import validate_many

    data_dir, seasons, season_types = _resolve_common(args)
    reports, ok = validate_many(data_dir, seasons, season_types)
    for r in reports:
        status = "OK  " if not r["problems"] else "FAIL"
        print(
            f"[{status}] {r['season']} {r['season_type']}: "
            f"{r.get('games_processed', 0)} games, "
            f"{r.get('games_failed', 0)} failed, "
            f"{len(r.get('score_mismatches', []))}/{r.get('score_checked', 0)} "
            f"score mismatches, {r.get('pts_per_100', float('nan'))} pts/100"
        )
        for problem in r["problems"]:
            print(f"       - {problem}")
        for m in r.get("score_mismatches", [])[:5]:
            print(f"       - {m}")
    print("\nVALIDATION " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


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
    path = train_spm(
        data_dir, seasons, season_types, lam=lam,
        garbage_weight=args.garbage_weight,
    )
    print(f"wrote {path}")
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import evaluate_variants
    from .model import load_stints
    from .prior import (
        last_season_prior,
        spm_model_path,
        spm_prior,
        two_phase_prior,
    )

    data_dir, seasons, season_types = _resolve_common(args)
    stints = load_stints(data_dir, seasons, season_types)
    lam = "cv" if args.ridge_lambda == "cv" else float(args.ridge_lambda)

    variants = {"plain": None}
    for scale in args.two_phase_scales:
        variants[f"two-phase(scale={scale})"] = (
            lambda train, s=scale: two_phase_prior(train, lam=lam, scale=s)
        )
    for scale in args.last_season_scales:
        # built from the season before the evaluated scope: disjoint data,
        # independent of the train/holdout split
        variants[f"last-season(scale={scale})"] = (
            lambda train, s=scale: last_season_prior(
                data_dir, seasons, season_types, lam=lam, scale=s
            )
        )
    if args.spm:
        if not spm_model_path(data_dir).exists():
            print("No SPM model found; run `fablerapm spm-train` first")
            return 1
        # SPM prior comes from features + a saved model, independent of the
        # train/holdout stint split, so it can't leak holdout outcomes
        variants["spm"] = lambda train: spm_prior(data_dir, seasons, season_types)
    if args.interactions:
        variants["interactions"] = {"prior_fn": None, "interactions": True}

    import pandas as pd

    tables = []
    for decay in args.decays:
        for gw in args.garbage_weights:
            for pw in args.playoff_weights:
                t = evaluate_variants(
                    stints, variants, lam=lam,
                    test_frac=args.test_frac, seed=args.seed,
                    decay=decay, playoff_weight=pw, garbage_weight=gw,
                    score_garbage=not args.holdout_no_garbage,
                    split=args.split,
                )
                t.insert(1, "decay", decay)
                t.insert(2, "garbage_wt", gw)
                t.insert(3, "playoff_wt", pw)
                tables.append(t)
    table = pd.concat(tables, ignore_index=True).sort_values(
        "holdout_mse", ignore_index=True
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
    p_build.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers for games already in the raw cache (pure "
        "re-parse, e.g. after a schema change or deleting data/stints). "
        "Uncached games are always fetched serially at ~1 req/s",
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
        "--prior", choices=["none", "two-phase", "spm", "last-season"],
        default="none",
        help="Shrinkage target: 'two-phase' uses a first-pass RAPM (helps "
        "star compression), 'spm' uses the trained box/tracking model, "
        "'last-season' uses the previous season's RAPM",
    )
    p_rapm.add_argument(
        "--prior-scale", type=float, default=1.0,
        help="Scale applied to two-phase/last-season priors (default 1.0; "
        "~0.5-0.8 is sensible for last-season)",
    )
    p_rapm.add_argument(
        "--decay", type=float, default=1.0,
        help="Per-season weight decay for pooled multi-season fits: stint "
        "weight *= decay ** years_before_most_recent (default 1.0 = equal)",
    )
    p_rapm.add_argument(
        "--playoff-weight", type=float, default=1.5,
        help="Weight multiplier for playoff/play-in stints when season "
        "types are pooled with --combine-types (default 1.5; has no effect "
        "on single-type fits). Tune with `evaluate --playoff-weights`",
    )
    p_rapm.add_argument(
        "--garbage-weight", type=float, default=1.0,
        help="Weight multiplier for garbage-time stints: 1.0 keeps (default), "
        "0 drops, in between downweights",
    )
    p_rapm.add_argument(
        "--garbage-rule", default=None, metavar="BASE,PER_MINUTE",
        help="Fit-time garbage definition replacing the parse-time tiers: "
        "a Q4/OT possession is garbage when |margin| >= BASE + PER_MINUTE "
        "x minutes left in the period (e.g. '12,1.5'). Needs stints parsed "
        "with the margin/secs_left schema; only matters with "
        "--garbage-weight != 1",
    )
    p_rapm.add_argument(
        "--interactions", action="store_true",
        help="Nonlinear two-phase RAPM: learned talent-concentration terms "
        "so diminishing returns of stacked lineups aren't deducted from "
        "stars",
    )
    p_rapm.add_argument(
        "--offense-curvature",
        choices=["diminishing", "free", "none"], default="diminishing",
        help="Sign constraint on offensive concentration terms: diminishing "
        "(concave only, default — robust to the shrinkage artifact), free "
        "(learn sign from data; validate with evaluate), none (terms off)",
    )
    p_rapm.add_argument(
        "--defense-curvature",
        choices=["diminishing", "free", "none"], default="diminishing",
        help="Same for defensive terms (defense may have weakest-link "
        "synergy rather than diminishing returns, so free is most "
        "defensible here)",
    )
    p_rapm.add_argument(
        "--tuned", action="store_true",
        help="Fit with the configuration learned by `fablerapm tune` "
        "(overrides the model flags above)",
    )
    p_rapm.set_defaults(func=cmd_rapm)

    p_tune = sub.add_parser(
        "tune",
        help="Learn every tunable free parameter from held-out games and "
        "save the winning config (use via `rapm --tuned`)",
    )
    _add_common(p_tune)
    p_tune.add_argument("--lambda", dest="ridge_lambda", default="cv")
    p_tune.add_argument(
        "--seeds", type=int, default=3,
        help="Number of inner holdout splits to average (default 3; more = "
        "slower but less noise-chasing). Chrono mode uses this many "
        "forward-chaining folds",
    )
    p_tune.add_argument("--test-frac", type=float, default=0.2)
    p_tune.add_argument(
        "--passes", type=int, default=1,
        help="Coordinate-descent passes over the parameter grids (2 lets "
        "the model-family choice react to the re-tuned lambda)",
    )
    p_tune.add_argument(
        "--split", choices=["chrono", "random"], default="chrono",
        help="Holdout scheme: chrono (default) trains on earlier games and "
        "tests on later ones — the honest protocol for current ratings; "
        "random reproduces seeded game shuffles",
    )
    p_tune.add_argument(
        "--outer-frac", type=float, default=0.2,
        help="Fraction of games held out UNTOUCHED for the final unbiased "
        "score of the winning config (default 0.2; 0 disables nesting and "
        "reverts to selection-set reporting, which is optimistic)",
    )
    # tuning on all 30 seasons is slow and mixes eras; recent seasons are
    # the sensible zero-decision default (override with --seasons)
    p_tune.set_defaults(func=cmd_tune, seasons="recent-3")

    p_val = sub.add_parser(
        "validate",
        help="Cross-check stored stints against official game-log scores "
        "(offline, run after build)",
    )
    _add_common(p_val)
    p_val.set_defaults(func=cmd_validate)

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
    p_spm.add_argument(
        "--garbage-weight", type=float, default=1.0,
        help="Garbage-time weight for the RAPM target fits; match what you "
        "use in `rapm` so the prior predicts the same quantity",
    )
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
        "--last-season-scales", type=float, nargs="*", default=[],
        help="Also evaluate last-season priors at these scales "
        "(needs the previous season's stints built)",
    )
    p_eval.add_argument(
        "--spm", action="store_true",
        help="Also evaluate the trained SPM prior (needs spm-train first). "
        "IMPORTANT: train the SPM on seasons disjoint from the evaluated "
        "ones — its RAPM targets otherwise saw these games",
    )
    p_eval.add_argument(
        "--interactions", action="store_true",
        help="Also evaluate nonlinear two-phase RAPM (lineup-talent "
        "interaction terms)",
    )
    p_eval.add_argument(
        "--decays", type=float, nargs="*", default=[1.0],
        help="Season-decay values to grid over (default: 1.0)",
    )
    p_eval.add_argument(
        "--garbage-weights", type=float, nargs="*", default=[1.0],
        help="Garbage-time weights to grid over (default: 1.0)",
    )
    p_eval.add_argument(
        "--playoff-weights", type=float, nargs="*", default=[1.0],
        help="Playoff-stint weights to grid over (default: 1.0)",
    )
    p_eval.add_argument(
        "--holdout-no-garbage", action="store_true",
        help="Exclude garbage-time rows from the holdout metric "
        "(recommended when tuning --garbage-weights)",
    )
    p_eval.add_argument("--test-frac", type=float, default=0.2)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument(
        "--split", choices=["random", "chrono"], default="random",
        help="Holdout scheme: random seeded game shuffle (default), or "
        "chrono — hold out the latest games, train only on earlier ones "
        "(--seed is then ignored)",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_cal = sub.add_parser(
        "calibrate-curvature",
        help="Null calibration for interaction curvature: simulate additive "
        "(zero-curvature) copies of the data from the fitted linear model, "
        "re-run the free-signed interaction fit on each, and report which "
        "real gammas fall outside the estimator's own null band",
    )
    _add_common(p_cal)
    p_cal.add_argument("--lambda", dest="ridge_lambda", default="cv")
    p_cal.add_argument(
        "--sims", type=int, default=100,
        help="Null simulations (default 100). Each one re-runs the full "
        "interaction fit, so expect several seconds per sim on a full "
        "season",
    )
    p_cal.add_argument("--seed", type=int, default=0)
    p_cal.add_argument(
        "--null", choices=["poisson", "resample"], default="poisson",
        help="Null outcome generator: poisson draws points around the "
        "linear model's rates (slightly conservative dispersion); resample "
        "redraws studentized residuals with replacement, matching the real "
        "data's dispersion exactly",
    )
    p_cal.add_argument(
        "--alpha", type=float, default=0.05,
        help="Two-sided size of the null band (default 0.05 -> the central "
        "95%% of null gammas)",
    )
    p_cal.add_argument(
        "--offense-curvature", choices=["diminishing", "free"], default="free",
        help="Curvature mode being calibrated (default free — the whole "
        "point is to test the unconstrained estimator)",
    )
    p_cal.add_argument(
        "--defense-curvature", choices=["diminishing", "free"], default="free",
    )
    p_cal.add_argument("--decay", type=float, default=1.0)
    p_cal.add_argument("--playoff-weight", type=float, default=1.0)
    p_cal.add_argument("--garbage-weight", type=float, default=1.0)
    p_cal.set_defaults(func=cmd_calibrate_curvature)

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
