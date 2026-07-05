# FableRAPM

NBA RAPM (Regularized Adjusted Plus-Minus) pipeline with **offensive and
defensive splits per 100 possessions**, for every season with play-by-play
data (1996-97 to present), regular season **and** playoffs.

No box-score approximations anywhere in the data path: possessions, lineups,
and points are parsed exactly from raw stats.nba.com play-by-play (period
starters, substitutions during free throws, and possession-counting edge
cases are resolved by [pbpstats](https://github.com/dblackrun/pbpstats),
which was built for exactly this). Raw API responses are cached on disk, so
the dataset is fully regenerable offline and cheap to keep up to date.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
# Tiny live end-to-end test (~8 API requests: 3 regular season + 2 playoff
# games of 2023-24, full pipeline through a RAPM fit with sanity checks)
fablerapm smoke-test

# Offline tests (no network): possession aggregation, model recovery on
# synthetic data with known player effects, CLI end-to-end
pytest

# Scrape & store everything (1996-97 -> current, regular season + playoffs).
# ~1 request/sec; a full history build takes many hours but is resumable —
# interrupt freely and re-run, it only fetches what's missing.
fablerapm build

# Or start smaller:
fablerapm build --seasons 2022-23:2024-25

# Fit RAPM: one CSV per (season, season type) by default
fablerapm rapm --seasons 2022-23:2024-25

# 3-season pooled run, regular season + playoffs together:
fablerapm rapm --seasons 2022-23:2024-25 --pool --combine-types
```

Staying up to date during a season is just re-running the same two commands
(cron-friendly): `build` refetches the current season's schedule, scrapes
only new final games, and appends; `rapm` refits from the stored stints.

## Outputs

`data/results/rapm_<seasons>_<types>.csv`:

| column | meaning |
|---|---|
| `orapm` | points added per 100 offensive possessions (positive = good) |
| `drapm` | points prevented per 100 defensive possessions (positive = good) |
| `rapm`  | `orapm + drapm` |
| `off_poss` / `def_poss` | exact possession counts on offense / defense |

Every CSV gets a `.meta.json` sidecar with the ridge strength, the
cross-validation table, league-average points/100, game/row/player counts,
and any accounting notes.

## How it works

1. **Scrape** (`fablerapm/seasons.py`, `stints.py`): season game logs and
   per-game play-by-play from stats.nba.com, throttled (~1 req/s with
   backoff/retry) and cached under `data/raw/` (pbpstats' file format).
2. **Stints** (`stints.py`, `build.py`): each game is reduced to rows of
   *(offense lineup, defense lineup, possessions, points)* using pbpstats'
   exact per-lineup attribution (`OffPoss` / `OpponentPoints`). Stored per
   season & season type in `data/stints/*.parquet` with a manifest of
   processed/failed games. Stint totals are reconciled against the final
   score of every game; mismatches are recorded as warnings in the manifest.
3. **Model** (`model.py`): weighted ridge regression on stint rows —
   `y` = offense points per 100, one indicator column per player per side,
   weight = possessions, unpenalized intercept. Ridge strength is chosen by
   **game-grouped K-fold CV** (`--lambda cv`, the default) or fixed
   (`--lambda 2500`).

### Conventions (explicit, not approximations)

- A possession belongs to the lineup pbpstats attributes its efficiency
  event to (free-throw possessions belong to the lineup at the time of the
  foul). Points belong to the lineup on the floor when they were scored.
- Points scored by the *defending* team during a possession (technical free
  throws) count toward that team's offense for the same lineup matchup. In
  the rare case no matching offensive possession exists, the orphaned points
  are dropped and counted in `meta.dropped_orphan_points` (a handful of
  points per season, at most).
- All possessions are stored; garbage time is *flagged*, and filtering or
  downweighting is a fit-time choice (`--garbage-weight`), not a data choice.
- SPM **features** are official full-season aggregates from stats.nba.com,
  so they always include garbage time and can't be decayed per possession;
  the weighting knobs apply to the stint regressions (including the SPM's
  RAPM targets via `spm-train --garbage-weight`). Building garbage-aware
  box features from our own play-by-play is possible future work.

### Failed games

A small number of games (mostly late-90s) have broken play-by-play (e.g.
undeterminable period starters). They are skipped and recorded in
`data/stints/*.manifest.json` with the error. Many can be fixed with
pbpstats override files in `data/raw/overrides/` (see the pbpstats docs),
then reprocessed with `fablerapm build --retry-failed`.

## Model variants & evaluation

`fit_rapm` accepts a `prior` mapping `player_id -> (orapm_prior,
drapm_prior)`; coefficients shrink toward the prior instead of zero. Two
priors are built in:

- **Two-phase RAPM** (`--prior two-phase`): phase one is a plain RAPM; phase
  two re-fits shrinking toward it, letting star coefficients escape flat
  shrinkage (diminishing-returns compression). `--prior-scale` scales the
  prior. Self-contained — needs only stint data.
- **SPM prior** (`--prior spm`): a statistical plus-minus built from
  per-player **box score + player-tracking** features (drives, touches,
  passing, contested shots, rebounding, speed/distance — tracking exists
  2013-14+; earlier seasons fall back to box/advanced only).
- **Last-season prior** (`--prior last-season --prior-scale 0.7`): the
  previous season's RAPM, scaled, as the shrinkage target.
- **Nonlinear two-phase** (`--interactions`): linear two-phase plus a
  *basis* of convex talent-concentration regressors per side (top-2
  product, all-pairs sum, squared total — `model.INTERACTION_FEATURES`),
  fit jointly with player coefficients anchored to the decompressed
  estimates and coefficients sign-constrained (NNLS) to the
  diminishing-returns hypothesis. Concavity is the only baked-in
  assumption; *which* shapes of talent saturate, and how strongly on
  offense vs defense, is learned from data (`gamma_off`/`gamma_def` in
  the meta). When lineups stacking creators produce less than the sum of
  their parts (the LeBron-next-to-Wade effect), the penalty flows to the
  concentration terms instead of the stars; with no concavity in the
  data, gammas land at zero and the fit reduces to linear two-phase. See
  `fit_interaction_rapm`'s docstring for why each piece is load-bearing;
  in controlled synthetic tests the detector is conservative (shrinkage
  bias competes with the signal), so treat nonzero gammas on real data
  as the finding, and arbitrate with `evaluate --interactions`.

Weighting knobs (all recorded in the output meta):

- `--decay 0.75`: in pooled multi-season fits, a stint's weight is
  multiplied by `decay ** years_before_most_recent_season`.
- `--playoff-weight` (default **1.5**): upweights playoff/play-in stints
  when types are pooled with `--combine-types` (no effect on single-type
  fits, where it would only distort the effective lambda).
- `--garbage-weight`: 1.0 keeps garbage time (default), 0 drops it, between
  downweights. Garbage time = Q4/OT possession starting with margin >= 25
  (any time), 18 (last 6 min), or 12 (last 3 min) — flagged per stint at
  parse time (`stints.GARBAGE_TIERS`), so changing the *rule* requires a
  re-parse (delete `data/stints/`, re-run `build`; offline from the raw
  cache), while changing the *weight* is instant.

```bash
# 1. scrape features (8 requests/season: 2 box + 6 tracking)
fablerapm features --seasons 2015-16:2024-25

# 2. train the SPM (ridge: features -> per-season RAPM, minutes-weighted CV)
fablerapm spm-train --seasons 2015-16:2023-24 --season-types regular

# 3. use it as the RAPM prior
fablerapm rapm --seasons 2024-25 --prior spm

# compare variants and tune weights on held-out games (possession-weighted
# MSE; priors are computed from training games only)
fablerapm evaluate --seasons 2022-23:2024-25 --two-phase-scales 0.5 1.0 --spm
fablerapm evaluate --seasons 2022-23:2024-25 \
    --decays 1.0 0.9 0.75 --garbage-weights 1.0 0.5 0.0 --holdout-no-garbage
```

`fablerapm evaluate` splits by game (default 20% holdout), fits each variant
on the train side, and reports holdout MSE against an intercept-only
baseline, sorted best-first over the full grid of decay/garbage/playoff
weights — the harness for learning every free parameter from data instead
of guessing it. After building a season, run `fablerapm validate`: it
cross-checks stint totals against the official game-log scores (an
independent source), coverage against the schedule, and league points/100
plausibility — all offline.

## Data layout

```
data/
  raw/          # cached API responses (pbpstats format): pbp/, schedule/, overrides/
  stints/       # per season+type: stint parquet + processed/failed manifest
  results/      # rapm_*.csv + rapm_*.meta.json
  smoke/        # isolated data dir used by `fablerapm smoke-test`
```

`FABLERAPM_DATA_DIR` or `--data-dir` overrides the location. Deleting
`data/stints` and re-running `build` re-parses from the raw cache with no
network; deleting `data/raw` re-scrapes from the API.
