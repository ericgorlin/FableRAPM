# FableRAPM

NBA RAPM (Regularized Adjusted Plus-Minus) pipeline with **offensive and
defensive splits per 100 possessions**, for every season with play-by-play
data (1996-97 to present), regular season **and** playoffs.

No box-score approximations anywhere in the data path: play-by-play-derived
lineup attribution with audited score reconciliation — possessions, lineups,
and points are parsed from raw stats.nba.com play-by-play (period
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
  Late-game (Q4/OT) stint rows also store the possession's start margin and
  seconds remaining, so even the garbage *definition* is a fit-time choice
  (`--garbage-rule`, searched by `tune`).
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
  prior. Self-contained — needs only stint data. Precisely speaking this is
  *iterated ridge* (relaxed shrinkage / decompression), not an independent
  prior: the shrinkage target is estimated from the same data, and along
  each ridge eigendirection the two-phase estimate has total shrinkage
  `s(2-s)` where plain ridge has `s` — always milder. Whether that milder,
  spectrum-reshaped shrinkage actually predicts better than plain ridge at
  *its own* best lambda is an empirical question; `tune` includes a lambda
  grid as a search coordinate precisely so a two-phase winner has to beat
  a properly retuned plain ridge, not just the baseline lambda.
- **SPM prior** (`--prior spm`): a statistical plus-minus built from
  per-player **box score + player-tracking** features (drives, touches,
  passing, contested shots, rebounding, speed/distance — tracking exists
  2013-14+; earlier seasons fall back to box/advanced only). The saved
  model artifact records its training seasons, and `--prior spm` warns
  automatically if you apply it to a season it was trained on (its RAPM
  targets saw those games).
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
  downweights. The default garbage definition is the parse-time flag:
  Q4/OT possession starting with margin >= 25 (any time), 18 (last 6 min),
  or 12 (last 3 min) (`stints.GARBAGE_TIERS`).
- `--garbage-rule BASE,PER_MINUTE`: replace the parse-time flag with a
  fit-time definition — a Q4/OT possession is garbage when `|margin| >=
  BASE + PER_MINUTE x minutes left in the period`. Works because every
  late-game stint row stores the possession's start margin and seconds
  remaining (`margin`/`secs_left` columns), so the garbage *rule* is now a
  tunable parameter (tune searches a small grid of them), not a data
  definition. Stints parsed before this schema lack the columns: delete
  `data/stints/`, re-run `build` (offline from the raw cache; use
  `--workers N` to parallelize the re-parse).

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
of guessing it. `--split chrono` holds out the *latest* games and trains
only on earlier ones (forward-chaining) — the honest protocol when the
model's job is current ratings or upcoming games, and what `tune` does by
default.

After building a season, run `fablerapm validate`: it cross-checks stint
totals against the official game-log scores (an independent source),
coverage against the schedule, and league points/100 plausibility — all
offline. Scope of that claim: reconciliation proves the *bookkeeping* is
right (every point in every game is attributed to some lineup, summing
exactly to official totals); it cannot by itself prove every point went to
the *correct* five-man lineup — a substitution-timing bug could in
principle misattribute events while preserving game totals. Lineup-level
attribution rests on pbpstats' event parsing plus this repo's fixture-game
test; a set of manually-verified golden games covering nasty edge cases is
the TODO that would close the gap.

## The full model, in three steps

`fablerapm rapm --interactions` implements:

1. **Base + prior-informed RAPM**: plain linear RAPM, then a second linear
   fit shrinking toward the first (decompresses stars that flat shrinkage
   compresses). With `--prior spm|last-season`, that prior anchors instead.
2. **Learn interaction terms**: sign-constrained least squares of the
   anchored fit's residuals on the talent-concentration basis (per side).
3. **Second-phase RAPM with 1+2 as input**: player coefficients re-fit
   against the interaction-adjusted target, shrinking toward step 1.

Steps 2–3 alternate to convergence. Every knob is a flag; `evaluate` is the
referee for all of them.

## Zero-decisions quickstart

If you don't want to choose anything yourself, run exactly this (each step
is resumable; interrupt freely):

```bash
pip install -e ".[dev]"
pytest                                   # offline checks
fablerapm smoke-test                     # ~8 live API requests
fablerapm build --seasons recent-3       # scrape 3 most recent seasons
fablerapm validate --seasons recent-3    # verify vs official scores
fablerapm tune                           # learn all tunable params (defaults to recent-3)
fablerapm rapm --seasons recent-3 --tuned
```

Then extend backwards at your leisure: `fablerapm build` (full 1996-97 ->
present, hours, resumable) followed by `validate` and
`fablerapm rapm --tuned` for per-season history. `--seasons recent-N` and
`all` work everywhere.

## Suggested experiments (in order)

```bash
pip install -e ".[dev]" && pytest        # 58 offline tests, no network
fablerapm smoke-test                     # ~8 live API requests, end-to-end
```

**1. First real data.** Build a recent season, verify it, look at it:

```bash
fablerapm build --seasons 2023-24            # ~1300 games, ~25 min
fablerapm validate --seasons 2023-24         # scores vs official game logs
fablerapm rapm --seasons 2023-24 --min-poss 2000
```

Sanity check: the top of the regular-season CSV should be recognizable
MVP-tier names, league avg ~115 pts/100 in `meta.json`, CV lambda usually
in the 1000s for a single season.

**2. Tune the free parameters on held-out games** (expect small gaps —
stint outcomes are noisy; re-run with a few `--seed`s before believing a
winner):

```bash
fablerapm build --seasons 2021-22:2023-24
fablerapm evaluate --seasons 2021-22:2023-24 \
    --decays 1.0 0.9 0.75 --garbage-weights 1.0 0.5 0.0 --holdout-no-garbage
```

**3. Priors.** Compare stabilizers on a single noisy season:

```bash
fablerapm evaluate --seasons 2023-24 --two-phase-scales 0.5 1.0 \
    --last-season-scales 0.5 0.7          # needs 2022-23 built
fablerapm features --seasons 2015-16:2023-24     # box + tracking, 8 req/season
fablerapm spm-train --seasons 2015-16:2022-23 --season-types regular
fablerapm evaluate --seasons 2023-24 --spm
fablerapm rapm --seasons 2023-24 --prior spm     # writes *_spm.csv
```

**4. The LeBron/KG experiment** (diminishing returns on stacked stars).
Build 2003-04 through 2013-14, then compare pooled fits:

```bash
fablerapm build --seasons 2003-04:2013-14        # long build, resumable
fablerapm rapm --seasons 2003-04:2013-14 --pool --combine-types --decay 0.9
fablerapm rapm --seasons 2003-04:2013-14 --pool --combine-types --decay 0.9 \
    --interactions
```

Look at: (a) LeBron vs KG ordering in the two CSVs, (b) LeBron's Miami-era
single seasons (2010-11:2013-14) with and without `--interactions`,
(c) `gamma_off` in the meta — nonzero values mean the data shows concave
offensive production (per 1 SD of each concentration feature, per 100).
Then let holdout arbitrate (`evaluate --seasons ... --interactions`), and
before believing any curvature is real, run the null calibration
(`calibrate-curvature`, experiment 5): a nonzero gamma only counts if it
falls outside what this estimator produces on additive data.

**5. Curvature signs.** The defaults constrain to diminishing returns
because the unconstrained estimator is biased toward fake convexity by
ridge shrinkage (demonstrated in `tests/test_interactions.py`) and the
collinear basis makes free-signed fits unstable. To learn signs from data
anyway — most defensible on defense, where weakest-link synergy is
plausible:

```bash
fablerapm rapm --seasons ... --interactions --defense-curvature free
fablerapm rapm --seasons ... --interactions --offense-curvature free  # skeptically
```

Believe a free-signed gamma only if it replicates across seasons and wins
in `evaluate`. The principled arbiter is implemented: **null calibration**
(`fablerapm calibrate-curvature`) simulates additive (zero-curvature)
copies of your data from the fitted linear model, re-runs the entire
free-signed interaction pipeline on each, and reports which real-data
curvature values fall outside the estimator's own null band:

```bash
fablerapm calibrate-curvature --seasons 2013-14 --sims 200
```

The headline line is the per-side *curvature score* (fitted interaction
contribution, in points/100, at top-decile-concentration lineups) — the
individual gammas sit on a collinear basis and trade off against each
other, so read them only for color. A score outside the null band means
the additive model is genuinely missing something; treat the side it lands
on with care (credit assignment between offense and defense blocks is the
least identified part — see `calibrate.py`), and still demand replication
across seasons before calling it basketball.

**5b. Or learn everything at once.** `tune` runs greedy coordinate descent
over all tunable parameters (garbage weight, decay, playoff weight, prior
kind + scale, interactions + defense curvature, and ridge lambda over a
multiplier grid around the CV pick), scored on held-out games averaged
over several inner splits, and saves the winner:

```bash
fablerapm tune --seasons 2022-23:2024-25 --seeds 3
fablerapm rapm --seasons 2022-23:2024-25 --pool --tuned
```

The evaluation is **nested and chronological by default**: an outer block
of the latest games (`--outer-frac`, default 20%) is set aside before any
tuning; the search selects on forward-chaining inner splits of the
remaining games (train strictly earlier than test; `--split random`
restores seeded shuffles); and at the end the winner and the untouched
default config are each scored once on the outer block. `outer_mse` vs
`outer_default_mse` in the output is the unbiased report of what tuning
bought — the inner MSE chose the winner, so it is optimistic as an
estimate and only valid for ranking. Lambda is searched *last* in each
pass so it re-tunes for whichever model family won (a two-phase or
interaction winner has to beat plain ridge at its own best lambda);
`--passes 2` lets the family choice react to the re-tuned lambda in turn.

The full search history is in `data/results/tuned_config.json`. Caveats:
holdout gaps between reasonable settings are small, so prefer more
`--seeds` over more `--passes`; and parameters whose grid doesn't apply
(decay on one season, playoff weight on one type, last-season prior
without the previous season built) are skipped automatically. What tune
does NOT search: possession attribution (parse-time), the interaction
feature basis, and the offensive sign constraint — those are structural
(see TODO below). The garbage-time *rule* IS searched (a grid of fit-time
`BASE + PER_MINUTE x minutes-left` thresholds alongside the parse-time
tiers) whenever the stint schema carries the late-game context columns.

**6. Full history**, once happy with settings:

```bash
fablerapm build                    # 1996-97 -> present, many hours, resumable
fablerapm validate
fablerapm rapm                     # per-season, both season types
```

## TODOs (in rough priority order)

The goal state is that *no* configuration is a human decision. What still
stands between here and there:

1. **Golden games for lineup attribution**: score reconciliation proves
   totals, not attribution (see the validate section). Hand-verify a small
   set of games covering the nasty edge cases — simultaneous substitutions,
   technical/flagrant free throws, substitutions between free throws,
   offensive rebounds and possession continuation, end-of-quarter events,
   overturned calls — and pin their exact stint rows as fixtures
   (`tests/test_fixture_game.py` shows the harness: fabricate or cache the
   raw pbp, assert exact rows through the real pbpstats path).
2. **Own box-score/tracking aggregation from play-by-play**: SPM features
   currently come from official full-season aggregates, so they include
   garbage time and can't be decayed or filtered consistently with the
   stint weights. pbpstats emits per-event stats with the same lineup
   attribution we already use; aggregating them ourselves makes features
   consistent with every weighting knob and extends "tracking-adjacent"
   features to all seasons.
3. **Richer interaction basis**: splines/kernels over lineup talent
   composition instead of three hand-picked convex shapes; the basis is a
   one-line list (`model.INTERACTION_FEATURES`), and tune/evaluate already
   referee additions. Position/role-aware concentration (creator vs big)
   once SPM features exist per player.
4. **Luck adjustment** as an evaluate-able variant: replace realized 3P%
   / opponent FT% with expected values at parse time (schema addition),
   then let tune decide if it helps.
5. **Aging curve in the last-season prior**: scale by a learned age curve
   rather than one global `prior_scale`.
6. **Smooth garbage downweighting**: the fit-time garbage rule is a hard
   threshold family; with margin/secs_left stored per row, a continuous
   weight surface (e.g. logistic in margin with a time-varying midpoint)
   is one function away, and tune can referee it against the thresholds.
7. **Game dates in the stint schema**: chronological splits currently
   order games by (season, type, game id) — schedule order — so
   postponed/rescheduled games (COVID-era makeups) can land slightly out
   of true date order. The cached league game log already has real dates;
   storing a `game_date` per stint row at build time would make chrono
   ordering exact and delete the heuristic.

Note on comparing numbers across the margin/secs schema change: late-game
rows now aggregate per possession context, so holdout/CV MSE *levels* are
higher than pre-schema runs purely from finer row granularity (the extra
within-stint variance is variant-independent — rankings and vs-baseline
gaps are still comparable, absolute MSEs are not).

Done and moved out of this list: null calibration for curvature signs
(`fablerapm calibrate-curvature`, experiment 5) with Poisson or
residual-resampling nulls (`--null resample`); nested + chronological
holdout evaluation for `tune` (outer untouched test block,
forward-chaining inner splits, lambda in the search grid — experiment 5b);
the garbage-time rule as a fit-time, tune-searched parameter
(`--garbage-rule`, margin/secs stored per late-game stint row); and
parallel cache re-parses (`build --workers N`).

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
