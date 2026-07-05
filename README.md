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
- All possessions are included — no garbage-time, heave, or leverage filters.

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

```bash
# 1. scrape features (8 requests/season: 2 box + 6 tracking)
fablerapm features --seasons 2015-16:2024-25

# 2. train the SPM (ridge: features -> per-season RAPM, minutes-weighted CV)
fablerapm spm-train --seasons 2015-16:2023-24 --season-types regular

# 3. use it as the RAPM prior
fablerapm rapm --seasons 2024-25 --prior spm

# compare variants on held-out games (possession-weighted MSE; priors are
# computed from training games only)
fablerapm evaluate --seasons 2022-23:2024-25 --two-phase-scales 0.5 1.0 --spm
```

`fablerapm evaluate` splits by game (default 20% holdout), fits each variant
on the train side, and reports holdout MSE against an intercept-only
baseline — the harness for testing any future calculation tweak.

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
