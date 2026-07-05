"""RAPM: weighted ridge regression on possession-level stint data.

Each stint row contributes one weighted observation:
    y = points per 100 possessions scored by the offense lineup
    x = indicators for the 5 offensive players (offense block) and the
        5 defensive players (defense block)
    weight = number of possessions

The intercept (league-average offense) is unpenalized; player coefficients
are shrunk toward zero (or toward an optional prior — the hook for adding a
box-score prior or a second phase later). Ridge strength is chosen by
game-grouped K-fold cross-validation by default.

Sign conventions in the output:
    ORAPM: points added per 100 offensive possessions (positive = good)
    DRAPM: points prevented per 100 defensive possessions (positive = good)
    RAPM = ORAPM + DRAPM
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .config import results_dir, season_end_year, season_type_slug, stints_path

logger = logging.getLogger(__name__)

DEFAULT_LAMBDA_GRID = [25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]


def load_stints(
    data_dir: Path, seasons: list[str], season_types: list[str]
) -> pd.DataFrame:
    """Load stored stint rows for the given seasons/types into one frame."""
    frames = []
    missing = []
    for season in seasons:
        for season_type in season_types:
            path = stints_path(data_dir, season, season_type)
            if not path.exists():
                missing.append(f"{season} {season_type} ({path})")
                continue
            df = pd.read_parquet(path)
            df["season"] = season
            df["season_type"] = season_type
            frames.append(df)
    if missing:
        raise FileNotFoundError(
            "No stint data for: "
            + "; ".join(missing)
            + ". Run `fablerapm build` for these seasons first."
        )
    return pd.concat(frames, ignore_index=True)


@dataclass
class Design:
    X: sparse.csr_matrix
    y: np.ndarray
    weights: np.ndarray  # regression weights: possessions x optional multiplier
    groups: np.ndarray  # game_id per row, for grouped CV
    player_ids: list[int]  # column j = offense, column n_players + j = defense
    off_poss: np.ndarray  # per player, raw (unweighted) possession counts
    def_poss: np.ndarray
    dropped_points: int  # points in rows with 0 possessions (orphaned technical FTs)
    n_interactions: int = 0  # trailing dense columns (lineup talent terms)
    interaction_info: dict | None = None  # standardization params for reuse


def _lineup_top2_product(lineups: np.ndarray, talent: dict, idx: int) -> np.ndarray:
    """Product of the two largest positive talents in each lineup.

    A talent-*concentration* feature: it is large only when two creators
    share the floor, and stays small for one star plus role players — which
    is what distinguishes redundancy (diminishing returns) from mere total
    talent. Symmetric sums or squared totals fail here: they are also high
    for a lone star's lineups, whose residuals have the opposite sign.
    """
    out = np.empty(len(lineups))
    for i, lineup in enumerate(lineups):
        t = sorted(
            (max(talent.get(int(p), (0.0, 0.0))[idx], 0.0) for p in lineup.split("-")),
            reverse=True,
        )
        out[i] = t[0] * t[1]
    return out


def _standardize(F: np.ndarray, w: np.ndarray, params: dict | None = None):
    if params is None:
        m = float(np.average(F, weights=w))
        s = float(np.sqrt(np.average((F - m) ** 2, weights=w))) or 1.0
        params = {"m": m, "s": s}
    return (F - params["m"]) / params["s"], params


def _parse_lineup(lineup: str) -> list[int]:
    return [int(p) for p in lineup.split("-")]


def build_design(
    stints: pd.DataFrame, interaction_talent: dict | None = None
) -> Design:
    zero_poss = stints["poss"] == 0
    dropped_points = int(stints.loc[zero_poss, "points"].sum())
    if dropped_points:
        logger.info(
            "Dropping %d rows with 0 possessions (%d orphaned points, "
            "e.g. technical FTs with no matching lineup possession)",
            int(zero_poss.sum()), dropped_points,
        )
    stints = stints.loc[~zero_poss].reset_index(drop=True)

    player_ids = sorted(
        {p for lineup in stints["off_lineup"] for p in _parse_lineup(lineup)}
        | {p for lineup in stints["def_lineup"] for p in _parse_lineup(lineup)}
    )
    col = {pid: j for j, pid in enumerate(player_ids)}
    n_players = len(player_ids)
    n_rows = len(stints)

    data = np.ones(n_rows * 10, dtype=np.float64)
    indices = np.empty(n_rows * 10, dtype=np.int64)
    indptr = np.arange(0, n_rows * 10 + 1, 10, dtype=np.int64)
    off_lineups = stints["off_lineup"].to_numpy()
    def_lineups = stints["def_lineup"].to_numpy()
    for i in range(n_rows):
        offs = [col[p] for p in _parse_lineup(off_lineups[i])]
        defs = [n_players + col[p] for p in _parse_lineup(def_lineups[i])]
        indices[i * 10 : i * 10 + 10] = offs + defs
    X = sparse.csr_matrix(
        (data, indices, indptr), shape=(n_rows, 2 * n_players)
    )

    poss = stints["poss"].to_numpy(dtype=np.float64)
    points = stints["points"].to_numpy(dtype=np.float64)
    y = 100.0 * points / poss
    # regression weights may carry decay / season-type multipliers; player
    # possession counts stay raw so the output columns remain exact tallies
    if "weight_mult" in stints:
        weights = poss * stints["weight_mult"].to_numpy(dtype=np.float64)
    else:
        weights = poss

    poss_by_col = np.asarray(X.multiply(poss[:, None]).sum(axis=0)).ravel()

    n_interactions = 0
    interaction_info = None
    if interaction_talent is not None:
        q_off, p_off = _standardize(
            _lineup_top2_product(off_lineups, interaction_talent, 0), weights
        )
        q_def, p_def = _standardize(
            _lineup_top2_product(def_lineups, interaction_talent, 1), weights
        )
        X = sparse.hstack(
            [X, sparse.csr_matrix(np.column_stack([q_off, q_def]))], format="csr"
        )
        n_interactions = 2
        interaction_info = {"off": p_off, "def": p_def}

    return Design(
        X=X,
        y=y,
        weights=weights,
        groups=stints["game_id"].to_numpy(),
        player_ids=player_ids,
        off_poss=poss_by_col[:n_players],
        def_poss=poss_by_col[n_players:],
        dropped_points=dropped_points,
        n_interactions=n_interactions,
        interaction_info=interaction_info,
    )


def _apply_weight_multipliers(
    stints: pd.DataFrame,
    decay: float = 1.0,
    playoff_weight: float = 1.0,
    garbage_weight: float = 1.0,
) -> pd.DataFrame:
    """Attach a weight_mult column and drop zero-weight rows."""
    if decay == 1.0 and playoff_weight == 1.0 and garbage_weight == 1.0:
        return stints
    mult = np.ones(len(stints))
    if decay != 1.0:
        if "season" not in stints:
            raise ValueError("decay requires a 'season' column in stints")
        end_years = stints["season"].map(season_end_year)
        mult *= float(decay) ** (end_years.max() - end_years).to_numpy(dtype=float)
    if playoff_weight != 1.0:
        if "season_type" not in stints:
            raise ValueError(
                "playoff_weight requires a 'season_type' column in stints"
            )
        # only meaningful when types are mixed; in a single-type fit a
        # uniform multiplier would just distort the effective lambda
        if stints["season_type"].nunique() > 1:
            mult *= np.where(
                stints["season_type"] == "Regular Season",
                1.0,
                float(playoff_weight),
            )
    if garbage_weight != 1.0:
        if "garbage" not in stints:
            raise ValueError(
                "garbage_weight requires a 'garbage' column: re-parse "
                "stints (delete data/stints, re-run `fablerapm build` — "
                "the raw cache makes this offline and fast)"
            )
        mult *= np.where(
            stints["garbage"].to_numpy(dtype=bool), float(garbage_weight), 1.0
        )
    stints = stints.assign(weight_mult=mult)
    keep = stints["weight_mult"] > 0
    if not keep.all():
        stints = stints.loc[keep].reset_index(drop=True)
    return stints


def _ridge(lam: float) -> Ridge:
    return Ridge(alpha=lam, fit_intercept=True, solver="sparse_cg", tol=1e-8)


def _weighted_mse(y_true, y_pred, w) -> float:
    return float(np.average((y_true - y_pred) ** 2, weights=w))


def cross_validate_lambda(
    design: Design,
    lambdas=DEFAULT_LAMBDA_GRID,
    n_folds: int = 5,
) -> tuple[float, list[dict]]:
    """Pick ridge strength by game-grouped K-fold CV (weighted MSE)."""
    n_groups = len(np.unique(design.groups))
    n_folds = min(n_folds, n_groups)
    if n_folds < 2:
        raise ValueError("Need at least 2 games for cross-validation")
    folds = list(
        GroupKFold(n_splits=n_folds).split(design.X, design.y, design.groups)
    )
    table = []
    for lam in lambdas:
        fold_mse = []
        for train_idx, val_idx in folds:
            model = _ridge(lam)
            model.fit(
                design.X[train_idx],
                design.y[train_idx],
                sample_weight=design.weights[train_idx],
            )
            pred = model.predict(design.X[val_idx])
            fold_mse.append(
                _weighted_mse(design.y[val_idx], pred, design.weights[val_idx])
            )
        table.append({"lambda": lam, "cv_mse": float(np.mean(fold_mse))})
        logger.info("lambda=%g: CV weighted MSE=%.4f", lam, table[-1]["cv_mse"])
    best = min(table, key=lambda r: r["cv_mse"])
    return best["lambda"], table


@dataclass
class RapmResult:
    players: pd.DataFrame
    meta: dict = field(default_factory=dict)


def fit_rapm(
    stints: pd.DataFrame,
    lam: float | str = "cv",
    lambdas=DEFAULT_LAMBDA_GRID,
    n_folds: int = 5,
    prior: dict[int, tuple[float, float]] | None = None,
    decay: float = 1.0,
    playoff_weight: float = 1.0,
    garbage_weight: float = 1.0,
) -> RapmResult:
    """Fit RAPM on stint rows.

    ``prior`` optionally maps player_id -> (orapm_prior, drapm_prior) in
    per-100 units (DRAPM prior in positive-is-good convention). Coefficients
    are shrunk toward the prior instead of zero; players missing from the
    prior shrink toward 0.

    ``decay`` down-weights older seasons in pooled multi-season fits: a
    stint's weight is multiplied by decay ** (years before the most recent
    season in the data). 1.0 (default) weights all seasons equally.

    ``playoff_weight`` multiplies the weight of non-regular-season stints
    when season types are pooled (1.0 = no adjustment).

    ``garbage_weight`` multiplies the weight of garbage-time stints
    (see stints.GARBAGE_TIERS): 1.0 keeps them, 0.0 drops them entirely,
    values in between downweight.
    """
    stints = _apply_weight_multipliers(stints, decay, playoff_weight, garbage_weight)
    design = build_design(stints)
    n_players = len(design.player_ids)

    y = design.y
    prior_vec = np.zeros(2 * n_players)
    if prior:
        for j, pid in enumerate(design.player_ids):
            if pid in prior:
                o, d = prior[pid]
                prior_vec[j] = o
                # defense columns measure impact on opponent scoring
                # (positive = bad), so flip the positive-is-good prior
                prior_vec[n_players + j] = -d
        y = y - design.X @ prior_vec

    cv_table = None
    if lam == "cv":
        lam, cv_table = cross_validate_lambda(design, lambdas, n_folds)

    model = _ridge(float(lam))
    model.fit(design.X, y, sample_weight=design.weights)
    coef = model.coef_ + prior_vec

    orapm = coef[:n_players]
    drapm = -coef[n_players : 2 * n_players]  # flip so positive = good defense
    players = pd.DataFrame(
        {
            "player_id": design.player_ids,
            "orapm": orapm,
            "drapm": drapm,
            "rapm": orapm + drapm,
            "off_poss": design.off_poss,
            "def_poss": design.def_poss,
        }
    ).sort_values("rapm", ascending=False, ignore_index=True)

    total_poss = float(design.weights.sum())
    meta = {
        "lambda": float(lam),
        "cv": cv_table,
        "n_stint_rows": int(design.X.shape[0]),
        "n_players": n_players,
        "n_games": int(len(np.unique(design.groups))),
        "total_possessions": total_poss,
        "avg_points_per_100": float(np.average(design.y, weights=design.weights)),
        "intercept": float(model.intercept_),
        "dropped_orphan_points": design.dropped_points,
        "prior": "custom" if prior else None,
        "decay": decay,
        "playoff_weight": playoff_weight,
        "garbage_weight": garbage_weight,
    }
    return RapmResult(players=players, meta=meta)


def fit_interaction_rapm(
    stints: pd.DataFrame,
    lam: float | str = "cv",
    prior: dict[int, tuple[float, float]] | None = None,
    decay: float = 1.0,
    playoff_weight: float = 1.0,
    garbage_weight: float = 1.0,
) -> RapmResult:
    """Nonlinear two-phase RAPM (diminishing returns on stacked lineups).

    Phase one is a plain linear RAPM. Its residuals are then regressed on
    two curvature features — positive-part-squared standardized sums of
    phase-one lineup talent (offense and defense) — and finally the player
    coefficients are re-fit against the curvature-adjusted target.

    Methodological details, all load-bearing (each earlier variant failed a
    controlled synthetic test):

    - The concentration feature is the standardized product of the two
      largest positive talents in the lineup — high only when two creators
      share the floor. Squared/summed total-talent features are also high
      for a lone star's lineups, whose residuals have the opposite sign,
      cancelling the signal.
    - Player coefficients are *anchored* (shrunk toward) the linear
      two-phase estimates while gamma is fit jointly by alternation. The
      anchor breaks the credit-assignment tie: without it, ridge lets one
      dense feature act as a penalty-cheap substitute for many shrunk
      player coefficients and gamma comes out positive (a pure shrinkage
      artifact); with it, the stacked-row shortfall must flow to gamma.
    - Signs are clamped to the diminishing-returns hypothesis (offense
      gamma <= 0, defense gamma >= 0 in points-allowed terms): if the data
      shows no concavity, gamma clamps to zero and the result degrades
      gracefully to the linear two-phase fit — the correction can give
      stars back their stacking penalty, never take credit from them.
    """
    stints = _apply_weight_multipliers(stints, decay, playoff_weight, garbage_weight)
    phase1 = fit_rapm(stints, lam=lam)
    lam2 = phase1.meta["lambda"]
    phase1_prior = {
        int(r.player_id): (float(r.orapm), float(r.drapm))
        for r in phase1.players.itertuples()
    }
    # linear two-phase: shrink toward phase one to undo star compression
    phase2 = fit_rapm(stints, lam=lam2, prior=phase1_prior)
    talent = {
        int(r.player_id): (float(r.orapm), float(r.drapm))
        for r in phase2.players.itertuples()
    }

    design = build_design(stints, interaction_talent=talent)
    n_players = len(design.player_ids)
    X_lin = design.X[:, : 2 * n_players]
    F = np.asarray(design.X[:, 2 * n_players :].todense())
    W = design.weights
    FtWF = (F.T * W) @ F

    # anchor: user prior if given, else the decompressed phase-2 estimates
    effective_prior = prior if prior is not None else talent
    prior_vec = np.zeros(2 * n_players)
    for j, pid in enumerate(design.player_ids):
        if pid in effective_prior:
            o, d = effective_prior[pid]
            prior_vec[j] = o
            prior_vec[n_players + j] = -d

    gamma = np.zeros(2)
    coef = prior_vec
    intercept = 0.0
    for _ in range(8):
        model = _ridge(float(lam2))
        model.fit(
            X_lin, design.y - F @ gamma - X_lin @ prior_vec, sample_weight=W
        )
        coef = model.coef_ + prior_vec
        intercept = float(model.intercept_)
        # residual against the linear part only, so it still contains the
        # F-explained variation; gamma is re-solved in full, not incremented
        residual = design.y - intercept - X_lin @ coef
        new_gamma = np.linalg.solve(FtWF, (F.T * W) @ residual)
        new_gamma[0] = min(new_gamma[0], 0.0)  # offense: diminishing returns
        new_gamma[1] = max(new_gamma[1], 0.0)  # defense (points-allowed terms)
        if np.allclose(new_gamma, gamma, atol=1e-6):
            gamma = new_gamma
            break
        gamma = new_gamma

    orapm = coef[:n_players]
    drapm = -coef[n_players:]
    players = pd.DataFrame(
        {
            "player_id": design.player_ids,
            "orapm": orapm,
            "drapm": drapm,
            "rapm": orapm + drapm,
            "off_poss": design.off_poss,
            "def_poss": design.def_poss,
        }
    ).sort_values("rapm", ascending=False, ignore_index=True)

    meta = dict(phase1.meta)
    meta.update(
        {
            "lambda": float(lam2),
            "phase1_lambda": float(lam2),
            "intercept": float(model.intercept_),
            "prior": "custom" if prior else "phase2",
            "interaction": {
                # per-100 effect per 1 SD of positive-part-squared lineup
                # talent; negative offense value = diminishing returns
                "coef_off_sq": float(gamma[0]),
                "coef_def_sq": float(gamma[1]),
                "info": design.interaction_info,
                "talent": {
                    str(pid): [round(o, 4), round(d, 4)]
                    for pid, (o, d) in talent.items()
                },
            },
        }
    )
    return RapmResult(players=players, meta=meta)


def run_rapm(
    data_dir: Path,
    seasons: list[str],
    season_types: list[str],
    pool_seasons: bool = False,
    combine_types: bool = False,
    lam: float | str = "cv",
    min_poss: float = 0,
    fetch_names: bool = True,
    out_dir: Path | None = None,
    prior_kind: str = "none",
    prior_scale: float = 1.0,
    decay: float = 1.0,
    playoff_weight: float = 1.0,
    garbage_weight: float = 1.0,
    interactions: bool = False,
) -> list[Path]:
    """Fit RAPM for each requested scope and write CSV + meta JSON.

    Default scope is one output per (season, season type); ``pool_seasons``
    pools all seasons into a single regression, ``combine_types`` pools
    regular season + playoffs together.
    """
    from .names import get_player_names

    out_dir = out_dir or results_dir(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    season_scopes = [seasons] if pool_seasons else [[s] for s in seasons]
    type_scopes = [season_types] if combine_types else [[t] for t in season_types]

    written = []
    for scope_seasons in season_scopes:
        for scope_types in type_scopes:
            stints = load_stints(data_dir, scope_seasons, scope_types)
            if stints.empty:
                logger.warning(
                    "No stint rows for %s %s; skipping", scope_seasons, scope_types
                )
                continue
            if prior_kind == "two-phase":
                from .prior import two_phase_prior

                prior = two_phase_prior(stints, lam=lam, scale=prior_scale)
            elif prior_kind == "spm":
                from .prior import spm_prior

                prior = spm_prior(data_dir, scope_seasons, scope_types)
            elif prior_kind == "last-season":
                from .prior import last_season_prior

                prior = last_season_prior(
                    data_dir, scope_seasons, scope_types,
                    lam=lam, scale=prior_scale,
                )
            elif prior_kind == "none":
                prior = None
            else:
                raise ValueError(f"Unknown prior kind {prior_kind!r}")
            fit_kwargs = dict(
                lam=lam, prior=prior, decay=decay,
                playoff_weight=playoff_weight, garbage_weight=garbage_weight,
            )
            if interactions:
                result = fit_interaction_rapm(stints, **fit_kwargs)
            else:
                result = fit_rapm(stints, **fit_kwargs)
            result.meta["interactions"] = interactions
            result.meta["prior"] = prior_kind
            result.meta["prior_scale"] = (
                prior_scale if prior_kind in ("two-phase", "last-season") else None
            )
            players = result.players

            names = {}
            if fetch_names:
                names = get_player_names(
                    data_dir,
                    set(players["player_id"]),
                    season=max(scope_seasons),
                )
            players.insert(
                1, "player_name",
                players["player_id"].map(lambda p: names.get(p, "")),
            )
            if min_poss > 0:
                players = players[
                    (players["off_poss"] + players["def_poss"]) / 2 >= min_poss
                ].reset_index(drop=True)

            season_label = (
                scope_seasons[0]
                if len(scope_seasons) == 1
                else f"{scope_seasons[0]}_to_{scope_seasons[-1]}"
            )
            type_label = "_".join(season_type_slug(t) for t in scope_types)
            base = f"rapm_{season_label.replace('-', '_')}_{type_label}"
            if prior_kind != "none":
                base += f"_{prior_kind.replace('-', '')}"
            if interactions:
                base += "_interactions"
            csv_path = out_dir / f"{base}.csv"
            players.to_csv(csv_path, index=False, float_format="%.3f")
            meta = dict(result.meta)
            meta["seasons"] = scope_seasons
            meta["season_types"] = scope_types
            (out_dir / f"{base}.meta.json").write_text(json.dumps(meta, indent=1))
            logger.info("Wrote %s (%d players)", csv_path, len(players))
            written.append(csv_path)
    return written
