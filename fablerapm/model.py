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

from .config import results_dir, season_type_slug, stints_path

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
    weights: np.ndarray
    groups: np.ndarray  # game_id per row, for grouped CV
    player_ids: list[int]  # column j = offense, column n_players + j = defense
    off_poss: np.ndarray  # per player
    def_poss: np.ndarray
    dropped_points: int  # points in rows with 0 possessions (orphaned technical FTs)


def _parse_lineup(lineup: str) -> list[int]:
    return [int(p) for p in lineup.split("-")]


def build_design(stints: pd.DataFrame) -> Design:
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
    weights = poss

    poss_by_col = np.asarray(X.multiply(weights[:, None]).sum(axis=0)).ravel()
    return Design(
        X=X,
        y=y,
        weights=weights,
        groups=stints["game_id"].to_numpy(),
        player_ids=player_ids,
        off_poss=poss_by_col[:n_players],
        def_poss=poss_by_col[n_players:],
        dropped_points=dropped_points,
    )


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
) -> RapmResult:
    """Fit RAPM on stint rows.

    ``prior`` optionally maps player_id -> (orapm_prior, drapm_prior) in
    per-100 units (DRAPM prior in positive-is-good convention). Coefficients
    are shrunk toward the prior instead of zero; players missing from the
    prior shrink toward 0.
    """
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
    drapm = -coef[n_players:]  # flip so positive = good defense
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
    }
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
            elif prior_kind == "none":
                prior = None
            else:
                raise ValueError(f"Unknown prior kind {prior_kind!r}")
            result = fit_rapm(stints, lam=lam, prior=prior)
            result.meta["prior"] = prior_kind
            result.meta["prior_scale"] = prior_scale if prior_kind == "two-phase" else None
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
            csv_path = out_dir / f"{base}.csv"
            players.to_csv(csv_path, index=False, float_format="%.3f")
            meta = dict(result.meta)
            meta["seasons"] = scope_seasons
            meta["season_types"] = scope_types
            (out_dir / f"{base}.meta.json").write_text(json.dumps(meta, indent=1))
            logger.info("Wrote %s (%d players)", csv_path, len(players))
            written.append(csv_path)
    return written
