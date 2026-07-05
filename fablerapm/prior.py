"""Priors for RAPM: SPM (statistical plus-minus from box/tracking features)
and two-phase (shrink toward a first-pass RAPM to counteract diminishing
returns on high-usage stars).

Both produce a ``dict[player_id, (orapm_prior, drapm_prior)]`` accepted by
``model.fit_rapm(prior=...)``.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

logger = logging.getLogger(__name__)

SPM_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


@dataclass
class SpmModel:
    """Ridge mapping standardized player features -> (ORAPM, DRAPM)."""

    feature_cols: list[str]
    mean: np.ndarray
    std: np.ndarray
    coef_o: np.ndarray
    coef_d: np.ndarray
    intercept_o: float
    intercept_d: float
    alpha_o: float
    alpha_d: float

    def to_json(self) -> str:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in self.__dict__.items()}
        return json.dumps(d, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "SpmModel":
        d = json.loads(text)
        for k in ("mean", "std", "coef_o", "coef_d"):
            d[k] = np.asarray(d[k], dtype=float)
        return cls(**d)


def _design(features: pd.DataFrame, cols, mean, std) -> np.ndarray:
    X = features[cols].to_numpy(dtype=float)
    X = (X - mean) / std
    return np.nan_to_num(X)  # missing features (e.g. no tracking) -> mean


def fit_spm(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    weights: np.ndarray | None = None,
) -> SpmModel:
    """Fit SPM on player rows.

    ``features``: player_id + numeric feature columns (one row per
    player-season). ``targets``: player_id, orapm, drapm from RAPM runs of
    the matching seasons. Rows are matched positionally after an inner merge
    on player_id (call with per-season frames concatenated; include a season
    column in both to disambiguate multi-season training).
    """
    on = ["player_id"] + [
        c for c in ("season", "season_type") if c in features and c in targets
    ]
    merged = features.merge(targets, on=on, how="inner", suffixes=("", "_t"))
    if len(merged) < 50:
        raise ValueError(f"Only {len(merged)} matched player rows to train SPM on")
    cols = [
        c for c in features.columns
        if c not in ("player_id", "season", "season_type", "minutes")
        and pd.api.types.is_numeric_dtype(features[c])
    ]
    X_raw = merged[cols].to_numpy(dtype=float)
    mean = np.nanmean(X_raw, axis=0)
    std = np.nanstd(X_raw, axis=0)
    std[std == 0] = 1.0
    X = _design(merged, cols, mean, std)
    if weights is None and "minutes" in merged:
        weights = merged["minutes"].to_numpy(dtype=float).clip(min=1.0)

    models = {}
    for target in ("orapm", "drapm"):
        ridge = RidgeCV(alphas=SPM_ALPHAS)
        ridge.fit(X, merged[target].to_numpy(dtype=float), sample_weight=weights)
        models[target] = ridge
    logger.info(
        "SPM fit on %d rows, %d features (alpha O=%g D=%g)",
        len(merged), len(cols), models["orapm"].alpha_, models["drapm"].alpha_,
    )
    return SpmModel(
        feature_cols=cols, mean=mean, std=std,
        coef_o=models["orapm"].coef_, coef_d=models["drapm"].coef_,
        intercept_o=float(models["orapm"].intercept_),
        intercept_d=float(models["drapm"].intercept_),
        alpha_o=float(models["orapm"].alpha_),
        alpha_d=float(models["drapm"].alpha_),
    )


def predict_spm(model: SpmModel, features: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Predict (orapm, drapm) priors for each player row.

    Players appearing multiple times (several seasons) are averaged weighted
    by minutes when available.
    """
    for c in model.feature_cols:
        if c not in features:
            features = features.assign(**{c: np.nan})
    X = _design(features, model.feature_cols, model.mean, model.std)
    pred = pd.DataFrame(
        {
            "player_id": features["player_id"].to_numpy(),
            "o": X @ model.coef_o + model.intercept_o,
            "d": X @ model.coef_d + model.intercept_d,
            "w": features["minutes"].to_numpy(dtype=float).clip(min=1.0)
            if "minutes" in features else 1.0,
        }
    )
    out = {}
    for pid, grp in pred.groupby("player_id"):
        w = grp["w"].to_numpy()
        out[int(pid)] = (
            float(np.average(grp["o"], weights=w)),
            float(np.average(grp["d"], weights=w)),
        )
    return out


def two_phase_prior(
    stints: pd.DataFrame,
    lam: float | str = "cv",
    scale: float = 1.0,
) -> dict[int, tuple[float, float]]:
    """Phase-one RAPM estimates as the shrinkage target for phase two.

    With scale=1, phase two re-fits the same data while shrinking toward the
    phase-one solution instead of zero, letting large (star) coefficients
    escape the flat shrinkage that causes diminishing-returns compression.
    """
    from .model import fit_rapm

    phase1 = fit_rapm(stints, lam=lam)
    return {
        int(r.player_id): (scale * r.orapm, scale * r.drapm)
        for r in phase1.players.itertuples()
    }


def last_season_prior(
    data_dir: Path,
    seasons: list[str],
    season_types: list[str],
    lam: float | str = "cv",
    scale: float = 0.7,
) -> dict[int, tuple[float, float]]:
    """RAPM from the season before the earliest season in scope, scaled.

    The cheapest stabilizer for single-season RAPM: last year's estimate is
    a real prior (computed from disjoint data, so nothing leaks). ``scale``
    < 1 reflects year-to-year regression toward the mean.
    """
    from .config import season_end_year, season_str
    from .model import fit_rapm, load_stints

    prev = season_str(min(season_end_year(s) for s in seasons) - 1)
    stints = load_stints(data_dir, [prev], season_types)
    logger.info("last-season prior: fitting %s %s", prev, season_types)
    result = fit_rapm(stints, lam=lam)
    return {
        int(r.player_id): (scale * r.orapm, scale * r.drapm)
        for r in result.players.itertuples()
    }


def spm_model_path(data_dir: Path) -> Path:
    return data_dir / "results" / "spm_model.json"


def train_spm(
    data_dir: Path,
    train_seasons: list[str],
    season_types: list[str],
    lam: float | str = "cv",
    garbage_weight: float = 1.0,
) -> Path:
    """Train an SPM from stored features + per-season RAPM fits, save JSON.

    Fits RAPM per (season, type) directly from stints (independent of any
    CSVs already written) so targets always exist when stint data does.
    Pass the same ``garbage_weight`` you use for final RAPM fits so the
    prior is trained to predict the same quantity. (Note the *features*
    are official season aggregates and always include garbage time — see
    README.)
    """
    from .features import load_features
    from .model import fit_rapm, load_stints

    target_frames = []
    for season in train_seasons:
        for season_type in season_types:
            stints = load_stints(data_dir, [season], [season_type])
            result = fit_rapm(stints, lam=lam, garbage_weight=garbage_weight)
            t = result.players[["player_id", "orapm", "drapm"]].copy()
            t["season"] = season
            t["season_type"] = season_type
            target_frames.append(t)
    targets = pd.concat(target_frames, ignore_index=True)
    features = load_features(data_dir, train_seasons, season_types)

    model = fit_spm(features, targets)
    path = spm_model_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.to_json())
    logger.info("Saved SPM model to %s", path)
    return path


def spm_prior(
    data_dir: Path, seasons: list[str], season_types: list[str]
) -> dict[int, tuple[float, float]]:
    """Load the saved SPM model and produce priors for the given scope."""
    from .features import load_features

    path = spm_model_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No SPM model at {path}. Run `fablerapm spm-train` first."
        )
    model = SpmModel.from_json(path.read_text())
    features = load_features(data_dir, seasons, season_types)
    return predict_spm(model, features)
