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
    """Ridge mapping standardized player features -> (ORAPM, DRAPM).

    ``train_seasons`` / ``train_season_types`` are provenance: the seasons
    whose RAPM fits produced the training targets. ``spm_prior`` warns when
    the model is applied to a season it was trained on, since its targets
    saw those games and any evaluation there is optimistic. None on
    artifacts saved before provenance existed.
    """

    feature_cols: list[str]
    mean: np.ndarray
    std: np.ndarray
    coef_o: np.ndarray
    coef_d: np.ndarray
    intercept_o: float
    intercept_d: float
    alpha_o: float
    alpha_d: float
    train_seasons: list[str] | None = None
    train_season_types: list[str] | None = None

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
        if c not in ("player_id", "season", "season_type", "minutes", "age")
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
    **fit_kwargs,
) -> dict[int, tuple[float, float]]:
    """Phase-one RAPM estimates as the shrinkage target for phase two.

    With scale=1, phase two re-fits the same data while shrinking toward the
    phase-one solution instead of zero, letting large (star) coefficients
    escape the flat shrinkage that causes diminishing-returns compression.
    Extra keyword arguments (decay, playoff_weight, garbage_weight,
    garbage_rule) are forwarded to the phase-one fit.
    """
    from .model import fit_rapm

    phase1 = fit_rapm(stints, lam=lam, **fit_kwargs)
    return {
        int(r.player_id): (scale * r.orapm, scale * r.drapm)
        for r in phase1.players.itertuples()
    }


# Aging curve: how much of last season's RAPM persists into this season,
# by age. Learned from consecutive-season RAPM pairs as a through-origin
# weighted regression slope per age bucket (O and D components stacked),
# shrunk toward the global slope by player count. The global slope is the
# data-driven version of the single --prior-scale number; buckets let
# young players keep more of their (improving) signal and old players less.
AGE_BUCKETS = [(0, 23), (24, 26), (27, 29), (30, 32), (33, 200)]
AGE_CURVE_SHRINK_N = 25  # pseudo-players pulling a bucket toward global
AGE_CURVE_MIN_POSS = 500  # per-season possessions to enter the regression


def age_curve_path(data_dir: Path) -> Path:
    return data_dir / "results" / "age_curve.json"


def _age_bucket(age: float) -> str:
    for lo, hi in AGE_BUCKETS:
        if lo <= age <= hi:
            return f"{lo}-{hi}" if lo > 0 and hi < 200 else (
                f"<={hi}" if lo == 0 else f"{lo}+"
            )
    raise ValueError(f"age {age} outside all buckets")


def player_ages(
    data_dir: Path, season: str, season_types: list[str]
) -> dict[int, float]:
    """Player ages in a season, from the stored feature parquets."""
    from .features import load_features

    features = load_features(data_dir, [season], season_types)
    if "age" not in features.columns or features["age"].isna().all():
        raise FileNotFoundError(
            f"Features for {season} lack the age column (built before ages "
            f"were stored); re-run `fablerapm features --seasons {season}` "
            "— offline when the raw responses are cached"
        )
    ages = features.dropna(subset=["age"]).groupby("player_id")["age"].max()
    return {int(pid): float(a) for pid, a in ages.items()}


def _ages_for_target(
    data_dir: Path, season: str, prev: str, season_types: list[str]
) -> dict[int, float]:
    """Ages in the target season, falling back to previous-season ages + 1
    (the target season's features may not be scraped yet mid-season)."""
    try:
        return player_ages(data_dir, season, season_types)
    except FileNotFoundError:
        ages = player_ages(data_dir, prev, season_types)
        return {pid: a + 1.0 for pid, a in ages.items()}


def learn_age_curve(
    data_dir: Path,
    seasons: list[str],
    season_types: list[str],
    lam: float | str = "cv",
    min_poss: float = AGE_CURVE_MIN_POSS,
) -> dict:
    """Learn per-age-bucket persistence of RAPM across consecutive seasons.

    For every consecutive pair among ``seasons``, fit RAPM on both, join
    players present in both (with at least ``min_poss`` average possessions
    per season), and regress season-t values on season-(t-1) values through
    the origin — O and D components stacked, weighted by the smaller
    season's possessions — within age buckets (age = target-season age).
    """
    from .config import season_end_year
    from .model import fit_rapm, load_stints

    ordered = sorted(seasons, key=season_end_year)
    pairs = [
        (a, b) for a, b in zip(ordered, ordered[1:])
        if season_end_year(b) == season_end_year(a) + 1
    ]
    if not pairs:
        raise ValueError(
            f"Need at least two consecutive seasons to learn an age curve, "
            f"got {seasons}"
        )

    obs: list[tuple[str, float, float, float]] = []  # bucket, x, y, w
    fits: dict[str, pd.DataFrame] = {}

    def fit(season: str) -> pd.DataFrame:
        if season not in fits:
            logger.info("age curve: fitting %s %s", season, season_types)
            result = fit_rapm(
                load_stints(data_dir, [season], season_types), lam=lam
            )
            players = result.players.set_index("player_id")
            players["poss"] = (players["off_poss"] + players["def_poss"]) / 2
            fits[season] = players
        return fits[season]

    for prev, curr in pairs:
        prev_fit, curr_fit = fit(prev), fit(curr)
        ages = _ages_for_target(data_dir, curr, prev, season_types)
        shared = prev_fit.index.intersection(curr_fit.index)
        for pid in shared:
            if pid not in ages:
                continue
            w = float(min(prev_fit.loc[pid, "poss"], curr_fit.loc[pid, "poss"]))
            if w < min_poss:
                continue
            bucket = _age_bucket(ages[pid])
            for comp in ("orapm", "drapm"):
                obs.append((
                    bucket,
                    float(prev_fit.loc[pid, comp]),
                    float(curr_fit.loc[pid, comp]),
                    w,
                ))

    if not obs:
        raise ValueError(
            "No player-season pairs survived the possession filter; "
            "lower --min-poss or add seasons"
        )

    df = pd.DataFrame(obs, columns=["bucket", "x", "y", "w"])
    df["wxy"] = df["w"] * df["x"] * df["y"]
    df["wxx"] = df["w"] * df["x"] * df["x"]
    global_scale = float(df["wxy"].sum() / df["wxx"].sum())
    buckets = {}
    for (lo, hi) in AGE_BUCKETS:
        label = _age_bucket(lo if lo > 0 else hi)
        grp = df[df["bucket"] == label]
        n = grp["x"].size // 2  # two components per player-season pair
        if grp["wxx"].sum() > 0:
            raw = float(grp["wxy"].sum() / grp["wxx"].sum())
        else:
            raw = global_scale
        # shrink small buckets toward the global slope; clip to sane range
        scale = (n * raw + AGE_CURVE_SHRINK_N * global_scale) / (
            n + AGE_CURVE_SHRINK_N
        )
        buckets[label] = {
            "scale": float(np.clip(scale, 0.0, 1.5)),
            "raw": raw,
            "players": int(n),
        }
    return {
        "buckets": buckets,
        "global_scale": global_scale,
        "seasons": ordered,
        "season_types": season_types,
        "lambda": lam if lam == "cv" else float(lam),
        "min_poss": float(min_poss),
        "n_pairs": len(pairs),
    }


def train_age_curve(
    data_dir: Path,
    seasons: list[str],
    season_types: list[str],
    lam: float | str = "cv",
    min_poss: float = AGE_CURVE_MIN_POSS,
) -> Path:
    curve = learn_age_curve(data_dir, seasons, season_types, lam, min_poss)
    path = age_curve_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(curve, indent=1))
    logger.info("Saved age curve to %s", path)
    return path


def load_age_curve(data_dir: Path) -> dict:
    path = age_curve_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No age curve at {path}. Run `fablerapm age-curve` first."
        )
    return json.loads(path.read_text())


def last_season_prior(
    data_dir: Path,
    seasons: list[str],
    season_types: list[str],
    lam: float | str = "cv",
    scale: float | str = 0.7,
) -> dict[int, tuple[float, float]]:
    """RAPM from the season before the earliest season in scope, scaled.

    The cheapest stabilizer for single-season RAPM: last year's estimate is
    a real prior (computed from disjoint data, so nothing leaks). ``scale``
    < 1 reflects year-to-year regression toward the mean; ``scale="age"``
    replaces the single number with the learned aging curve (players
    missing an age get the curve's global scale).
    """
    from .config import season_end_year, season_str
    from .model import fit_rapm, load_stints

    target = min(seasons, key=season_end_year)
    prev = season_str(season_end_year(target) - 1)
    stints = load_stints(data_dir, [prev], season_types)
    logger.info("last-season prior: fitting %s %s", prev, season_types)
    result = fit_rapm(stints, lam=lam)

    if scale == "age":
        curve = load_age_curve(data_dir)
        ages = _ages_for_target(data_dir, target, prev, season_types)

        def scale_of(pid: int) -> float:
            age = ages.get(pid)
            if age is None:
                return float(curve["global_scale"])
            return float(curve["buckets"][_age_bucket(age)]["scale"])
    else:
        s = float(scale)

        def scale_of(pid: int) -> float:
            return s

    return {
        int(r.player_id): (
            scale_of(int(r.player_id)) * r.orapm,
            scale_of(int(r.player_id)) * r.drapm,
        )
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
    model.train_seasons = list(train_seasons)
    model.train_season_types = list(season_types)
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
    if model.train_seasons:
        overlap = sorted(set(model.train_seasons) & set(seasons))
        # same season string but disjoint season types = disjoint games
        if overlap and model.train_season_types is not None and not (
            set(model.train_season_types) & set(season_types)
        ):
            overlap = []
        if overlap:
            logger.warning(
                "SPM prior was trained on %s, which overlaps the current "
                "scope: its RAPM targets saw these games, so any holdout "
                "evaluation here is optimistic. Retrain on disjoint seasons "
                "(`fablerapm spm-train`).",
                ", ".join(overlap),
            )
    features = load_features(data_dir, seasons, season_types)
    return predict_spm(model, features)
