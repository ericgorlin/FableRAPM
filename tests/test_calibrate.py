"""Null calibration on synthetic worlds with known curvature (offline).

Two ends of the contract:
  - In a truly additive world, the estimator's nonzero free-signed gammas
    must be recognized as bias: side-level curvature scores land inside
    the null band (no false finding).
  - When real curvature is injected everywhere (a global concave penalty),
    the fitted non-additivity must escape the null band somewhere.

The detection test asserts at the side-score level, not per-gamma: the
basis is collinear, so individual coordinates are unstable, and in a
teamless random-lineup world the jointly solved gamma blocks can shift
attribution across sides (documented in calibrate.py). The calibration's
claim is "real non-additivity vs estimator bias", and that is what these
tests pin down.
"""

import json

import numpy as np
import pandas as pd
import pytest

from fablerapm.calibrate import calibrate_curvature, format_report
from fablerapm.cli import main
from fablerapm.config import stints_path

from test_model import BASE_PER_POSS, N_PLAYERS, make_league, simulate_stints


def simulate_concave(true_off, true_def, k=10.0, n_games=500, seed=9):
    """Global diminishing returns: every offense pays k * (product of its
    two largest positive true talents) per possession."""
    rng = np.random.default_rng(seed)
    players = np.arange(N_PLAYERS)
    rows = []
    for g in range(n_games):
        for _ in range(25):
            ten = rng.choice(players, size=10, replace=False)
            off, dfn = ten[:5], ten[5:]
            t = sorted((max(x, 0.0) for x in true_off[off]), reverse=True)
            rate = (
                BASE_PER_POSS
                + true_off[off].sum()
                + true_def[dfn].sum()
                - k * t[0] * t[1]
            )
            poss = int(rng.integers(4, 15))
            rows.append({
                "game_id": f"G{g:05d}",
                "off_team_id": 1,
                "def_team_id": 2,
                "off_lineup": "-".join(str(p + 1) for p in sorted(off)),
                "def_lineup": "-".join(str(p + 1) for p in sorted(dfn)),
                "poss": poss,
                "points": int(rng.poisson(max(rate, 0.05) * poss)),
            })
    return pd.DataFrame(rows)


def starry_league():
    true_off, true_def = make_league()
    true_off[:6] = [0.06, 0.055, 0.05, 0.045, 0.04, 0.035]
    return true_off, true_def


def test_linear_world_yields_no_finding():
    # dataset seed note: a correctly-calibrated null band flags ~alpha of
    # additive datasets by construction, so any fixed seed is a draw from
    # that distribution. Across dataset seeds 1-7 the side-score
    # percentiles are spread roughly uniformly (5-90) with a flag rate at
    # the nominal alpha; seed 1 is a representative interior draw (off at
    # the 80th null percentile, def at the 55th), pinned so the test is
    # deterministic without sitting on the band's edge.
    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=300, seed=1)
    report = calibrate_curvature(stints, lam=500.0, n_sims=40, seed=0)

    # structure: both sides + all six gamma terms summarized
    assert set(report["sides"]) == {"off", "def"}
    assert len(report["terms"]) == 6
    for t in list(report["terms"].values()) + list(report["sides"].values()):
        assert t["null_lo"] <= t["null_hi"]
        assert np.isfinite(t["bias_corrected"])
    # additive truth: neither side's curvature-at-stacked-lineups score may
    # be reported as a finding — this is the false-positive contract
    assert not report["sides"]["off"]["outside_null"]
    assert not report["sides"]["def"]["outside_null"]
    assert "within null" in format_report(report)


def test_concave_world_is_flagged():
    true_off, true_def = starry_league()
    stints = simulate_concave(true_off, true_def, k=10.0)
    report = calibrate_curvature(stints, lam=500.0, n_sims=40, seed=0)
    # real curvature exists in every lineup: the fitted non-additivity must
    # escape the estimator's own null band on at least one side
    assert (
        report["sides"]["off"]["outside_null"]
        or report["sides"]["def"]["outside_null"]
    )
    assert "OUTSIDE NULL" in format_report(report)


def test_resample_null_matches_real_dispersion():
    import pandas as pd

    from fablerapm.calibrate import simulate_null_points
    from fablerapm.model import fit_rapm

    true_off, true_def = make_league()
    stints = simulate_stints(true_off, true_def, n_games=80, seed=12)
    result = fit_rapm(stints, lam=1000.0)
    rng = np.random.default_rng(0)

    sims = {
        kind: simulate_null_points(stints, result, rng, null=kind)
        for kind in ("poisson", "resample")
    }
    for kind, sim in sims.items():
        assert (sim["points"] >= 0).all()
        assert len(sim) == len(stints)
        # totals in the right ballpark (means match the fitted rates)
        assert 0.8 < sim["points"].sum() / stints["points"].sum() < 1.2
    # resampling redraws real residuals, so the studentized spread should
    # track the real data's more closely than a pure Poisson draw must
    poss = stints["poss"].to_numpy(dtype=float)

    def spread(points):
        y = 100.0 * np.asarray(points, dtype=float) / poss
        return np.std(y * np.sqrt(poss))

    real = spread(stints["points"])
    assert abs(spread(sims["resample"]["points"]) - real) < 0.25 * real

    with pytest.raises(ValueError, match="null"):
        simulate_null_points(stints, result, rng, null="bogus")


def test_cli_calibrate_curvature_writes_report(tmp_path):
    data_dir = tmp_path / "data"
    true_off, true_def = make_league()
    df = simulate_stints(true_off, true_def, n_games=40, seed=8)
    path = stints_path(data_dir, "2024-25", "Regular Season")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    rc = main([
        "calibrate-curvature", "--data-dir", str(data_dir),
        "--seasons", "2024-25", "--season-types", "regular",
        "--sims", "3", "--lambda", "1000",
    ])
    assert rc == 0
    out = data_dir / "results" / "curvature_calibration_2024_25_regularseason.json"
    assert out.exists()
    report = json.loads(out.read_text())
    assert report["n_sims"] == 3
    assert report["seasons"] == ["2024-25"]
    assert set(report["sides"]) == {"off", "def"}
    # 3 sims can't support a 95% null band: flags must be suppressed
    assert report["band_reliable"] is False
    assert not any(
        t["outside_null"]
        for t in list(report["terms"].values()) + list(report["sides"].values())
    )
