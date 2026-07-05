"""Stint extraction from pbpstats-shaped possession stats (offline)."""

import pytest

from fablerapm.stints import check_stint_rows, possessions_to_stint_rows

TEAM_A, TEAM_B = 100, 200
LINEUP_A = "-".join(str(i) for i in range(1, 6))  # players 1-5
LINEUP_B = "-".join(str(i) for i in range(6, 11))  # players 6-10
LINEUP_B2 = "-".join(str(i) for i in [6, 7, 8, 9, 11])  # sub: 10 -> 11


class FakePossession:
    def __init__(self, stats, score=None, period=1, start_time="10:00", margin=0):
        self.possession_stats = stats
        self.period = period
        self.start_time = start_time
        self.start_score_margin = margin
        if score is not None:
            self.events = [type("E", (), {"score": score})()]


def player_rows(stat_key, value, team_id, lineup, opp_team_id, opp_lineup):
    """One stat row per player on the credited team, like pbpstats emits."""
    return [
        {
            "player_id": int(p),
            "team_id": team_id,
            "opponent_team_id": opp_team_id,
            "lineup_id": lineup,
            "opponent_lineup_id": opp_lineup,
            "stat_key": stat_key,
            "stat_value": value,
        }
        for p in lineup.split("-")
    ]


def possession(off_team, off_lineup, def_team, def_lineup, points, **kwargs):
    """A normal possession: OffPoss for the offense, points against the defense."""
    stats = player_rows("OffPoss", 1, off_team, off_lineup, def_team, def_lineup)
    if points:
        stats += player_rows(
            "OpponentPoints", points, def_team, def_lineup, off_team, off_lineup
        )
    return FakePossession(stats, **kwargs)


def test_basic_aggregation():
    possessions = [
        possession(TEAM_A, LINEUP_A, TEAM_B, LINEUP_B, 2),
        possession(TEAM_B, LINEUP_B, TEAM_A, LINEUP_A, 0),
        possession(TEAM_A, LINEUP_A, TEAM_B, LINEUP_B, 3),
        possession(TEAM_B, LINEUP_B2, TEAM_A, LINEUP_A, 2),
    ]
    rows = possessions_to_stint_rows(possessions, "TEST1")
    by_key = {(r["off_lineup"], r["def_lineup"]): r for r in rows}
    assert len(rows) == 3
    a_stint = by_key[(LINEUP_A, LINEUP_B)]
    assert a_stint["poss"] == 2 and a_stint["points"] == 5
    assert a_stint["off_team_id"] == TEAM_A and a_stint["def_team_id"] == TEAM_B
    assert by_key[(LINEUP_B, LINEUP_A)]["poss"] == 1
    assert by_key[(LINEUP_B, LINEUP_A)]["points"] == 0
    b2_stint = by_key[(LINEUP_B2, LINEUP_A)]
    assert b2_stint["poss"] == 1 and b2_stint["points"] == 2


def test_orphaned_points_kept_as_zero_poss_row():
    # Technical FT scored by B during A's possession, and B's lineup never
    # has an offensive possession against A's lineup: row with poss=0.
    tech = FakePossession(
        player_rows("OffPoss", 1, TEAM_A, LINEUP_A, TEAM_B, LINEUP_B)
        + player_rows("OpponentPoints", 1, TEAM_A, LINEUP_A, TEAM_B, LINEUP_B)
    )
    rows = possessions_to_stint_rows([tech], "TEST2")
    by_key = {(r["off_lineup"], r["def_lineup"]): r for r in rows}
    assert by_key[(LINEUP_A, LINEUP_B)]["poss"] == 1
    orphan = by_key[(LINEUP_B, LINEUP_A)]
    assert orphan["poss"] == 0 and orphan["points"] == 1


def test_invalid_lineup_size_raises():
    bad = FakePossession(
        player_rows("OffPoss", 1, TEAM_A, "1-2-3-4", TEAM_B, LINEUP_B)
    )
    with pytest.raises(ValueError, match="not divisible by 5|5 players"):
        possessions_to_stint_rows([bad], "TEST3")


def test_garbage_time_rule():
    from types import SimpleNamespace

    from fablerapm.stints import is_garbage_time

    def g(period, clock, margin):
        return is_garbage_time(
            SimpleNamespace(period=period, start_time=clock, start_score_margin=margin)
        )

    assert g(4, "10:00", 30) and g(4, "10:00", -30)
    assert not g(4, "10:00", 20)  # threshold is 25 early in Q4
    assert g(4, "5:00", 20)  # 18 inside 6 minutes
    assert g(4, "2:00", -13)  # 12 inside 3 minutes
    assert not g(2, "1:00", 40)  # never before Q4
    assert g(5, "3:00", 12)  # overtime counts


def test_garbage_possessions_split_into_separate_rows():
    possessions = [
        possession(TEAM_A, LINEUP_A, TEAM_B, LINEUP_B, 2),
        possession(TEAM_A, LINEUP_A, TEAM_B, LINEUP_B, 3,
                   period=4, start_time="2:00", margin=28),
    ]
    rows = possessions_to_stint_rows(possessions, "TESTG")
    by_garbage = {r["garbage"]: r for r in rows}
    assert len(rows) == 2
    assert by_garbage[False]["poss"] == 1 and by_garbage[False]["points"] == 2
    assert by_garbage[True]["poss"] == 1 and by_garbage[True]["points"] == 3


def test_score_reconciliation():
    possessions = [
        possession(TEAM_A, LINEUP_A, TEAM_B, LINEUP_B, 2),
        possession(TEAM_B, LINEUP_B, TEAM_A, LINEUP_A, 3),
    ]
    possessions[-1].events = [
        type("E", (), {"score": {TEAM_A: 2, TEAM_B: 3}})()
    ]
    rows = possessions_to_stint_rows(possessions, "TEST4")
    assert check_stint_rows(possessions, rows) == []
    possessions[-1].events[0].score = {TEAM_A: 2, TEAM_B: 5}
    assert len(check_stint_rows(possessions, rows)) == 1
