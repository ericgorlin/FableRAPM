"""Offline integration test through the REAL pbpstats parsing path.

Fabricates a raw stats.nba.com playbyplayv2 response for a small synthetic
game (4 periods of alternating made 2s, a jump ball, and one substitution),
writes it into the raw-data cache, and runs the actual pipeline entry point
(`game_stint_rows` -> pbpstats file loader -> possession parsing -> stint
aggregation). Every possession, point, and lineup in the fixture is known, so
the expected stint rows are exact.
"""

import json

from fablerapm.config import ensure_raw_dirs
from fablerapm.stints import game_stint_rows

GAME_ID = "0022300999"
HOME, AWAY = 1610612737, 1610612738
H = [101, 102, 103, 104, 105]  # home starters; 106 subs in for 105 in Q3
V = [201, 202, 203, 204, 205]

HEADERS = [
    "GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD",
    "WCTIMESTRING", "PCTIMESTRING", "HOMEDESCRIPTION", "NEUTRALDESCRIPTION",
    "VISITORDESCRIPTION", "SCORE", "SCOREMARGIN",
    "PERSON1TYPE", "PLAYER1_ID", "PLAYER1_NAME", "PLAYER1_TEAM_ID",
    "PLAYER1_TEAM_CITY", "PLAYER1_TEAM_NICKNAME", "PLAYER1_TEAM_ABBREVIATION",
    "PERSON2TYPE", "PLAYER2_ID", "PLAYER2_NAME", "PLAYER2_TEAM_ID",
    "PLAYER2_TEAM_CITY", "PLAYER2_TEAM_NICKNAME", "PLAYER2_TEAM_ABBREVIATION",
    "PERSON3TYPE", "PLAYER3_ID", "PLAYER3_NAME", "PLAYER3_TEAM_ID",
    "PLAYER3_TEAM_CITY", "PLAYER3_TEAM_NICKNAME", "PLAYER3_TEAM_ABBREVIATION",
    "VIDEO_AVAILABLE_FLAG",
]


class EventBuilder:
    def __init__(self):
        self.rows = []
        self.event_num = 0

    def add(self, msg_type, period, clock, p1=0, p1_team=None, p2=0,
            p2_team=None, p3=0, p3_team=None, action_type=0, desc="event"):
        self.event_num += 1
        self.rows.append([
            GAME_ID, self.event_num, msg_type, action_type, period,
            None, clock, desc, None, None, None, None,
            4, p1, f"P{p1}" if p1 else None, p1_team, None, None, None,
            5, p2, f"P{p2}" if p2 else None, p2_team, None, None, None,
            5, p3, f"P{p3}" if p3 else None, p3_team, None, None, None,
            0,
        ])


def lineup(players):
    return "-".join(str(p) for p in sorted(players))


def make_fixture_pbp():
    b = EventBuilder()
    for period in range(1, 5):
        b.add(12, period, "12:00", desc="Start Period")
        if period == 1:
            # jump ball: PLAYER1/2 jump, PLAYER3 recovers (away wins tip)
            b.add(10, 1, "11:55", p1=H[4], p1_team=HOME, p2=V[4],
                  p2_team=AWAY, p3=V[0], p3_team=AWAY, desc="Jump Ball")
        home_five = H if period <= 2 else [101, 102, 103, 104, 106]
        away_shooters, home_shooters = list(V), list(home_five)
        # 10 alternating made 2s: away at :00, home at :30 of each minute
        for i in range(5):
            minute = 11 - i
            b.add(1, period, f"{minute}:00", p1=away_shooters[i],
                  p1_team=AWAY, desc=f"P{away_shooters[i]} 2' Shot Made")
            if period == 3 and i == 4:
                # sub before home's final possession: 105 out, 106 in
                b.add(8, 3, f"{minute - 1}:45", p1=105, p1_team=HOME,
                      p2=106, p2_team=HOME, desc="SUB: 106 FOR 105")
                home_shooters[4] = 106
            shooter = home_shooters[i] if period != 3 else (
                H[i] if i < 4 else home_shooters[4]
            )
            b.add(1, period, f"{minute - 1}:30", p1=shooter,
                  p1_team=HOME, desc=f"P{shooter} 2' Shot Made")
        b.add(13, period, "0:00", desc="End Period")
    return {"resultSets": [{"name": "PlayByPlay", "headers": HEADERS,
                            "rowSet": b.rows}]}


def test_fixture_game_through_real_pbpstats(tmp_path):
    data_dir = tmp_path / "data"
    raw = ensure_raw_dirs(data_dir)
    (raw / "pbp" / f"stats_{GAME_ID}.json").write_text(
        json.dumps(make_fixture_pbp())
    )

    rows, warnings = game_stint_rows(data_dir, GAME_ID)
    assert warnings == []

    by_key = {(r["off_lineup"], r["def_lineup"]): r for r in rows}
    h1 = lineup(H)  # 101-105
    h2 = lineup([101, 102, 103, 104, 106])
    v = lineup(V)

    # Away had 5 scoring possessions/period vs starters in Q1-Q2, and in Q3
    # its 5 makes came before the home sub; Q4 is against the new home five.
    # After home's final make each period, away holds the ball until the
    # period ends: +1 scoreless possession per period (vs h2 in Q3/Q4, since
    # the sub happens before home's final make).
    away_vs_h1 = by_key[(v, h1)]
    assert away_vs_h1["poss"] == 3 * 5 + 2  # Q1-Q3 makes + Q1-Q2 period ends
    assert away_vs_h1["points"] == 30
    away_vs_h2 = by_key[(v, h2)]
    assert away_vs_h2["poss"] == 5 + 2  # Q4 makes + Q3-Q4 period ends
    assert away_vs_h2["points"] == 10

    # Home: Q1-Q2 all with starters; Q3 first 4 makes with starters, last
    # make after the sub; Q4 with the new five.
    home_h1 = by_key[(h1, v)]
    assert home_h1["poss"] == 2 * 5 + 4
    assert home_h1["points"] == 28
    home_h2 = by_key[(h2, v)]
    assert home_h2["poss"] == 1 + 5
    assert home_h2["points"] == 12

    assert sum(r["points"] for r in rows) == 80
    assert sum(r["poss"] for r in rows) == 44
    assert {r["off_team_id"] for r in rows} == {HOME, AWAY}
