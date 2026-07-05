import datetime

import pytest

from fablerapm.config import (
    current_season_end_year,
    parse_season_types,
    parse_seasons,
    season_str,
)


def test_season_str():
    assert season_str(1997) == "1996-97"
    assert season_str(2000) == "1999-00"
    assert season_str(2026) == "2025-26"


def test_current_season_rollover():
    assert current_season_end_year(datetime.date(2026, 7, 5)) == 2026
    assert current_season_end_year(datetime.date(2026, 10, 15)) == 2027


def test_parse_seasons_forms():
    assert parse_seasons("2019-20") == ["2019-20"]
    assert parse_seasons("1998-99:2001-02") == [
        "1998-99", "1999-00", "2000-01", "2001-02",
    ]
    assert parse_seasons("2019-20,2021-22") == ["2019-20", "2021-22"]
    all_seasons = parse_seasons("all")
    assert all_seasons[0] == "1996-97"
    assert len(all_seasons) >= 30


def test_parse_season_types():
    assert parse_season_types("regular,playoffs") == ["Regular Season", "Playoffs"]
    assert parse_season_types("Regular Season") == ["Regular Season"]
    assert parse_season_types("playin") == ["Play In"]
    with pytest.raises(ValueError):
        parse_season_types("preseason")
