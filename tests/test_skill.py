"""Tests for src/skill.py on the shared fixture (see tests/test_pipeline.py for the card table).

Skill buckets in the fixture -> tiers (low < 0.52 <= mid < 0.58 <= high):

game colors won bucket tier | A in hand | B in hand
1    WU     T   0.50   low  | yes       | no
2    WU     F   0.54   mid  | yes       | yes
3    WU     T   0.50   low  | no        | -
4    BR     F   0.40   low  | -         | yes
5    BR     T   0.40   low  | -         | no
6    WUB    T   0.60   high | yes       | -
"""

from pathlib import Path

import duckdb
import pytest

from src.load import build_database
from src.skill import (
    card_winrates_by_tier,
    color_pair_winrates_by_tier,
    skill_adjusted_card_winrates,
    tier_summary,
    wilson_ci,
)

FIXTURE = Path(__file__).parent / "fixtures" / "game_data_small.csv"
A = "Alpha, the Tester"


@pytest.fixture
def con(tmp_path):
    db_path = tmp_path / "test.duckdb"
    build_database(FIXTURE, db_path)
    connection = duckdb.connect(str(db_path), read_only=True)
    yield connection
    connection.close()


def test_wilson_ci_known_value():
    low, high = wilson_ci(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(0.2775, abs=1e-4)


def test_wilson_ci_contains_rate_and_narrows_with_n():
    low_small, high_small = wilson_ci(55, 100)
    low_big, high_big = wilson_ci(5500, 10000)
    assert low_small < 0.55 < high_small
    assert low_big < 0.55 < high_big
    assert high_big - low_big < high_small - low_small


def test_tier_summary(con):
    df = tier_summary(con)
    assert list(df["tier"]) == ["low", "mid", "high"]
    assert list(df["games"]) == [4, 1, 1]
    assert list(df["wins"]) == [3, 0, 1]
    assert df["share"].sum() == pytest.approx(1.0)


def test_tier_summary_skips_missing_bucket():
    con = duckdb.connect()
    con.execute("CREATE TABLE games AS SELECT * FROM (VALUES (0.40, true), (NULL, true)) t(user_game_win_rate_bucket, won)")
    df = tier_summary(con)
    assert list(df["games"]) == [1]


def test_color_pair_winrates_by_tier(con):
    df = color_pair_winrates_by_tier(con).set_index(["color_pair", "tier"])
    assert (df.loc[("WU", "low"), "games"], df.loc[("WU", "low"), "wins"]) == (2, 2)
    assert (df.loc[("WU", "mid"), "games"], df.loc[("WU", "mid"), "wins"]) == (1, 0)
    assert df.loc[("BR", "low"), "win_rate"] == pytest.approx(0.5)
    assert ("WUB", "high") not in df.index  # not a pair


def test_card_winrates_by_tier(con):
    df = card_winrates_by_tier(con).set_index(["card", "tier"])
    assert df.loc[(A, "low"), "games"] == 1 and df.loc[(A, "low"), "wins"] == 1
    assert df.loc[(A, "mid"), "wins"] == 0
    assert df.loc[(A, "high"), "wins"] == 1
    assert ("Beta", "high") not in df.index


def test_skill_adjusted_card_winrates(con):
    df = skill_adjusted_card_winrates(con).set_index("card")
    assert list(df.index) == [A]  # Beta has no high-tier game -> dropped
    a = df.loc[A]
    assert a["raw_wr"] == pytest.approx(2 / 3)
    assert a["high_share"] == pytest.approx(1 / 3)
    # tier weights from all games: low 4/6, mid 1/6, high 1/6
    assert a["adjusted_wr"] == pytest.approx(4 / 6 * 1 + 1 / 6 * 0 + 1 / 6 * 1)
    assert a["bias"] == pytest.approx(2 / 3 - 5 / 6)


def test_skill_adjusted_min_games(con):
    assert skill_adjusted_card_winrates(con, min_games=2).empty
