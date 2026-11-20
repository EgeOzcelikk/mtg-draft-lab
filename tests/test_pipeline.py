"""Tests for src/load.py and src/winrates.py on a small synthetic fixture.

Fixture (tests/fixtures/game_data_small.csv), 6 games, cards A = "Alpha, the Tester", B = "Beta":

game colors won | A: oh drawn deck | B: oh drawn deck
1    WU     T   |    1   0     1   |    0   0     1
2    WU     F   |    0   1     1   |    1   0     2
3    WU     T   |    0   0     1   |    0   0     0  (B in sideboard)
4    BR     F   |    0   0     0   |    0   1     1  (A in sideboard)
5    BR     T   |    0   0     0   |    0   0     1
6    WUB    T   |    1   1     2   |    0   0     0
"""

from pathlib import Path

import duckdb
import pytest

from src.load import build_database, split_columns
from src.winrates import card_winrates, color_pair_winrates

FIXTURE = Path(__file__).parent / "fixtures" / "game_data_small.csv"
A = "Alpha, the Tester"


@pytest.fixture
def con(tmp_path):
    db_path = tmp_path / "test.duckdb"
    build_database(FIXTURE, db_path)
    connection = duckdb.connect(str(db_path), read_only=True)
    yield connection
    connection.close()


def test_split_columns_keeps_commas_in_card_names():
    game_cols, cards = split_columns(["won", "deck_Alpha, the Tester", "drawn_Alpha, the Tester", "deck_Beta"])
    assert game_cols == ["won"]
    assert cards == ["Alpha, the Tester", "Beta"]


def test_games_table(con):
    assert con.sql("SELECT count(*) FROM games").fetchall() == [(6,)]
    columns = [row[0] for row in con.sql("SELECT column_name FROM (DESCRIBE games)").fetchall()]
    assert columns[0] == "game_id"
    assert not any(c.startswith(("deck_", "drawn_", "sideboard_")) for c in columns)


def test_card_games_has_only_main_deck_cards(con):
    rows = con.sql("SELECT game_id, card, deck FROM card_games ORDER BY game_id, card").fetchall()
    assert rows == [
        (1, A, 1), (1, "Beta", 1),
        (2, A, 1), (2, "Beta", 2),
        (3, A, 1),
        (4, "Beta", 1),
        (5, "Beta", 1),
        (6, A, 2),
    ]


def test_color_pair_winrates(con):
    df = color_pair_winrates(con)
    assert list(df["color_pair"]) == ["WU", "BR"]  # WUB is not a pair
    wu = df.set_index("color_pair").loc["WU"]
    assert (wu["games"], wu["wins"]) == (3, 2)
    assert wu["win_rate"] == pytest.approx(2 / 3)


def test_color_pair_min_games(con):
    assert list(color_pair_winrates(con, min_games=3)["color_pair"]) == ["WU"]


def test_card_winrates(con):
    df = card_winrates(con).set_index("card")
    a, b = df.loc[A], df.loc["Beta"]

    assert (a["n_gp"], a["gp_wr"]) == (4, pytest.approx(3 / 4))
    assert (a["n_gih"], a["gih_wr"]) == (3, pytest.approx(2 / 3))
    assert (a["n_oh"], a["oh_wr"]) == (2, pytest.approx(1.0))
    assert (a["n_gd"], a["gd_wr"]) == (2, pytest.approx(1 / 2))

    assert (b["n_gp"], b["gp_wr"]) == (4, pytest.approx(1 / 2))
    assert (b["n_gih"], b["gih_wr"]) == (2, pytest.approx(0.0))
    assert (b["n_oh"], b["oh_wr"]) == (1, pytest.approx(0.0))
    assert (b["n_gd"], b["gd_wr"]) == (1, pytest.approx(0.0))


def test_card_min_games_filters_on_gih(con):
    assert list(card_winrates(con, min_games=3)["card"]) == [A]
