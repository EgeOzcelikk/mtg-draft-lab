"""Tests for src/features.py and src/model.py.

Cards fixture: Alpha = 2-mana creature (listed in two expansions), Beta = land.
Decks in the game fixture -> expected features:

game | Alpha Beta | n_lands mv_2 avg_mv | splash on_play
1    | 1     1    | 1       1    2.0    | no     yes
2    | 1     2    | 2       1    2.0    | no     no
3    | 1     0    | 0       1    2.0    | no     yes
4    | 0     1    | 1       0    0.0    | no     yes
5    | 0     1    | 1       0    0.0    | G      no
6    | 2     0    | 0       2    2.0    | no     yes
"""

from pathlib import Path

import duckdb
import pytest

from src.features import build_features
from src.load import build_database

FIXTURES = Path(__file__).parent / "fixtures"
CARDS = FIXTURES / "cards_small.csv"


@pytest.fixture
def con(tmp_path):
    db_path = tmp_path / "test.duckdb"
    build_database(FIXTURES / "game_data_small.csv", db_path)
    connection = duckdb.connect(str(db_path), read_only=True)
    yield connection
    connection.close()


def test_build_features(con):
    df = build_features(con, CARDS).set_index("game_id")
    assert df["n_lands"].tolist() == [1, 2, 0, 1, 1, 0]
    assert df["mv_2"].tolist() == [1, 1, 1, 0, 0, 2]
    assert df["avg_mv"].tolist() == [2.0, 2.0, 2.0, 0.0, 0.0, 2.0]
    assert df["has_splash"].tolist() == [False, False, False, False, True, False]
    assert df["on_play"].tolist() == [True, False, True, True, False, True]
    # No card in another bucket; the 7-mana card is not in any deck.
    other = ["mv_0_1", "mv_3", "mv_4", "mv_5", "mv_6plus"]
    assert (df[other] == 0).all().all()


def write_cards(path: Path, rows: list[str]) -> Path:
    path.write_text("id,expansion,name,rarity,color_identity,mana_value,types,is_booster\n" + "\n".join(rows) + "\n")
    return path


def test_missing_mana_value_raises(con, tmp_path):
    cards = write_cards(tmp_path / "cards.csv", ['1,TST,"Alpha, the Tester",common,W,2,Creature,True'])
    with pytest.raises(ValueError, match="1 deck cards"):
        build_features(con, cards)


def test_ambiguous_mana_value_raises(con, tmp_path):
    cards = write_cards(tmp_path / "cards.csv", [
        '1,TST,"Alpha, the Tester",common,W,2,Creature,True',
        '2,OLD,"Alpha, the Tester",common,W,3,Creature,True',
        "3,TST,Beta,common,,0,Land,True",
    ])
    with pytest.raises(ValueError, match="Alpha, the Tester"):
        build_features(con, cards)

