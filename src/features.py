"""Deck features per game, for the week 3 win prediction model.

Only information that is known before the game starts and describes the deck:
- main_colors, has_splash
- n_lands, and the mana curve of the non-land cards (counts by mana value)
- avg_mv: average mana value of the non-land cards
- on_play (not a deck feature, but fixed before the game and a known ~coin flip effect)

Deliberately NOT used:
- user_game_win_rate_bucket: contains the outcome of these games (target leakage)
- num_mulligans, drawn, opening_hand, num_turns: happen during the game
- user_n_games_bucket: describes the player, not the deck

Mana values come from the 17Lands card list (data/raw/cards.csv). The game dump
also contains bonus sheet cards from other sets, so cards are matched by name over
all expansions; every name in this set has one mana value, and the code stops if a
deck card has no mana value or more than one.

Usage:
    python -m src.features
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from src.load import DEFAULT_DB

DEFAULT_CARDS = Path("data/raw/cards.csv")

# Mana curve buckets for non-land cards: (column, lowest mana value, highest mana value).
CURVE_BUCKETS = [
    ("mv_0_1", 0, 1),
    ("mv_2", 2, 2),
    ("mv_3", 3, 3),
    ("mv_4", 4, 4),
    ("mv_5", 5, 5),
    ("mv_6plus", 6, 99),
]

# One row per card name. "Land" is checked only in the type line before " - "
# so that a subtype can never be mistaken for the Land card type.
CARD_INFO_SQL = """
    SELECT name AS card,
           count(DISTINCT mana_value) AS n_mana_values,
           any_value(mana_value) AS mana_value,
           bool_or(split_part(types, ' - ', 1) LIKE '%Land%') AS is_land
    FROM read_csv('{cards_csv}')
    GROUP BY name
"""


def check_card_info(con: duckdb.DuckDBPyConnection) -> None:
    """Stop if a deck card has no mana value or an ambiguous one."""
    problems = con.sql("""
        SELECT DISTINCT c.card
        FROM card_games AS c
        LEFT JOIN card_info AS i USING (card)
        WHERE i.card IS NULL OR i.n_mana_values <> 1
        ORDER BY c.card
    """).fetchall()
    if problems:
        names = ", ".join(row[0] for row in problems[:10])
        raise ValueError(f"{len(problems)} deck cards without a unique mana value, e.g. {names}")


def build_features(con: duckdb.DuckDBPyConnection, cards_csv: Path) -> pd.DataFrame:
    """Return one row per game with the target `won` and the deck features."""
    cards_sql = str(cards_csv).replace("'", "''")
    con.execute(f"CREATE OR REPLACE TEMP TABLE card_info AS {CARD_INFO_SQL.format(cards_csv=cards_sql)}")
    check_card_info(con)

    curve_cols = ",\n".join(
        f"coalesce(sum(c.deck) FILTER (WHERE NOT i.is_land AND i.mana_value BETWEEN {lo} AND {hi}), 0) AS {name}"
        for name, lo, hi in CURVE_BUCKETS
    )
    deck = con.sql(f"""
        SELECT c.game_id,
               coalesce(sum(c.deck) FILTER (WHERE i.is_land), 0) AS n_lands,
               {curve_cols},
               -- 0 for a deck without non-land cards (does not happen in real decks)
               coalesce(sum(c.deck * i.mana_value) FILTER (WHERE NOT i.is_land)
                        / sum(c.deck) FILTER (WHERE NOT i.is_land), 0) AS avg_mv
        FROM card_games AS c
        JOIN card_info AS i USING (card)
        GROUP BY c.game_id
    """)
    return con.sql("""
        SELECT g.game_id, g.draft_id, g.draft_time, g.won,
               g.main_colors,
               coalesce(g.splash_colors, '') <> '' AS has_splash,
               g.on_play,
               d.* EXCLUDE (game_id)
        FROM games AS g
        JOIN deck AS d USING (game_id)
        ORDER BY g.game_id
    """).df()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    args = parser.parse_args()

    con = duckdb.connect(str(args.db), read_only=True)
    df = build_features(con, args.cards)
    con.close()

    print(f"{len(df):,} games, {df.columns.size} columns")
    print(df.drop(columns=["game_id", "draft_id", "draft_time"]).describe().T.round(2).to_string())


if __name__ == "__main__":
    main()
