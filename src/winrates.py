"""Win rates by color pair and by card, computed in DuckDB.

Card metrics (a game counts once per card, however many copies were seen):
- GP  (game played):          card in the main deck          deck > 0
- GIH (game in hand):         in opening hand or drawn later opening_hand + drawn > 0
- OH  (opening hand):         in the opening hand            opening_hand > 0
- GD  (game drawn):           drawn after the opening hand   drawn > 0
OH and GD overlap when one copy starts in hand and another is drawn.
Tutored copies are not counted as "in hand".

Usage:
    python -m src.winrates --min-games 500
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from src.load import DEFAULT_DB

COLOR_PAIR_SQL = """
    SELECT
        main_colors          AS color_pair,
        count(*)             AS games,
        count(*) FILTER (WHERE won) AS wins,
        avg(won::INT)        AS win_rate
    FROM games
    WHERE length(main_colors) = 2
    GROUP BY main_colors
    HAVING count(*) >= $min_games
    ORDER BY win_rate DESC
"""

CARD_SQL = """
    SELECT
        c.card,
        count(*)                                                    AS n_gp,
        avg(g.won::INT)                                             AS gp_wr,
        count(*)        FILTER (WHERE c.opening_hand + c.drawn > 0) AS n_gih,
        avg(g.won::INT) FILTER (WHERE c.opening_hand + c.drawn > 0) AS gih_wr,
        count(*)        FILTER (WHERE c.opening_hand > 0)           AS n_oh,
        avg(g.won::INT) FILTER (WHERE c.opening_hand > 0)           AS oh_wr,
        count(*)        FILTER (WHERE c.drawn > 0)                  AS n_gd,
        avg(g.won::INT) FILTER (WHERE c.drawn > 0)                  AS gd_wr
    FROM card_games AS c
    JOIN games AS g USING (game_id)
    GROUP BY c.card
    HAVING n_gih >= $min_games
    ORDER BY gih_wr DESC
"""


def color_pair_winrates(con: duckdb.DuckDBPyConnection, min_games: int = 0) -> pd.DataFrame:
    """Win rate per two-color main_colors (splash ignored, 1- and 3+-color decks excluded)."""
    return con.execute(COLOR_PAIR_SQL, {"min_games": min_games}).df()


def card_winrates(con: duckdb.DuckDBPyConnection, min_games: int = 0) -> pd.DataFrame:
    """GP/GIH/OH/GD win rates per card; keeps cards with at least min_games games in hand."""
    return con.execute(CARD_SQL, {"min_games": min_games}).df()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--min-games", type=int, default=500, help="minimum games (GIH games for cards)")
    args = parser.parse_args()

    con = duckdb.connect(str(args.db), read_only=True)
    pd.set_option("display.width", 140)
    print("Color pairs:")
    print(color_pair_winrates(con, args.min_games).to_string(index=False, float_format="{:.3f}".format))
    cards = card_winrates(con, args.min_games)
    print(f"\nCards with >= {args.min_games} GIH games: {len(cards)}. Top 10 by GIH WR:")
    print(cards.head(10).to_string(index=False, float_format="{:.3f}".format))
    con.close()


if __name__ == "__main__":
    main()
