"""Load a 17Lands game_data dump into DuckDB.

Builds two tables:
- games:      one row per game (game_id + the non-card columns of the dump)
- card_games: one row per (game, card) for every card in the main deck,
              with the counts deck / opening_hand / drawn / tutored / sideboard

Usage:
    python -m src.load
    python -m src.load --csv data/raw/<file>.csv.gz --db data/processed/sos.duckdb
"""

import argparse
import time
from pathlib import Path

import duckdb

DEFAULT_CSV = Path("data/raw/game_data_public.SOS.PremierDraft.csv.gz")
DEFAULT_DB = Path("data/processed/sos.duckdb")

# Per-card column groups in the dump, e.g. "deck_Abigale, Poet Laureate".
CARD_PREFIXES = ["deck", "opening_hand", "drawn", "tutored", "sideboard"]


def quote_ident(name: str) -> str:
    """Quote an SQL identifier; card names contain commas, spaces and apostrophes."""
    return '"' + name.replace('"', '""') + '"'


def split_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    """Split dump columns into game-level columns and card names."""
    card_cols = [c for c in columns if any(c.startswith(p + "_") for p in CARD_PREFIXES)]
    game_cols = [c for c in columns if c not in set(card_cols)]
    cards = sorted(c.removeprefix("deck_") for c in card_cols if c.startswith("deck_"))
    return game_cols, cards


def build_database(csv_path: Path, db_path: Path) -> None:
    """(Re)build the DuckDB file from the raw CSV. The raw file is only read."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)  # derived file: always rebuilt from scratch

    con = duckdb.connect(str(db_path))
    csv_sql = str(csv_path).replace("'", "''")

    # Read the CSV once; game_id ties both output tables to the same row.
    con.execute(
        f"CREATE TEMP TABLE raw AS "
        f"SELECT row_number() OVER () AS game_id, * FROM read_csv('{csv_sql}')"
    )
    columns = [row[0] for row in con.sql("SELECT column_name FROM (DESCRIBE raw)").fetchall()]
    game_cols, cards = split_columns(columns)

    con.execute(f"CREATE TABLE games AS SELECT {', '.join(map(quote_ident, game_cols))} FROM raw")

    # UNPIVOT with several value columns: each card's 5 columns become one row.
    groups = ",\n".join(
        "(" + ", ".join(quote_ident(f"{p}_{card}") for p in CARD_PREFIXES) + f") AS {quote_ident(card)}"
        for card in cards
    )
    con.execute(f"""
        CREATE TABLE card_games AS
        SELECT game_id, card, {', '.join(CARD_PREFIXES)}
        FROM (
            UNPIVOT raw
            ON {groups}
            INTO NAME card VALUE {', '.join(CARD_PREFIXES)}
        )
        WHERE deck > 0
    """)
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    start = time.perf_counter()
    build_database(args.csv, args.db)
    con = duckdb.connect(str(args.db), read_only=True)
    for table in ["games", "card_games"]:
        n = con.sql(f"SELECT count(*) FROM {table}").fetchall()[0][0]
        print(f"{table}: {n:,} rows")
    con.close()
    print(f"done in {time.perf_counter() - start:.0f}s -> {args.db}")


if __name__ == "__main__":
    main()
