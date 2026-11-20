"""Win rates split by player skill, selection bias and confidence intervals.

Skill tier comes from user_game_win_rate_bucket, the player's overall game win rate
on 17Lands (rounded to 0.02 steps):
    low  < 0.52 <= mid < 0.58 <= high  (about equal thirds of games)
Games without a bucket are left out (in this dump they are all wins, which looks like
a data artefact). The bucket includes the games we analyse, so it is fine for grouping
but must never be used as a model feature (target leakage).

Skill-adjusted win rate (direct standardization): a card's per-tier GIH win rates,
weighted by the tier mix of ALL games instead of the tier mix of the players who
happened to play the card. raw - adjusted = how much of the raw win rate comes from
who plays the card rather than from the card itself.

Usage:
    python -m src.skill --min-games 200
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.load import DEFAULT_DB

TIERS = ["low", "mid", "high"]

TIER_SQL = """
    CASE
        WHEN user_game_win_rate_bucket < 0.52 THEN 'low'
        WHEN user_game_win_rate_bucket < 0.58 THEN 'mid'
        ELSE 'high'
    END
"""

TIER_SUMMARY_SQL = f"""
    SELECT {TIER_SQL} AS tier, count(*) AS games, count(*) FILTER (WHERE won) AS wins
    FROM games
    WHERE user_game_win_rate_bucket IS NOT NULL
    GROUP BY tier
"""

COLOR_PAIR_TIER_SQL = f"""
    SELECT main_colors AS color_pair, {TIER_SQL} AS tier,
           count(*) AS games, count(*) FILTER (WHERE won) AS wins
    FROM games
    WHERE length(main_colors) = 2 AND user_game_win_rate_bucket IS NOT NULL
    GROUP BY color_pair, tier
"""

# GIH games only (card in opening hand or drawn), same definition as src/winrates.py.
CARD_TIER_SQL = f"""
    SELECT c.card, {TIER_SQL} AS tier,
           count(*) AS games, count(*) FILTER (WHERE g.won) AS wins
    FROM card_games AS c
    JOIN games AS g USING (game_id)
    WHERE c.opening_hand + c.drawn > 0 AND g.user_game_win_rate_bucket IS NOT NULL
    GROUP BY c.card, tier
"""


def wilson_ci(wins, n, z: float = 1.96):
    """95% Wilson score interval for a win rate; works on scalars and arrays.

    Unlike the simple p +/- z*sqrt(p(1-p)/n) interval it stays inside [0, 1]
    and behaves well for small n or rates near 0 and 1.
    """
    wins, n = np.asarray(wins, dtype=float), np.asarray(n, dtype=float)
    p = wins / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return center - half, center + half


def _add_rate_and_ci(df: pd.DataFrame) -> pd.DataFrame:
    df["win_rate"] = df["wins"] / df["games"]
    df["ci_low"], df["ci_high"] = wilson_ci(df["wins"], df["games"])
    return df


def _order_tiers(df: pd.DataFrame) -> pd.DataFrame:
    df["tier"] = pd.Categorical(df["tier"], categories=TIERS, ordered=True)
    return df


def tier_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Games, share of games and win rate per skill tier."""
    df = _order_tiers(con.execute(TIER_SUMMARY_SQL).df()).sort_values("tier", ignore_index=True)
    df["share"] = df["games"] / df["games"].sum()
    return _add_rate_and_ci(df)


def color_pair_winrates_by_tier(con: duckdb.DuckDBPyConnection, min_games: int = 0) -> pd.DataFrame:
    """Win rate with 95% CI per (color pair, tier); keeps rows with at least min_games games."""
    df = _order_tiers(con.execute(COLOR_PAIR_TIER_SQL).df())
    df = df[df["games"] >= min_games].sort_values(["color_pair", "tier"], ignore_index=True)
    return _add_rate_and_ci(df)


def card_winrates_by_tier(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """GIH win rate with 95% CI per (card, tier)."""
    df = _order_tiers(con.execute(CARD_TIER_SQL).df()).sort_values(["card", "tier"], ignore_index=True)
    return _add_rate_and_ci(df)


def skill_adjusted_card_winrates(con: duckdb.DuckDBPyConnection, min_games: int = 0) -> pd.DataFrame:
    """Raw vs skill-adjusted GIH win rate per card.

    Only cards with at least min_games GIH games in EVERY tier are kept; otherwise the
    adjusted rate would rest on a tiny or missing tier.
    Columns: games, raw_wr (+ Wilson CI), high_share (share of the card's GIH games
    played by the high tier), adjusted_wr, bias = raw_wr - adjusted_wr.
    """
    weights = tier_summary(con).set_index("tier")["share"]
    by_tier = card_winrates_by_tier(con)

    games = by_tier.pivot(index="card", columns="tier", values="games").reindex(columns=TIERS)
    wins = by_tier.pivot(index="card", columns="tier", values="wins").reindex(columns=TIERS)
    rates = by_tier.pivot(index="card", columns="tier", values="win_rate").reindex(columns=TIERS)
    keep = (games.fillna(0) >= min_games).all(axis=1) & games.notna().all(axis=1)

    df = pd.DataFrame({"games": games.sum(axis=1), "wins": wins.sum(axis=1)})
    df["raw_wr"] = df["wins"] / df["games"]
    df["ci_low"], df["ci_high"] = wilson_ci(df["wins"], df["games"])
    df["high_share"] = games["high"].fillna(0) / df["games"]
    df["adjusted_wr"] = (rates * weights.reindex(TIERS)).sum(axis=1, min_count=len(TIERS))
    df["bias"] = df["raw_wr"] - df["adjusted_wr"]
    df = df[keep].astype({"games": int, "wins": int})
    return df.reset_index().rename_axis(columns=None).sort_values("bias", ascending=False, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--min-games", type=int, default=200, help="minimum GIH games per tier for a card")
    args = parser.parse_args()

    con = duckdb.connect(str(args.db), read_only=True)
    pd.set_option("display.width", 160)
    fmt = "{:.3f}".format

    print("Skill tiers:")
    print(tier_summary(con).to_string(index=False, float_format=fmt))

    pairs = color_pair_winrates_by_tier(con)
    print("\nColor pair win rate by tier:")
    print(pairs.pivot(index="color_pair", columns="tier", values="win_rate").to_string(float_format=fmt))

    cards = skill_adjusted_card_winrates(con, args.min_games)
    cols = ["card", "games", "raw_wr", "ci_low", "ci_high", "high_share", "adjusted_wr", "bias"]
    print(f"\nCards with >= {args.min_games} GIH games in every tier: {len(cards)}")
    print("Most flattered by skilled players (raw > adjusted):")
    print(cards[cols].head(10).to_string(index=False, float_format=fmt))
    print("\nMost penalized by weaker players (raw < adjusted):")
    print(cards[cols].tail(10).iloc[::-1].to_string(index=False, float_format=fmt))
    con.close()


if __name__ == "__main__":
    main()
