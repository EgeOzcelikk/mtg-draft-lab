# mtg-draft-lab

> **Data from [17Lands](https://www.17lands.com)** (public datasets, CC BY 4.0). This project is not affiliated with or endorsed by 17Lands.

## Question
In MTG draft, which cards and color pairs really win, and where are the popular 17Lands win rates misleading (for example player skill bias, or different archetypes of the same colors grouped together)?

## Data source
_TODO: set name, files used, download date, license link._

## Approach
1. Load the dump into DuckDB and compute win rates per two-color pair and per card (GP, GIH, OH, GD).
2. Split players into three skill tiers by their overall 17Lands game win rate (`user_game_win_rate_bucket`): low < 0.52 <= mid < 0.58 <= high, roughly a third of the games each. Compare card and color pair win rates per tier, with 95% Wilson confidence intervals.
3. Skill-adjusted card win rate (direct standardization): weight a card's per-tier GIH win rates by the tier mix of all games, so a card is not rewarded for being picked mostly by strong players.

## Results
_Work in progress; final findings and chart come in week 4._

Early numbers (SOS Premier Draft, 317 cards with >= 200 GIH games in every tier):
- Skill selection is strong: across cards, the share of a card's games played by the high tier correlates with its raw GIH win rate at r = 0.80.
- Adjusting for skill moves card win rates by -3.5 to +1.4 percentage points. For 104 of 317 cards the shift is larger than half the width of the card's own 95% CI. The median card moves 9 places in the ranking, the largest move is 63 places.
- Color pairs depend on skill very differently: BR wins 33.0% for low-tier and 63.0% for high-tier players, RG 49.5% vs 58.5%.

## Limitations
- The skill bucket is computed from the player's games, including the games analysed here. It is usable for grouping, but not as a model feature (target leakage).
- 131 games have no skill bucket and are all wins; they are excluded from the skill analysis.
- Standardization only removes the observed skill difference. Other confounders (deck quality, draft position, archetype within the same colors) remain.

## How to run
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# put game_data_public.SOS.PremierDraft.csv.gz from the 17Lands public datasets page into data/raw/
python -m src.load                       # builds data/processed/sos.duckdb (games, card_games)
python -m src.winrates --min-games 500   # color pair and card win rates
python -m src.skill --min-games 200      # win rates by skill tier, skill-adjusted card win rates
python -m pytest -q                      # tests on a small synthetic fixture
```

## Tech stack
Python, DuckDB (SQL), pandas, scikit-learn, matplotlib, Streamlit, pytest.
