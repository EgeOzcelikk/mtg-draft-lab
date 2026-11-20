# mtg-draft-lab

> **Data from [17Lands](https://www.17lands.com)** (public datasets, CC BY 4.0). This project is not affiliated with or endorsed by 17Lands.

## Question
In MTG draft, which cards and color pairs really win, and where are the popular 17Lands win rates misleading (for example player skill bias, or different archetypes of the same colors grouped together)?

## Data source
_TODO: set name, files used, download date, license link._

## Approach
_TODO_

## Results
_TODO: three findings with a chart._

## Limitations
_TODO_

## How to run
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# put game_data_public.SOS.PremierDraft.csv.gz from the 17Lands public datasets page into data/raw/
python -m src.load                       # builds data/processed/sos.duckdb (games, card_games)
python -m src.winrates --min-games 500   # color pair and card win rates
python -m pytest -q                      # tests on a small synthetic fixture
```

## Tech stack
Python, DuckDB (SQL), pandas, scikit-learn, matplotlib, Streamlit, pytest.
