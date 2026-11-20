# CLAUDE.md

## Project question
In MTG draft, which cards and color pairs really win, and where are the popular 17Lands win rates misleading (for example player skill bias, or different archetypes of the same colors grouped together)?

## Role split
This is project 1: you build, I make one small change per milestone.

## 4-week plan
- Week 1: load the data into DuckDB; SQL win rates by card and color pair.
- Week 2: split results by player skill; show selection bias; simple confidence intervals.
- Week 3: logistic regression that predicts a game win from deck features (colors, mana curve), compared with a baseline; report AUC and precision/recall.
- Week 4: Streamlit dashboard and a README with 3 findings.

## Data rules (17Lands usage guidelines: https://www.17lands.com/usage_guidelines)
- Use only the downloadable dumps from the 17Lands Public Datasets page (generally CC BY 4.0). Never scrape the 17Lands API.
- Put "Data from 17Lands" with a link at the top of the README and the dashboard, clearly visible, not in a footnote.
- Don't imply that 17Lands endorses the project.
- Pick a set released at least ~3 weeks ago, because game data is published about 3 weeks into a set.
- Raw dumps stay untouched in data/raw/ (git-ignored); everything derived is rebuilt by scripts.
