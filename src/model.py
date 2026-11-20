"""Predict a game win from deck features with logistic regression.

Split: by draft_time (70% / 15% / 15% of games). All games of one draft share a
draft_time, so a draft never ends up in two sets, and the test set is "the future".
Train fits the models, validation picks the regularization strength C, test is
used once for the final numbers.

Uncertainty: 95% bootstrap intervals for the test AUCs and for the AUC difference
deck - colors_only. Whole drafts are resampled (cluster bootstrap), because games of
the same draft share a deck and a player and are not independent.

Models:
- prior:       predicts the training win rate for every game (AUC 0.5 by definition)
- colors_only: logistic regression on main_colors
- deck:        main_colors + splash + on_play + lands + mana curve

Usage:
    python -m src.model
"""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CURVE_BUCKETS, DEFAULT_CARDS, build_features
from src.load import DEFAULT_DB

NUMERIC = ["n_lands"] + [name for name, _, _ in CURVE_BUCKETS] + ["avg_mv"]
BINARY = ["has_splash", "on_play"]
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0]


def time_split(df: pd.DataFrame, train: float = 0.70, val: float = 0.15) -> tuple[pd.DataFrame, ...]:
    """Split games by draft_time quantiles; ties (same draft) stay in one set."""
    t1, t2 = df["draft_time"].quantile([train, train + val])
    return (
        df[df["draft_time"] <= t1],
        df[(df["draft_time"] > t1) & (df["draft_time"] <= t2)],
        df[df["draft_time"] > t2],
    )


def colors_encoder() -> OneHotEncoder:
    # Color combinations under 1% of training games share one "infrequent" column.
    return OneHotEncoder(min_frequency=0.01, handle_unknown="infrequent_if_exist")


def make_model(name: str, C: float = 1.0) -> Pipeline:
    if name == "prior":
        return Pipeline([("clf", DummyClassifier(strategy="prior"))])
    if name == "colors_only":
        prep = ColumnTransformer([("colors", colors_encoder(), ["main_colors"])])
    elif name == "deck":
        prep = ColumnTransformer([
            ("colors", colors_encoder(), ["main_colors"]),
            ("num", StandardScaler(), NUMERIC),
            ("bin", "passthrough", BINARY),
        ])
    else:
        raise ValueError(f"unknown model {name!r}")
    return Pipeline([("prep", prep), ("clf", LogisticRegression(C=C, max_iter=1000))])


def evaluate(y: pd.Series, p: np.ndarray, threshold: float = 0.5) -> dict:
    pred = p >= threshold
    return {
        "auc": roc_auc_score(y, p),
        "log_loss": log_loss(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "predicted_win_share": pred.mean(),
    }


def bootstrap_auc(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, groups: np.ndarray,
    n_boot: int = 1000, seed: int = 0,
) -> pd.DataFrame:
    """95% percentile intervals for AUC(a), AUC(b) and AUC(a) - AUC(b).

    Resamples groups (drafts) with replacement and keeps all games of a drawn group.
    Both models are scored on the same resample (paired), so the difference interval
    is narrower than comparing two separate intervals.
    """
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    codes, _ = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(codes.max() + 2))
    rows_of = [order[starts[g]:starts[g + 1]] for g in range(codes.max() + 1)]

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[g] for g in rng.integers(0, len(rows_of), len(rows_of))])
        auc_a, auc_b = roc_auc_score(y[idx], p_a[idx]), roc_auc_score(y[idx], p_b[idx])
        samples.append((auc_a, auc_b, auc_a - auc_b))
    samples = np.array(samples)
    point = roc_auc_score(y, p_a), roc_auc_score(y, p_b)
    return pd.DataFrame({
        "estimate": [point[0], point[1], point[0] - point[1]],
        "ci_low": np.percentile(samples, 2.5, axis=0),
        "ci_high": np.percentile(samples, 97.5, axis=0),
    }, index=["auc_a", "auc_b", "diff"])


def fit_and_select(name: str, train: pd.DataFrame, val: pd.DataFrame) -> tuple[Pipeline, float]:
    """Fit on train; for logistic models pick C by validation AUC."""
    grid = [1.0] if name == "prior" else C_GRID
    best = None
    for C in grid:
        model = make_model(name, C).fit(train, train["won"])
        auc = roc_auc_score(val["won"], model.predict_proba(val)[:, 1])
        if best is None or auc > best[2]:
            best = (model, C, auc)
    return best[0], best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    args = parser.parse_args()

    con = duckdb.connect(str(args.db), read_only=True)
    df = build_features(con, args.cards)
    con.close()

    train, val, test = time_split(df)
    for label, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{label:5} {len(part):>7,} games  {part['draft_time'].min()} .. {part['draft_time'].max()}"
              f"  win rate {part['won'].mean():.3f}")

    rows, models = [], {}
    for name in ["prior", "colors_only", "deck"]:
        model, C = fit_and_select(name, train, val)
        models[name] = model
        rows.append({"model": name, "C": C, **evaluate(test["won"], model.predict_proba(test)[:, 1])})
    print("\nTest set (threshold 0.5):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    boot = bootstrap_auc(
        test["won"], models["deck"].predict_proba(test)[:, 1],
        models["colors_only"].predict_proba(test)[:, 1], test["draft_id"],
    ).rename(index={"auc_a": "deck", "auc_b": "colors_only", "diff": "deck - colors_only"})
    print("\nTest AUC, 95% cluster bootstrap over drafts (1000 resamples):")
    print(boot.round(4).to_string())

    deck = models["deck"]
    coefs = pd.Series(deck.named_steps["clf"].coef_[0], index=deck.named_steps["prep"].get_feature_names_out())
    print("\nDeck model coefficients (numeric features standardized):")
    print(coefs.sort_values().round(3).to_string())

    p = deck.predict_proba(test)[:, 1]
    observed, predicted = calibration_curve(test["won"], p, n_bins=10, strategy="quantile")
    print("\nCalibration of the deck model on test (10 equal-size bins):")
    print(pd.DataFrame({"mean_predicted": predicted, "observed_win_rate": observed}).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
