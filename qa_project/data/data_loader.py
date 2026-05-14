"""
data/data_loader.py
─────────────────────────────────────────────────────────────
Loads the train and test CSV datasets, validates the schema,
and exposes clean DataFrame objects to the rest of the project.
"""

import pandas as pd
from pathlib import Path
from config import TRAIN_CSV, TEST_CSV
from utils import log_info, log_ok, log_warn, log_error

# Expected columns in both CSVs
REQUIRED_COLUMNS = [
    "question_id", "source_dataset", "question", "ground_truth_answer",
    "closed_book_answer", "open_book_answer", "closed_book_correct",
    "open_book_correct", "supporting_facts_retrieved", "error_type",
    "error_subtype", "cot_closed_correct", "cot_open_correct",
    "confidence_score_closed", "confidence_score_open",
    "num_hops_required", "domain",
]

BINARY_COLS = [
    "closed_book_correct", "open_book_correct",
    "supporting_facts_retrieved", "cot_closed_correct", "cot_open_correct",
]

FLOAT_COLS = ["confidence_score_closed", "confidence_score_open"]


def _validate(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Check schema, coerce types, warn on issues."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        log_error(f"Missing columns in {path.name}: {missing}")
        raise ValueError(f"Schema mismatch in {path.name}")

    for col in BINARY_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["num_hops_required"] = pd.to_numeric(
        df["num_hops_required"], errors="coerce"
    ).fillna(1).astype(int)

    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if not null_cols.empty:
        log_warn(f"Null values detected in {path.name}:\n{null_cols}")

    return df


def load_train() -> pd.DataFrame:
    """Load and validate the training / main evaluation dataset (100 rows)."""
    log_info(f"Loading training dataset from {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    df = _validate(df, TRAIN_CSV)
    log_ok(f"Training dataset loaded: {len(df)} questions")
    return df


def load_test() -> pd.DataFrame:
    """Load and validate the held-out test dataset (50 rows)."""
    log_info(f"Loading test dataset from {TEST_CSV}")
    df = pd.read_csv(TEST_CSV)
    df = _validate(df, TEST_CSV)
    log_ok(f"Test dataset loaded: {len(df)} questions")
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper — returns (train_df, test_df)."""
    return load_train(), load_test()


def get_subset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Filter by source_dataset name.
    dataset: 'Natural Questions' | 'TriviaQA' | 'HotpotQA'
    """
    subset = df[df["source_dataset"] == dataset].copy()
    if subset.empty:
        log_warn(f"No rows found for dataset='{dataset}'")
    return subset


def get_error_subset(df: pd.DataFrame, error_type: str) -> pd.DataFrame:
    """
    Filter by error_type: 'knowledge' | 'reasoning' | 'none'
    """
    return df[df["error_type"] == error_type].copy()


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary table of question counts by source and error type."""
    summary = (
        df.groupby(["source_dataset", "error_type"])
          .size()
          .reset_index(name="count")
          .pivot(index="source_dataset", columns="error_type", values="count")
          .fillna(0)
          .astype(int)
    )
    summary["total"] = summary.sum(axis=1)
    return summary
