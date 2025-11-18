# src/merge_datasets.py
"""
Merge public scam-call datasets from Hugging Face and Kaggle.
Saves balanced train/val/test splits to datasets/.
"""
from pathlib import Path
from typing import List

import kagglehub
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from src.config import HF_DATASETS, KAGGLE_DATASETS, MERGED_CSV, TEST_CSV, TRAIN_CSV, VAL_CSV

logger = structlog.get_logger()


def load_hf_dataset(name: str) -> pd.DataFrame:
    """Load and clean a Hugging Face dataset."""
    try:
        ds = load_dataset(name, split="train")
        df = pd.DataFrame(ds)

        # Text column
        text_col = next(
            (c for c in ["dialogue", "conversation", "text", "content"] if c in df.columns),
            None,
        )
        if not text_col:
            logger.warning("No text column", dataset=name)
            return pd.DataFrame()

        # Label column
        label_col = next((c for c in ["label", "is_toxic"] if c in df.columns), None)
        if label_col is None:
            logger.warning("No label column", dataset=name)
            return pd.DataFrame()

        df["text"] = df[text_col].astype(str).fillna("")
        df["label"] = df[label_col].astype(int)

        # Keep ALL text (even short) – scam scripts are often brief
        df = df[["text", "label"]].copy()
        logger.info("Loaded HF dataset", name=name, size=len(df))
        return df
    except Exception as e:
        logger.error("HF load failed", name=name, error=str(e))
        return pd.DataFrame()


def _load_kumarperiya() -> pd.DataFrame:
    """Load kumarperiya with correct column names: transaction_time → text, fraud → label."""
    try:
        path = Path(kagglehub.dataset_download("kumarperiya/comprehensive-indian-online-fraud-dataset"))
        csvs = list(path.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError("No CSV found")

        df = pd.read_csv(csvs[0], encoding="utf-8", on_bad_lines="skip")

        # --- Text column ---
        if "transaction_time" in df.columns:
            text_col = "transaction_time"
        else:
            obj_cols = df.select_dtypes(include=["object"]).columns
            if len(obj_cols) == 0:
                raise ValueError("No object column for text")
            text_col = obj_cols[0]
            logger.warning("Using fallback text column", fallback=text_col)

        # --- Label column ---
        if "fraud" in df.columns:
            label_col = "fraud"
        else:
            # Fallback: any numeric column with 0/1 values
            num_cols = df.select_dtypes(include=["number"]).columns
            for col in num_cols:
                if df[col].dropna().isin([0, 1]).all():
                    label_col = col
                    logger.warning("Using fallback label column", fallback=label_col)
                    break
            else:
                raise ValueError("No fraud or binary label column")

        df_out = pd.DataFrame({"text": df[text_col].astype(str).fillna(""), "label": df[label_col].astype(int)})

        logger.info(
            "Loaded special Kaggle dataset",
            handle="kumarperiya/comprehensive-indian-online-fraud-dataset",
            file=csvs[0].name,
            size=len(df_out),
            scam_count=df_out["label"].sum(),
            legit_count=len(df_out) - df_out["label"].sum(),
        )
        return df_out
    except Exception as e:
        logger.error("kumarperiya load failed", error=str(e))
        return pd.DataFrame()


def merge_and_split() -> None:
    """Download → merge → balance → split → save."""
    logger.info("Starting dataset merge")
    data_list: List[pd.DataFrame] = []

    # ----- Hugging Face -------------------------------------------------
    for name in HF_DATASETS:
        df = load_hf_dataset(name)
        if not df.empty:
            data_list.append(df)

    # ----- Special: kumarperiya -----------------------------------------
    df_k = _load_kumarperiya()
    if not df_k.empty:
        data_list.append(df_k)

    if not data_list:
        raise RuntimeError("No usable datasets loaded")

    # ----- Merge & dedupe ------------------------------------------------
    merged = pd.concat(data_list, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["text"])
    after = len(merged)
    logger.info(
        "Merged raw data",
        total_samples=after,
        removed_duplicates=before - after,
        scam_count=merged["label"].sum(),
        legit_count=after - merged["label"].sum(),
    )

    # ----- Balance -------------------------------------------------------
    min_count = min(merged["label"].value_counts())
    balanced = pd.concat(
        [
            merged[merged["label"] == 0].sample(min_count, random_state=42),
            merged[merged["label"] == 1].sample(min_count, random_state=42),
        ]
    )
    logger.info("Balanced dataset", size=len(balanced), per_class=min_count)

    # ----- Split ---------------------------------------------------------
    train_df, temp = train_test_split(balanced, test_size=0.3, stratify=balanced["label"], random_state=42)
    val_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

    # ----- Save ----------------------------------------------------------
    balanced.to_csv(MERGED_CSV, index=False)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    logger.info(
        "Datasets saved",
        train=len(train_df),
        val=len(val_df),
        test=len(test_df),
    )


if __name__ == "__main__":
    merge_and_split()
