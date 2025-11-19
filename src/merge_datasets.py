"""
Merge public scam-call datasets from Hugging Face and Kaggle.
Saves balanced train/val/test splits to datasets/.
"""

from pathlib import Path
from typing import List

import kagglehub
import pandas as pd
import structlog
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from datasets import load_dataset
from src.config import HF_DATASETS, MERGED_CSV

logger = structlog.get_logger()
_marian_resources = None


def _get_ko_en_translator():
    """Lazy-load a local MarianMT ko→en translator model/tokenizer."""
    global _marian_resources
    if _marian_resources is None:
        model_name = "Helsinki-NLP/opus-mt-ko-en"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _marian_resources = (tokenizer, model, device)
    return _marian_resources


def _translate_korean_texts(texts: List[str]) -> List[str]:
    """Translate a list of Korean strings to English locally."""
    if not texts:
        return []

    tokenizer, model, device = _get_ko_en_translator()
    translated: List[str] = []
    batch_size = 32

    for start in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[start : start + batch_size]]
        try:
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_length=512,
                    num_beams=1,  # Greedy decoding for speed
                    early_stopping=True,
                )
            outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
            translated.extend(outputs)

            if (start // batch_size) % 10 == 0:
                logger.info("Translation progress", completed=start + len(batch), total=len(texts))

        except Exception as e:
            logger.warning(
                "Batch translation failed, falling back to original text",
                error=str(e),
                batch_start=start,
            )
            translated.extend(batch)

    return translated


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

        # Label column - check for various naming conventions
        label_col = next(
            (c for c in ["label", "labels", "is_toxic", "is_spam", "is_scam"] if c in df.columns),
            None,
        )
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
        # --- Text column ---
        # Prioritize description/text over transaction_time (which contains garbage)
        if "description" in df.columns:
            text_col = "description"
        elif "text" in df.columns:
            text_col = "text"
        elif "transaction_time" in df.columns:
            # This column often contains timestamps, so we avoid it if possible
            # But if it's the only option, we might have to use it (unlikely for this dataset)
            logger.warning("Using transaction_time as text column (risk of garbage)")
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

        # Drop NaNs in label column
        df = df.dropna(subset=[label_col])
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


def _load_robocall_audio() -> pd.DataFrame:
    """Load Robocall Audio Dataset from local CSV file."""
    try:
        # Load from local metadata.csv
        csv_path = Path("data/robocall-audio-dataset/metadata.csv")
        if not csv_path.exists():
            logger.warning("Robocall metadata.csv not found", path=str(csv_path))
            return pd.DataFrame()

        df = pd.read_csv(csv_path)

        # Dataset has 'transcript' column for text
        if "transcript" not in df.columns:
            logger.warning("No transcript column in robocall dataset", columns=df.columns.tolist())
            return pd.DataFrame()

        # All robocalls are scams (label=1)
        df_out = pd.DataFrame({"text": df["transcript"].astype(str).fillna(""), "label": 1})  # All robocalls are scams

        # Filter out very short transcriptions
        df_out = df_out[df_out["text"].str.len() > 10]

        logger.info("Loaded Robocall Audio dataset", size=len(df_out))
        return df_out
    except Exception as e:
        logger.error("Robocall Audio load failed", error=str(e))
        return pd.DataFrame()


def _load_sms_spam() -> pd.DataFrame:
    """Load SMS Spam Collection from HuggingFace."""
    try:
        # Load from HuggingFace
        ds = load_dataset("ucirvine/sms_spam", split="train")
        df = pd.DataFrame(ds)

        # Dataset has 'sms' for text and 'label' for spam/ham
        text_col = next((c for c in ["sms", "message", "text"] if c in df.columns), None)
        label_col = next((c for c in ["label", "spam"] if c in df.columns), None)

        if not text_col or not label_col:
            logger.warning("Missing columns in SMS spam dataset", columns=df.columns.tolist())
            return pd.DataFrame()

        # Convert ham/spam to 0/1
        df_out = pd.DataFrame({"text": df[text_col].astype(str).fillna("")})

        # Handle different label formats
        if df[label_col].dtype == "object":
            # Map 'ham'/'spam' to 0/1
            df_out["label"] = df[label_col].map({"ham": 0, "spam": 1}).fillna(0).astype(int)
        else:
            df_out["label"] = df[label_col].astype(int)

        logger.info("Loaded SMS Spam dataset", size=len(df_out), spam_count=df_out["label"].sum())
        return df_out
    except Exception as e:
        logger.error("SMS Spam load failed", error=str(e))
        return pd.DataFrame()


def _load_korean_vishing() -> pd.DataFrame:
    """Load Korean Voice Phishing dataset from local CSV and translate to English."""
    try:
        csv_path = Path("data/Korean_Voice_Phishing_Detection/Data_Collection_Preprocessing/KorCCVi_v2.csv")
        if not csv_path.exists():
            logger.warning("Korean vishing CSV not found", path=str(csv_path))
            return pd.DataFrame()

        # Read CSV - has header: id,transcript,confidence,label
        df = pd.read_csv(csv_path)

        if "transcript" not in df.columns or "label" not in df.columns:
            logger.warning("Missing required columns in Korean dataset", columns=df.columns.tolist())
            return pd.DataFrame()

        # Filter out very short text first
        df = df[df["transcript"].astype(str).str.len() > 10].copy()

        logger.info("Translating Korean text to English", total_samples=len(df))
        translated_texts = _translate_korean_texts(df["transcript"].tolist())

        # Create output dataframe with translated text
        df_out = pd.DataFrame({"text": translated_texts, "label": df["label"].astype(int).tolist()})

        logger.info(
            "Loaded and translated Korean Vishing dataset", size=len(df_out), vishing_count=df_out["label"].sum()
        )
        return df_out
    except Exception as e:
        logger.error("Korean Vishing load failed", error=str(e))
        return pd.DataFrame()


def _load_phishing_voice_dataset() -> pd.DataFrame:
    """Load the transcribed PhishingVoiceDataset."""
    try:
        csv_path = Path("data/PhishingVoiceDataset/dataset.csv")
        if not csv_path.exists():
            logger.warning("PhishingVoiceDataset CSV not found", path=str(csv_path))
            return pd.DataFrame()

        df = pd.read_csv(csv_path)

        # Ensure columns exist
        if "text" not in df.columns or "label" not in df.columns:
            logger.warning("Missing columns in PhishingVoiceDataset", columns=df.columns.tolist())
            return pd.DataFrame()

        df_out = df[["text", "label"]].copy()
        logger.info("Loaded PhishingVoiceDataset", size=len(df_out), phishing_count=df_out["label"].sum())
        return df_out
    except Exception as e:
        logger.error("PhishingVoiceDataset load failed", error=str(e))
        return pd.DataFrame()


def merge_and_split() -> None:
    """Download → merge → balance → save merged dataset only."""
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

    # ----- External: Robocall Audio -------------------------------------
    df_robocall = _load_robocall_audio()
    if not df_robocall.empty:
        data_list.append(df_robocall)

    # ----- External: SMS Spam -------------------------------------------
    df_sms = _load_sms_spam()
    if not df_sms.empty:
        data_list.append(df_sms)

    # ----- External: Korean Vishing -------------------------------------
    df_korean = _load_korean_vishing()
    if not df_korean.empty:
        data_list.append(df_korean)

    # ----- External: Phishing Voice Dataset -----------------------------
    df_phishing_voice = _load_phishing_voice_dataset()
    if not df_phishing_voice.empty:
        data_list.append(df_phishing_voice)

    if not data_list:
        raise RuntimeError("No usable datasets loaded")

    # ----- Merge & dedupe ------------------------------------------------
    merged = pd.concat(data_list, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["text"])

    # ----- Filter Garbage ------------------------------------------------
    # Remove rows that are just timestamps (e.g. "01/03/2024 20:26")
    # Regex for date/time patterns
    timestamp_pattern = r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\s+\d{1,2}:\d{1,2}"
    merged = merged[~merged["text"].str.match(timestamp_pattern, na=False)]

    # Remove very short text (likely noise)
    merged = merged[merged["text"].str.len() > 5]

    # Remove rows with only numbers/symbols
    merged = merged[merged["text"].str.contains(r"[a-zA-Z]", na=False)]

    after = len(merged)
    logger.info(
        "Merged raw data",
        total_samples=after,
        removed_duplicates=before - after,
        scam_count=merged["label"].sum(),
        legit_count=after - merged["label"].sum(),
    )

    # ----- Save merged dataset only --------------------------------------
    merged.to_csv(MERGED_CSV, index=False)

    logger.info(
        "Merged dataset saved",
        path=str(MERGED_CSV),
        total_samples=len(merged),
        scam_count=merged["label"].sum(),
        legit_count=len(merged) - merged["label"].sum(),
    )


if __name__ == "__main__":
    merge_and_split()
