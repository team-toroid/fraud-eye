"""
Train DistilBERT on merged dataset and export TFLite + vocab.
"""

import os
from typing import Dict, Tuple

import pandas as pd
import structlog
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from datasets import Dataset
from src.config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FINAL_MODEL_DIR,
    FP16,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGS_DIR,
    MAX_LENGTH,
    MERGED_CSV,
    MODEL_NAME,
    SYNTHETIC_CSV,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)

logger = structlog.get_logger(__name__)


def compute_metrics(pred) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and F1 score."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def load_and_tokenize(csv_path: str, tokenizer=None) -> Tuple[Dataset, AutoTokenizer]:
    """Load CSV and tokenize → return (dataset, tokenizer)."""
    df = pd.read_csv(csv_path)

    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    ds = Dataset.from_pandas(df).map(tokenize, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return ds, tokenizer


def train_model() -> None:
    """Train DistilBERT → export PyTorch + TFLite + vocab."""
    logger.info("Loading datasets")

    # Load merged dataset
    merged_df = pd.read_csv(MERGED_CSV)

    # Load synthetic dataset and split: 20% for train/test, 80% for validation
    synthetic_df = pd.read_csv(SYNTHETIC_CSV)

    # Split synthetic: 20% for training, 80% for validation
    synthetic_train_portion, synthetic_val = train_test_split(
        synthetic_df, test_size=0.99, stratify=synthetic_df["label"], random_state=27  # 80% for validation
    )

    # Merge the 20% synthetic with fraudeye_merged
    combined_df = pd.concat([merged_df, synthetic_train_portion], ignore_index=True)

    # Split the combined dataset 80:20 for train:test
    train_df, test_df = train_test_split(
        combined_df, test_size=0.2, stratify=combined_df["label"], random_state=16  # 80% train, 20% test
    )

    # Use remaining 80% of synthetic for validation
    val_df = synthetic_val

    logger.info(
        "Dataset split completed",
        total_merged=len(merged_df),
        synthetic_train_portion=len(synthetic_train_portion),
        synthetic_val=len(synthetic_val),
        combined_before_split=len(combined_df),
        final_train=len(train_df),
        final_test=len(test_df),
        final_val=len(val_df),
        train_scam_ratio=f"{train_df['label'].sum() / len(train_df):.2%}",
        test_scam_ratio=f"{test_df['label'].sum() / len(test_df):.2%}",
        val_scam_ratio=f"{val_df['label'].sum() / len(val_df):.2%}",
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize all datasets
    def tokenize_df(df):
        ds = Dataset.from_pandas(df).map(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
            ),
            batched=True,
        )
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return ds

    train_ds = tokenize_df(train_df)
    test_ds = tokenize_df(test_df)
    val_ds = tokenize_df(val_df)

    logger.info("Datasets tokenized successfully")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Training args
    args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Changed to F1 for better model selection
        greater_is_better=True,  # Higher F1 is better
        fp16=FP16,
        logging_dir=str(LOGS_DIR),
        logging_steps=50,
        report_to=[],
        warmup_ratio=WARMUP_RATIO,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        save_total_limit=3,  # Keep only best 3 checkpoints
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    logger.info("Starting training with optimized configuration")
    trainer.train()

    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_results = trainer.evaluate(test_ds)
    logger.info("Test set results", **test_results)

    # Save final PyTorch model + tokenizer
    final_dir = str(FINAL_MODEL_DIR)
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("PyTorch model + tokenizer saved", path=final_dir)


if __name__ == "__main__":
    train_model()
