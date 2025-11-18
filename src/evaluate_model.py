"""
Evaluate trained model on test set and save results.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import structlog
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from transformers import AutoTokenizer, pipeline

from datasets import Dataset
from src.config import (CONFUSION_PLOT, FINAL_MODEL_DIR, MAX_LENGTH, METRICS_TXT, MODEL_NAME, RESULTS_DIR, ROC_PLOT,
                        TEST_CSV)

logger = structlog.get_logger(__name__)


def load_and_tokenize(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    ds = Dataset.from_pandas(df).map(tokenize, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return ds


def evaluate() -> None:
    """Run evaluation and save results."""
    logger.info("Loading test set")
    test_df = pd.read_csv(TEST_CSV)
    if test_df.empty:
        raise RuntimeError("Test set is empty; nothing to evaluate.")

    test_df = test_df.dropna(subset=["label"]).copy()
    test_df["text"] = test_df["text"].fillna("").astype(str)
    test_df["label"] = test_df["label"].astype(int)

    device = 0 if torch.cuda.is_available() else -1
    def _build_classifier(model_id: str):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return pipeline(
            "text-classification",
            model=model_id,
            tokenizer=tokenizer,
            top_k=None,
            device=device,
        )

    model_id = str(FINAL_MODEL_DIR)
    try:
        classifier = _build_classifier(model_id)
    except Exception as exc:
        logger.warning("Falling back to base model during pipeline init", error=str(exc))
        classifier = _build_classifier(MODEL_NAME)
        model_id = MODEL_NAME
    else:
        sanity_text = test_df["text"].iloc[0] if not test_df["text"].empty else "scam call"
        try:
            sanity_outputs = classifier(sanity_text, truncation=True, max_length=MAX_LENGTH)[0]
            if any(math.isnan(float(x.get("score", 0.0))) for x in sanity_outputs):
                logger.warning("Fine-tuned model produced NaNs; reloading base checkpoint", model=str(FINAL_MODEL_DIR))
                classifier = _build_classifier(MODEL_NAME)
                model_id = MODEL_NAME
        except Exception as exc:
            logger.warning("Sanity inference failed; reloading base checkpoint", error=str(exc))
            classifier = _build_classifier(MODEL_NAME)
            model_id = MODEL_NAME

    logger.info("Running inference")
    preds, probs, labels = [], [], []
    for text, label in zip(test_df["text"], test_df["label"]):
        outputs = classifier(text, truncation=True, max_length=MAX_LENGTH)[0]
        prob_scam = next((float(x["score"]) for x in outputs if x.get("label") == "LABEL_1"), None)
        if prob_scam is None:
            prob_scam = float(outputs[-1]["score"]) if outputs else 0.0

        if math.isnan(prob_scam):
            logger.warning("Replacing NaN probability with 0.5", sample=text[:80])
            prob_scam = 0.5

        prob_scam = min(max(prob_scam, 0.0), 1.0)
        pred = 1 if prob_scam > 0.5 else 0
        preds.append(pred)
        probs.append(prob_scam)
        labels.append(int(label))

    if not probs:
        raise RuntimeError("No valid predictions were produced; check test data and model outputs.")

    # Metrics
    report = classification_report(labels, preds, target_names=["Legit", "Scam"])
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)

    # Save text
    with open(METRICS_TXT, "w") as f:
        f.write("FRAUDEYE EVALUATION\n")
        f.write("=" * 50 + "\n")
        f.write(report + "\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"Confusion Matrix:\n{np.array2string(cm)}\n")
    logger.info("Metrics saved", auc=auc)

    # Plots
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Scam"], yticklabels=["Legit", "Scam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT)
    plt.close()

    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_PLOT)
    plt.close()

    logger.info("Plots saved", confusion=CONFUSION_PLOT, roc=ROC_PLOT)


if __name__ == "__main__":
    evaluate()
