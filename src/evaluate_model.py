"""
Evaluate trained model on test set and save results.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import structlog
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, pipeline

from datasets import Dataset
from src.config import (
    CONFUSION_PLOT,
    EDGE_CASE_CSV,
    FINAL_MODEL_DIR,
    MAX_LENGTH,
    MERGED_CSV,
    METRICS_TXT,
    MODEL_NAME,
    RESULTS_DIR,
    ROC_PLOT,
    SYNTHETIC_CSV,
)

logger = structlog.get_logger(__name__)


def load_and_tokenize(csv_path: str) -> Dataset:
    """
    Load a CSV file and tokenize the text column.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the dataset.

    Returns
    -------
    Dataset
        A Hugging Face Dataset object with tokenized text.
    """
    df = pd.read_csv(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    ds = Dataset.from_pandas(df).map(tokenize, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return ds


def run_evaluation(classifier, df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    """
    Run evaluation on a specific dataframe and save results.

    Parameters
    ----------
    classifier : pipeline
        The text classification pipeline.
    df : pd.DataFrame
        The dataframe containing the data to evaluate.
    dataset_name : str
        Name of the dataset being evaluated (e.g., "Validation Set").
    output_dir : Path
        Directory where the evaluation results will be saved.
    """
    logger.info(f"Running inference on {dataset_name}")

    if df.empty:
        logger.warning(f"{dataset_name} is empty; skipping evaluation.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.dropna(subset=["label"]).copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    preds, probs, labels = [], [], []
    for text, label in zip(df["text"], df["label"]):
        outputs = classifier(text, truncation=True, max_length=MAX_LENGTH)[0]
        prob_scam = next((float(x["score"]) for x in outputs if x.get("label") == "LABEL_1"), None)
        if prob_scam is None:
            prob_scam = float(outputs[-1]["score"]) if outputs else 0.0

        if math.isnan(prob_scam):
            prob_scam = 0.5

        prob_scam = min(max(prob_scam, 0.0), 1.0)
        pred = 1 if prob_scam > 0.5 else 0
        preds.append(pred)
        probs.append(prob_scam)
        labels.append(int(label))

    if not probs:
        logger.error(f"No valid predictions for {dataset_name}")
        return

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=["Legit", "Scam"], zero_division=0)

    metrics_path = output_dir / "metrics.txt"
    cm_plot_path = output_dir / "confusion_matrix.png"
    roc_plot_path = output_dir / "roc_curve.png"
    pr_plot_path = output_dir / "pr_curve.png"
    dist_plot_path = output_dir / "prob_dist.png"

    with open(metrics_path, "w") as f:
        f.write(f"FRAUDEYE EVALUATION ({dataset_name.upper()})\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"ROC-AUC:   {auc:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(report + "\n")
        f.write(f"Confusion Matrix:\n{np.array2string(cm)}\n")

    logger.info(f"Metrics saved for {dataset_name}", accuracy=acc, f1=f1, auc=auc, path=str(metrics_path))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Scam"], yticklabels=["Legit", "Scam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({dataset_name})")
    plt.tight_layout()
    plt.savefig(cm_plot_path)
    plt.close()

    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({dataset_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(roc_plot_path)
        plt.close()
    except Exception:
        logger.warning(f"Could not generate ROC curve for {dataset_name} (possibly single class)")

    try:
        precision, recall, _ = precision_recall_curve(labels, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"F1 = {f1:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve ({dataset_name})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pr_plot_path)
        plt.close()
    except Exception:
        logger.warning(f"Could not generate PR curve for {dataset_name}")

    plt.figure(figsize=(8, 6))
    sns.histplot(
        data={"Probability": probs, "Label": ["Scam" if l == 1 else "Legit" for l in labels]},
        x="Probability",
        hue="Label",
        bins=30,
        kde=True,
        element="step",
    )
    plt.title(f"Prediction Probability Distribution ({dataset_name})")
    plt.xlabel("Predicted Probability of Scam")
    plt.tight_layout()
    plt.savefig(dist_plot_path)
    plt.close()

    logger.info(f"Plots saved for {dataset_name}", dir=str(output_dir))


def evaluate() -> None:
    """
    Run evaluation and save results.

    This function loads the datasets, splits them according to the training logic,
    initializes the classifier, and runs evaluation on both the validation set
    and the edge case dataset.
    """
    logger.info("Loading datasets")

    merged_df = pd.read_csv(MERGED_CSV)
    synthetic_df = pd.read_csv(SYNTHETIC_CSV)

    synthetic_train_portion, synthetic_val = train_test_split(
        synthetic_df, test_size=0.99, stratify=synthetic_df["label"], random_state=27
    )

    combined_df = pd.concat([merged_df, synthetic_train_portion], ignore_index=True)

    train_df, test_df = train_test_split(
        combined_df, test_size=0.2, stratify=combined_df["label"], random_state=16
    )

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

    val_results_dir = RESULTS_DIR / "validation"
    run_evaluation(classifier, val_df, "Validation Set", output_dir=val_results_dir)

    if EDGE_CASE_CSV.exists():
        logger.info("Loading edge case dataset")
        edge_df = pd.read_csv(EDGE_CASE_CSV)
        edge_results_dir = RESULTS_DIR / "edge_cases"
        run_evaluation(classifier, edge_df, "Edge Cases", output_dir=edge_results_dir)
    else:
        logger.warning("Edge case dataset not found", path=str(EDGE_CASE_CSV))


if __name__ == "__main__":
    evaluate()
