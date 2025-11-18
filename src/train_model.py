"""
Train DistilBERT on merged dataset and export TFLite + vocab.
"""

import os

import pandas as pd
import structlog
import tensorflow as tf
import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, TFAutoModelForSequenceClassification,
                          Trainer, TrainingArguments)

from datasets import Dataset
from src.config import (BATCH_SIZE, CHECKPOINTS_DIR, EPOCHS, FINAL_MODEL_DIR, FP16, LEARNING_RATE, LOGS_DIR, MAX_LENGTH,
                        MODEL_NAME, TF_MODEL_DIR, TFLITE_MODEL, TRAIN_CSV, VAL_CSV, VOCAB_TXT, WEIGHT_DECAY)

logger = structlog.get_logger(__name__)


def load_and_tokenize(csv_path: str):
    """Load CSV and tokenize → return (dataset, tokenizer)."""
    df = pd.read_csv(csv_path)
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
    train_ds, tokenizer = load_and_tokenize(TRAIN_CSV)
    val_ds, _ = load_and_tokenize(VAL_CSV)

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
        metric_for_best_model="eval_loss",
        fp16=FP16,
        logging_dir=str(LOGS_DIR),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    logger.info("Starting training")
    trainer.train()

    # Save final PyTorch model + tokenizer
    final_dir = str(FINAL_MODEL_DIR)
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("PyTorch model + tokenizer saved", path=final_dir)

    # Convert to TensorFlow SavedModel
    logger.info("Converting to TensorFlow SavedModel")
    tf_model = TFAutoModelForSequenceClassification.from_pretrained(final_dir, from_pt=True)

    # Create a concrete function for inference
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, MAX_LENGTH], dtype=tf.int32, name="input_ids"),
            tf.TensorSpec(shape=[None, MAX_LENGTH], dtype=tf.int32, name="attention_mask"),
        ]
    )
    def serve(input_ids, attention_mask):
        return tf_model(input_ids=input_ids, attention_mask=attention_mask, training=False)

    # Export SavedModel (without debug info → faster TFLite conversion)
    tf_saved_dir = str(TF_MODEL_DIR)
    os.makedirs(tf_saved_dir, exist_ok=True)

    # ← FIXED: Use save_debug_info (TF 2.16+); fallback for older TF
    save_options = None
    if hasattr(tf.saved_model.SaveOptions, "save_debug_info"):
        save_options = tf.saved_model.SaveOptions(save_debug_info=False)
    tf.saved_model.save(tf_model, tf_saved_dir, signatures={"serving_default": serve}, options=save_options)
    logger.info("TensorFlow SavedModel exported", path=tf_saved_dir)

    # Convert to TFLite with clean, real progress
    logger.info("Converting to TFLite")

    # Suppress TF warnings and logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 2 = filter WARNING and below
    tf.get_logger().setLevel("ERROR")

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.experimental_new_converter = True  # Faster MLIR path

    # Phase-based progress (weights tuned for DistilBERT)
    phases = [
        ("Loading SavedModel", 20),
        ("Optimizing graph", 40),
        ("Quantizing & converting", 30),
        ("Finalizing TFLite", 10),
    ]
    total_weight = sum(w for _, w in phases)
    pbar = tqdm.tqdm(
        total=total_weight,
        desc="TFLite conversion",
        unit="%",
        bar_format="{l_bar}{bar}| {elapsed} | {postfix}",
        colour="cyan",
    )

    # Start: Loading phase
    pbar.set_postfix(phase="Loading SavedModel")
    pbar.update(phases[0][1] * 0.3)  # Partial progress on start

    try:
        tflite_model = converter.convert()

        # Complete remaining phases
        for phase_name, weight in phases[1:]:
            pbar.set_postfix(phase=phase_name)
            pbar.update(weight)

    except Exception as e:
        pbar.close()
        logger.error("TFLite conversion failed", error=str(e))
        raise
    finally:
        pbar.close()

    # Save TFLite model
    with open(TFLITE_MODEL, "wb") as f:
        f.write(tflite_model)
    logger.info("TFLite model exported", path=TFLITE_MODEL)

    # Save vocab
    with open(VOCAB_TXT, "w") as f:
        for token in tokenizer.get_vocab().keys():
            f.write(token + "\n")
    logger.info("Vocab saved", path=VOCAB_TXT)


if __name__ == "__main__":
    train_model()
