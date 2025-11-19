"""
Convert trained PyTorch model to TensorFlow, TFLite, and ONNX formats.
"""

import os

import structlog
import tensorflow as tf
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TFAutoModelForSequenceClassification

from src.config import FINAL_MODEL_DIR, MAX_LENGTH, ONNX_MODEL, TF_MODEL_DIR, TFLITE_MODEL, VOCAB_TXT

logger = structlog.get_logger(__name__)


def convert_models() -> None:
    """Load trained PyTorch model and convert to TF, TFLite, and ONNX."""
    final_dir = str(FINAL_MODEL_DIR)
    if not os.path.exists(final_dir):
        logger.error("Final model directory not found. Run training first.", path=final_dir)
        return

    logger.info("Loading PyTorch model and tokenizer", path=final_dir)
    tokenizer = AutoTokenizer.from_pretrained(final_dir)

    # ==========================================
    # 1. Convert to TensorFlow SavedModel
    # ==========================================
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

    # Use save_debug_info (TF 2.16+); fallback for older TF
    save_options = None
    if hasattr(tf.saved_model.SaveOptions, "save_debug_info"):
        save_options = tf.saved_model.SaveOptions(save_debug_info=False)
    tf.saved_model.save(tf_model, tf_saved_dir, signatures={"serving_default": serve}, options=save_options)
    logger.info("TensorFlow SavedModel exported", path=tf_saved_dir)

    # ==========================================
    # 2. Convert to TFLite (primary Android format)
    # ==========================================
    logger.info("Converting to TFLite")
    tflite_success = False

    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        # Enable Select TF Ops and Custom Ops to support Erfc (used in GELU)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.allow_custom_ops = True

        # Try conversion
        tflite_model = converter.convert()

        # Save TFLite model
        with open(TFLITE_MODEL, "wb") as f:
            f.write(tflite_model)
        logger.info("TFLite model exported", path=TFLITE_MODEL, size_mb=len(tflite_model) / 1024 / 1024)
        tflite_success = True
    except Exception as e:
        logger.warning("TFLite conversion failed, will try ONNX as fallback", error=str(e))

    # ==========================================
    # 3. Convert to ONNX (alternative Android format)
    # ==========================================
    logger.info("Converting to ONNX format")
    try:
        # Load PyTorch model for ONNX export
        model = AutoModelForSequenceClassification.from_pretrained(final_dir)
        model.eval()

        # Create dummy input
        dummy_text = "This is a sample text for ONNX export"
        inputs = tokenizer(
            dummy_text, return_tensors="pt", max_length=MAX_LENGTH, padding="max_length", truncation=True
        )

        # Export to ONNX
        onnx_path = ONNX_MODEL
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=18,
            do_constant_folding=True,
        )
        logger.info("ONNX model exported", path=onnx_path)
    except Exception as e:
        logger.warning("ONNX conversion failed", error=str(e))
        if not tflite_success:
            logger.error("Both TFLite and ONNX conversions failed")
            raise RuntimeError("Failed to export model in any Android-compatible format")

    # ==========================================
    # 4. Save Vocab
    # ==========================================
    with open(VOCAB_TXT, "w") as f:
        for token in tokenizer.get_vocab().keys():
            f.write(token + "\n")
    logger.info("Vocab saved", path=VOCAB_TXT)


if __name__ == "__main__":
    convert_models()
