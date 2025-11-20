from pathlib import Path
from typing import List, Tuple

# === Directories ===
ROOT_DIR = Path(__file__).parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
TRAINING_DIR = MODELS_DIR / "training"
CONVERSION_DIR = MODELS_DIR / "conversion"
FINAL_OUTPUT_DIR = MODELS_DIR / "final"

CHECKPOINTS_DIR = TRAINING_DIR / "checkpoints"
LOGS_DIR = TRAINING_DIR / "logs"
TF_MODEL_DIR = CONVERSION_DIR / "tf_model"

for d in [
    DATASETS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    TRAINING_DIR,
    CONVERSION_DIR,
    FINAL_OUTPUT_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    TF_MODEL_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# === Model Config ===
MODEL_NAME: str = "distilbert-base-uncased"
MAX_LENGTH: int = 128
BATCH_SIZE: int = 32  # Increased for more stable gradients
EPOCHS: int = 15  # Increased for better convergence
LEARNING_RATE: float = 2e-5  # Reduced for more stable training
WEIGHT_DECAY: float = 0.01
FP16: bool = False
WARMUP_RATIO: float = 0.15  # Increased warmup for better convergence
GRADIENT_ACCUMULATION_STEPS: int = 2  # Effective batch size = 64
EARLY_STOPPING_PATIENCE: int = 3  # Stop if no improvement for 3 epochs

# === Dataset Files ===
MERGED_CSV = DATASETS_DIR / "fraudeye_merged.csv"
SYNTHETIC_CSV = DATASETS_DIR / "fraudeye_synthetic.csv"
TRAIN_CSV = DATASETS_DIR / "fraudeye_train.csv"
VAL_CSV = DATASETS_DIR / "fraudeye_val.csv"
TEST_CSV = DATASETS_DIR / "fraudeye_test.csv"
EDGE_CASE_CSV = DATASETS_DIR / "fraudeye_edge_case.csv"

# === Model Outputs ===
FINAL_MODEL_DIR = TRAINING_DIR / "fraudeye-distilbert"
TFLITE_MODEL = FINAL_OUTPUT_DIR / "scam_detector.tflite"
ONNX_MODEL = FINAL_OUTPUT_DIR / "scam_detector.onnx"
VOCAB_TXT = FINAL_OUTPUT_DIR / "vocab.txt"

# === Results ===
METRICS_TXT = RESULTS_DIR / "metrics.txt"
CONFUSION_PLOT = RESULTS_DIR / "confusion_matrix.png"
ROC_PLOT = RESULTS_DIR / "roc_curve.png"

# === Datasets ===
HF_DATASETS: List[str] = [
    "BothBosu/scam-dialogue",
    "BothBosu/multi-agent-scam-conversation",
    "BothBosu/single-agent-scam-conversations",
    "BothBosu/Scammer-Conversation",
    "menaattia/phone-scam-dataset",
]

# External datasets from other sources
EXTERNAL_HF_DATASETS: List[str] = [
    "wspr-ncsu/robocall-audio-dataset",  # Robocall transcriptions
]

KAGGLE_DATASETS: List[Tuple[str, str, str, str]] = [
    (
        "mealss/call-transcripts-scam-determinations",
        "call_transcripts_scam_determinations.csv",
        "transcript",
        "is_scam",
    ),
    ("narayanyadav/fraud-call-india-dataset", "fraud_call_detection.csv", "description", "label"),
    ("kumarperiya/comprehensive-indian-online-fraud-dataset", "fraud_detection.csv", "description", "fraud_type"),
    ("divanshu22/scam-dataset", "scam_dataset.csv", "text", "label"),
    ("meloncc/shujing", "telecom_fraud.csv", "call_log", "is_fraud"),
]

# SMS Spam dataset from HuggingFace
SMS_SPAM_DATASET = "ucirvine/sms_spam"
