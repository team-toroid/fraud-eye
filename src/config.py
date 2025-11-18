from pathlib import Path
from typing import List, Tuple

# === Directories ===
ROOT_DIR = Path(__file__).parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = MODELS_DIR / "logs"
TF_MODEL_DIR = MODELS_DIR / "tf_model"

for d in [DATASETS_DIR, MODELS_DIR, RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, TF_MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Model Config ===
MODEL_NAME: str = "distilbert-base-uncased"
MAX_LENGTH: int = 128
BATCH_SIZE: int = 16
EPOCHS: int = 4
LEARNING_RATE: float = 3e-5
WEIGHT_DECAY: float = 0.01
FP16: bool = True

# === Dataset Files ===
MERGED_CSV = DATASETS_DIR / "fraudeye_merged.csv"
TRAIN_CSV = DATASETS_DIR / "fraudeye_train.csv"
VAL_CSV = DATASETS_DIR / "fraudeye_val.csv"
TEST_CSV = DATASETS_DIR / "fraudeye_test.csv"

# === Model Outputs ===
FINAL_MODEL_DIR = MODELS_DIR / "fraudeye-distilbert"
TFLITE_MODEL = MODELS_DIR / "scam_detector.tflite"
VOCAB_TXT = MODELS_DIR / "vocab.txt"

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
    "FredZhang7/all-scam-spam",
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
