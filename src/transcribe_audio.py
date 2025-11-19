from pathlib import Path
from typing import Set

import librosa
import pandas as pd
import structlog
import torch
from transformers import pipeline

logger = structlog.get_logger()


def transcribe_dataset() -> None:
    """
    Transcribe audio files from data/PhishingVoiceDataset using OpenAI Whisper.
    Structure:
        data/PhishingVoiceDataset/
            NonPhishing/ (Label 0)
            Phishing/    (Label 1)
    Saves progress after every file to prevent data loss.
    """
    # Check for GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Using device", device=device)

    # Initialize Whisper pipeline
    # Using openai/whisper-tiny for speed. Can be changed to base/small/medium/large.
    # Note: This requires 'ffmpeg' to be installed on the system.
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=device,
            chunk_length_s=30,
        )
    except Exception as e:
        logger.error(
            "Failed to initialize pipeline. Ensure 'transformers', 'torch' are installed. "
            "You may also need 'ffmpeg' and 'librosa'/'soundfile'.",
            error=str(e),
        )
        return

    base_dir = Path("data/PhishingVoiceDataset")
    if not base_dir.exists():
        logger.error("Directory not found", path=str(base_dir))
        return

    output_path = base_dir / "dataset.csv"

    # Load existing data if available (for resuming)
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        data = existing_df.to_dict("records")
        processed_files: Set[str] = set(existing_df["filename"].tolist())
        logger.info("Resuming from existing dataset", entries=len(data))
    else:
        data = []
        processed_files: Set[str] = set()

    # Define categories and labels
    # User prompt mentioned "Phishing as label 0", but standard convention is 1 for malicious.
    # Using 1 for Phishing to maintain consistency with other datasets (e.g. Robocall).
    categories = {"NonPhishing": 0, "Phishing": 1}

    for category, label in categories.items():
        cat_dir = base_dir / category
        if not cat_dir.exists():
            logger.warning("Category directory not found", path=str(cat_dir))
            continue

        files = list(cat_dir.glob("*.mp3"))
        logger.info("Found files in category", category=category, count=len(files))

        for idx, file_path in enumerate(files):
            # Skip already processed files
            if file_path.name in processed_files:
                logger.info("Skipping already processed file", filename=file_path.name)
                continue

            try:
                logger.info("Transcribing file", filename=file_path.name, progress=f"{idx+1}/{len(files)}")
                # Load audio using librosa (uses soundfile/audioread internally)
                # Whisper expects 16kHz audio
                audio, sr = librosa.load(str(file_path), sr=16000)

                # Pass dictionary to pipeline
                result = pipe({"array": audio, "sampling_rate": sr}, batch_size=8)
                text = result["text"]

                data.append({"text": text, "label": label, "filename": file_path.name, "category": category})

                # Save after every file to prevent data loss
                df = pd.DataFrame(data)
                df.to_csv(output_path, index=False)
                logger.info("Saved progress", total_transcriptions=len(data))

            except Exception as e:
                logger.error("Failed to transcribe file", filename=file_path.name, error=str(e))

    if not data:
        logger.warning("No data transcribed")
        return

    logger.info("Transcription complete", total_samples=len(data))


if __name__ == "__main__":
    transcribe_dataset()
