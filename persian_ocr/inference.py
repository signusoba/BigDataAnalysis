"""Load a locally trained artifact and return ranked, reviewable predictions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
from PIL import Image
import torch

from .corpus import sha256_file
from .letters import LETTERS
from .models import CompactCNN, hog_features, mobilenet_model, model_inputs
from .preprocessing import preprocess_image, preview_data_url


class Recognizer:
    def __init__(self, artifacts_dir: Path):
        self.loaded = False
        self.model_name = "untrained"
        self.metadata = {}
        self.model = None
        metadata_path = Path(artifacts_dir) / "model_metadata.json"
        if not metadata_path.exists():
            return
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.model_name = self.metadata["selected_model"]
        path = Path(artifacts_dir) / ("hog.joblib" if self.model_name == "hog" else f"{self.model_name}.pt")
        if not path.exists():
            return
        if self.metadata.get("artifact_sha256") != sha256_file(path):
            raise RuntimeError("Model artifact checksum differs from its training report. Retrain or restore the matching artifact.")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        if self.model_name == "hog":
            # Only load artifacts generated locally by the trusted training command.
            self.model = joblib.load(path)
        elif self.model_name in {"cnn", "mobilenet"}:
            self.model = CompactCNN() if self.model_name == "cnn" else mobilenet_model(pretrained=False)
            self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self.model.eval()
        else:
            raise RuntimeError(f"Unknown trained model: {self.model_name}")
        self.loaded = True

    def predict(self, image: Image.Image) -> dict:
        if not self.loaded:
            raise RuntimeError("No trained model is available. Generate the synthetic corpus and run the training command first.")
        begin = time.perf_counter()
        prepared = preprocess_image(image)
        if self.model_name == "hog":
            probabilities = self.model.predict_proba(hog_features(prepared[None]))[0]
        else:
            with torch.inference_mode():
                logits = self.model(model_inputs(torch.from_numpy(prepared[None]), self.model_name))
                probabilities = logits.softmax(1).numpy()[0]
        ranked = np.argsort(probabilities)[::-1][:5]
        top = [{**LETTERS[int(index)], "score": float(probabilities[index])} for index in ranked]
        margin = top[0]["score"] - top[1]["score"]
        needs_review = top[0]["score"] < self.metadata.get("review_threshold", 0.85) or margin < self.metadata.get("review_margin", 0.20)
        warnings = ["Prototype recognizes one isolated letter. Scores are uncalibrated; verify the label before saving an annotation."]
        if self.metadata.get("data", {}).get("domain", "").startswith("synthetic"):
            warnings.append("Trained and tested on synthetic font images; camera-image accuracy is not established.")
        if needs_review:
            warnings.append("Low score or small score margin: manual review is required.")
        return {"prediction": top[0], "top_predictions": top, "needs_review": bool(needs_review),
                "warnings": warnings, "model_name": self.model_name,
                "latency_ms": (time.perf_counter() - begin) * 1000,
                "preprocessed_image": preview_data_url(prepared)}


def measure_deployment_latency(recognizer: Recognizer, validation_records: list[dict], sample_count: int = 64) -> dict:
    """Benchmark the complete local response path on validation images only.

    Includes EXIF/grayscale/cropping/resizing, model inference, label mapping,
    ranked scores and PNG preview encoding. Excludes disk I/O, HTTP and capture.
    """
    if not validation_records or any(row.get("split") != "validation" for row in validation_records):
        raise ValueError("Provide only nonempty validation records for deployment timing.")
    indices = np.linspace(0, len(validation_records) - 1, min(sample_count, len(validation_records)), dtype=int)
    images = []
    for index in indices:
        with Image.open(validation_records[int(index)]["image_path"]) as image:
            images.append(image.copy())
    for image in images[:3]:
        recognizer.predict(image)
    times = []
    for image in images:
        begin = time.perf_counter()
        recognizer.predict(image)
        times.append((time.perf_counter() - begin) * 1000)
    return {"median": float(np.median(times)), "p95": float(np.percentile(times, 95)),
            "count": len(times), "split": "validation", "device": "CPU",
            "scope": "preprocessing + prediction + ranked label mapping + preview PNG encoding; excludes disk I/O, HTTP, and capture"}
