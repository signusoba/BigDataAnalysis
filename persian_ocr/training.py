"""Train comparisons, select on validation, then evaluate the locked test set."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import time

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn

from .corpus import group_identities, read_manifest, sha256_file
from .letters import LETTERS
from .models import CompactCNN, hog_features, mobilenet_model, model_inputs
from .preprocessing import IMAGE_SIZE, preprocess_image


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _load_images(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    images = []
    for row in records:
        with Image.open(row["image_path"]) as image:
            images.append(preprocess_image(image))
    return np.stack(images), np.asarray([row["index"] for row in records], dtype=np.int64)


def _augment(images: torch.Tensor) -> torch.Tensor:
    # Mild affine variation; no flips, component removal, or destructive thresholding.
    count = len(images)
    angle = (torch.rand(count) - 0.5) * math.radians(10)
    scale = 0.92 + torch.rand(count) * 0.16
    transform = torch.zeros(count, 2, 3)
    transform[:, 0, 0] = torch.cos(angle) * scale
    transform[:, 0, 1] = -torch.sin(angle) * scale
    transform[:, 1, 0] = torch.sin(angle) * scale
    transform[:, 1, 1] = torch.cos(angle) * scale
    transform[:, :, 2] = (torch.rand(count, 2) - 0.5) * 0.08
    grid = nn.functional.affine_grid(transform, images.shape, align_corners=False)
    return nn.functional.grid_sample(images, grid, align_corners=False, padding_mode="zeros")


def _torch_probabilities(model, images: np.ndarray, name: str, batch_size: int = 128) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start:start + batch_size])
            predictions.append(torch.softmax(model(model_inputs(batch, name)), dim=1).numpy())
    return np.concatenate(predictions)


def _probabilities(candidate: dict, images: np.ndarray) -> np.ndarray:
    if candidate["name"] == "hog":
        return candidate["model"].predict_proba(hog_features(images))
    return _torch_probabilities(candidate["model"], images, candidate["name"])


def _latency(candidate: dict, images: np.ndarray) -> dict:
    samples = images[:min(64, len(images))]
    for sample in samples[:3]:
        _probabilities(candidate, sample[None])
    times = []
    for sample in samples:
        begin = time.perf_counter()
        _probabilities(candidate, sample[None])
        times.append((time.perf_counter() - begin) * 1000)
    return {"median": float(np.median(times)), "p95": float(np.percentile(times, 95)),
            "count": len(times), "scope": "single-image CPU prediction including HOG extraction; excludes image loading/preprocessing"}


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = probabilities.argmax(axis=1)
    indices = np.arange(len(LETTERS))
    precision, recall, f1, support = precision_recall_fscore_support(labels, predicted, labels=indices, zero_division=0)
    matrix = confusion_matrix(labels, predicted, labels=indices)
    count = len(labels)
    accuracy = float(accuracy_score(labels, predicted))
    z = 1.959963984540054
    divisor = 1 + z * z / count
    center = (accuracy + z * z / (2 * count)) / divisor
    radius = z * math.sqrt(accuracy * (1 - accuracy) / count + z * z / (4 * count * count)) / divisor
    present = support > 0
    return {"accuracy": accuracy, "macro_precision": float(precision[present].mean()),
            "macro_recall": float(recall[present].mean()), "macro_f1": float(f1[present].mean()),
            "count": count, "correct": int((predicted == labels).sum()),
            "accuracy_ci95": [max(0.0, center - radius), min(1.0, center + radius)],
            "ci_method": "Wilson binomial 95%; synthetic samples within a font are correlated, so this is descriptive, not a camera generalization guarantee.",
            "confusion_matrix": matrix.tolist(), "confusion_labels": [item["character"] for item in LETTERS],
            "per_class": [{**letter, "support": int(support[i]), "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i]),
                           "accuracy": float(recall[i])} for i, letter in enumerate(LETTERS)]}


def _train_torch(name, train_x, train_y, validation_x, validation_y, epochs, batch_size) -> tuple[dict, list]:
    model = CompactCNN() if name == "cnn" else mobilenet_model(pretrained=True)
    train_tensor, label_tensor = torch.from_numpy(train_x), torch.from_numpy(train_y)
    feature_train = feature_validation = None
    if name == "mobilenet":
        # Frozen ImageNet transfer comparator: cache frozen features once, train the head.
        model.eval()
        def features(images):
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(images), batch_size):
                    batch = model_inputs(torch.from_numpy(images[start:start + batch_size]), name)
                    chunks.append(model.features(batch).mean((2, 3)))
            return torch.cat(chunks).clone()
        feature_train, feature_validation = features(train_x), features(validation_x)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.001)
    best_score, best_state, best_epoch, history = -1.0, None, 0, []
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        if name == "mobilenet":
            model.features.eval()
        loss_total = 0.0
        order = torch.randperm(len(train_y))
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            if name == "mobilenet":
                logits = model.classifier(feature_train[chosen])
            else:
                logits = model(_augment(train_tensor[chosen]))
            loss = loss_function(logits, label_tensor[chosen])
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(chosen)
        model.eval()
        if name == "mobilenet":
            with torch.inference_mode():
                probabilities = model.classifier(feature_validation).softmax(1).numpy()
        else:
            probabilities = _torch_probabilities(model, validation_x, name)
        report = classification_metrics(validation_y, probabilities)
        history.append({"epoch": epoch, "train_loss": loss_total / len(train_y),
                        "validation_accuracy": report["accuracy"], "validation_macro_f1": report["macro_f1"]})
        if report["macro_f1"] > best_score:
            best_score, best_state, best_epoch = report["macro_f1"], copy.deepcopy(model.state_dict()), epoch
        print(f"{name} epoch {epoch}/{epochs}: loss={loss_total / len(train_y):.4f} validation_f1={report['macro_f1']:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return {"name": name, "model": model, "best_epoch": best_epoch}, history


def train(data: Path, output: Path, epochs: int = 12, models: list[str] | None = None,
          additional_manifests: list[Path] | None = None, batch_size: int = 64, overwrite: bool = False) -> dict:
    models = models or ["cnn", "hog"]
    if not models or len(models) != len(set(models)) or set(models) - {"cnn", "hog", "mobilenet"}:
        raise ValueError("Choose unique models from cnn, hog, mobilenet.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    if (output / "metrics.json").exists() and not overwrite:
        raise ValueError("Output already has an evaluation. Use a new directory or explicit --overwrite; do not tune against prior test results.")
    torch_cache = Path(os.environ.setdefault("TORCH_HOME", str((output / "cache" / "torch").resolve())))
    torch_cache.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    manifests = [data if data.is_file() else data / "manifest.jsonl"] + (additional_manifests or [])
    records = [row for manifest in manifests for row in read_manifest(manifest)]
    # Check all manifests together: separate imports must not bypass leakage checks.
    observed = {}
    for row in records:
        for key, value in [("hash", row["sha256"])] + group_identities(row):
            if value is None:
                continue
            identity = (key, value)
            if identity in observed and observed[identity] != row["split"]:
                raise ValueError(f"Cross-manifest data leakage: {key} {value}")
            observed[identity] = row["split"]
    splits = {split: [row for row in records if row["split"] == split] for split in ("train", "validation", "test")}
    for split, rows in splits.items():
        if {row["index"] for row in rows} != set(range(32)):
            raise ValueError(f"{split} must contain all 32 classes; annotate missing classes before training.")
    output.mkdir(parents=True, exist_ok=True)
    train_x, train_y = _load_images(splits["train"])
    validation_x, validation_y = _load_images(splits["validation"])
    report = {"seed": 42, "experiment_seeds_completed": [42], "classes": LETTERS,
              "data": {"domain": "synthetic isolated printed Persian letters" if all(r.get("source") == "synthetic_font" for r in records) else "mixed synthetic and manually reviewed isolated-letter crops",
                       "counts": {split: len(rows) for split, rows in splits.items()},
                       "unique_image_hashes": len({r["sha256"] for r in records}),
                       "source_counts": {source: sum(r.get("source", "unknown") == source for r in records) for source in sorted({r.get("source", "unknown") for r in records})},
                       "manifests": [{"name": str(path), "sha256": sha256_file(path)} for path in manifests],
                       "font_families": {split: sorted({r["font_family"] for r in rows if "font_family" in r}) for split, rows in splits.items()},
                       "split_policy": "Font families, source groups, and exact image hashes cannot cross splits."},
              "models": {}, "limitations": [
                  "No model can be promised 100% accuracy on unseen images; measured test accuracy is evidence only for this split.",
                  "Camera and handwriting accuracy has not been measured unless separately reviewed camera samples appear in the source-specific test report.",
                  "Synthetic renderings are related by font; report counts are images, not independent real originals.",
                  "Softmax/SVM scores are uncalibrated model scores, not guaranteed probabilities of correctness.",
                  "Only seed 42 was run; the planned three-seed experiment is not yet performed.",
                  "Scope is one isolated canonical letter; no word segmentation or connected/positional character recognition."]}
    candidates = {}
    for name in models:
        begin = time.perf_counter()
        if name == "hog":
            model = make_pipeline(StandardScaler(), SVC(C=10, kernel="rbf", gamma="scale", probability=True, random_state=42))
            model.fit(hog_features(train_x), train_y)
            candidate, history = {"name": name, "model": model}, []
        else:
            candidate, history = _train_torch(name, train_x, train_y, validation_x, validation_y, epochs, batch_size)
        candidates[name] = candidate
        validation = classification_metrics(validation_y, _probabilities(candidate, validation_x))
        validation["latency_ms"] = _latency(candidate, validation_x)
        report["models"][name] = {"validation": validation, "training_seconds": time.perf_counter() - begin,
                                  "history": history, "best_epoch": candidate.get("best_epoch"),
                                  "pretrained": name == "mobilenet",
                                  "training_mode": "frozen ImageNet features + trained 32-class head" if name == "mobilenet" else "trained on provided train split"}
        if name == "hog":
            joblib.dump(candidate["model"], output / "hog.joblib")
        else:
            torch.save(candidate["model"].state_dict(), output / f"{name}.pt")
    best_f1 = max(report["models"][name]["validation"]["macro_f1"] for name in models)
    tied = [name for name in models if best_f1 - report["models"][name]["validation"]["macro_f1"] < 0.01]
    selected = min(tied, key=lambda name: report["models"][name]["validation"]["latency_ms"]["median"])
    selection = {"selected_model": selected, "split": "validation", "metric": "macro_f1",
                 "tie_rule": "Candidates within less than 0.01 macro-F1 of best: choose lower validation median CPU latency.",
                 "locked_before_test": True, "validation_scores": {name: report["models"][name]["validation"]["macro_f1"] for name in models},
                 "artifact_sha256": {name: sha256_file(output / ("hog.joblib" if name == "hog" else f"{name}.pt")) for name in models}}
    _write_json(output / "selection.json", selection)
    print(f"Locked selected model from validation: {selected}; evaluating test once per candidate.", flush=True)
    test_x, test_y = _load_images(splits["test"])
    for name, candidate in candidates.items():
        probabilities = _probabilities(candidate, test_x)
        evaluation = classification_metrics(test_y, probabilities)
        evaluation["latency_ms"] = _latency(candidate, test_x)
        evaluation["by_source"] = {}
        for source in sorted({row.get("source", "unknown") for row in splits["test"]}):
            mask = np.asarray([row.get("source", "unknown") == source for row in splits["test"]])
            evaluation["by_source"][source] = classification_metrics(test_y[mask], probabilities[mask])
        report["models"][name]["test"] = evaluation
        print(f"{name}: test_accuracy={evaluation['accuracy']:.4f}, test_macro_f1={evaluation['macro_f1']:.4f}", flush=True)
    report.update(selected_model=selected, selection=selection,
                  artifact_sha256=selection["artifact_sha256"], test_evaluations_per_candidate=1)
    _write_json(output / "metrics.json", report)
    metadata = {"model_name": selected, "selected_model": selected, "loaded": True, "classes": LETTERS,
                "input_shape": [1, IMAGE_SIZE, IMAGE_SIZE], "ink_value": 1, "background_value": 0,
                "data": report["data"], "validation": report["models"][selected]["validation"],
                "test": report["models"][selected]["test"], "limitations": report["limitations"],
                "review_threshold": 0.85, "review_margin": 0.20,
                "artifact_sha256": selection["artifact_sha256"][selected],
                "selection": selection}
    _write_json(output / "model_metadata.json", metadata)
    from .inference import Recognizer, measure_deployment_latency
    deployment_timing = measure_deployment_latency(Recognizer(output), splits["validation"])
    report["deployment_latency_ms"] = deployment_timing
    metadata["deployment_latency_ms"] = deployment_timing
    _write_json(output / "metrics.json", report)
    _write_json(output / "model_metadata.json", metadata)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/letters"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--models", nargs="+", choices=["cnn", "hog", "mobilenet"], default=["cnn", "hog"])
    parser.add_argument("--additional-manifest", type=Path, action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.models, args.additional_manifest, args.batch_size, args.overwrite)


if __name__ == "__main__":
    main()
