import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest

from persian_ocr.corpus import read_manifest
from persian_ocr.inference import Recognizer
from persian_ocr.letters import LETTERS, normalize_label
from persian_ocr.preprocessing import preprocess_image, preview_data_url
from persian_ocr.training import classification_metrics, train


def glyph_image():
    image = Image.new("L", (120, 100), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 55, 90, 66), fill=0)
    draw.ellipse((46, 20, 53, 27), fill=0)
    draw.ellipse((64, 20, 71, 27), fill=0)
    return image


def test_canonical_labels_reject_words_and_normalize_variants():
    assert len(LETTERS) == 32
    assert len({row["character"] for row in LETTERS}) == 32
    assert normalize_label(" ك ") == "ک"
    assert normalize_label("ي") == "ی"
    assert normalize_label("ﺏ") == "ب"
    for invalid in ("", "سلام", "1", "آ", "ب ت"):
        with pytest.raises(ValueError):
            normalize_label(invalid)


def test_preprocessing_preserves_detached_dots_and_polarity():
    prepared = preprocess_image(glyph_image())
    assert prepared.shape == (1, 64, 64)
    assert prepared.dtype == np.float32
    assert prepared[0, :32].sum() > 3  # Both separate dots remain above the base.
    assert prepared[0, 32:].sum() > 20
    inverted = Image.fromarray(255 - np.asarray(glyph_image()))
    np.testing.assert_allclose(preprocess_image(inverted), prepared, atol=1 / 255)
    assert preview_data_url(prepared).startswith("data:image/png;base64,")


def test_transparent_background_and_blank_rejection():
    image = glyph_image().convert("RGBA")
    array = np.asarray(image).copy()
    array[:, :, 3] = np.where(array[:, :, 0] == 255, 0, 255)
    np.testing.assert_allclose(preprocess_image(Image.fromarray(array)), preprocess_image(glyph_image()))
    for blank in (Image.new("L", (64, 64), 255), Image.new("L", (64, 64), 0)):
        with pytest.raises(ValueError, match="blank"):
            preprocess_image(blank)


def manifest_row(path, split="train", **extra):
    return {"path": path, "label": "ب", "split": split, "source": "camera", "verified": True,
            "reviewer": "reviewer1", "group_id": "capture1", **extra}


def save_manifest(root: Path, rows):
    path = root / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_manifest_requires_verified_real_ground_truth_and_group(tmp_path):
    glyph_image().save(tmp_path / "glyph.png")
    for override in ({"verified": False}, {"group_id": None}, {"reviewer": ""}):
        path = save_manifest(tmp_path, [manifest_row("glyph.png", **override)])
        with pytest.raises(ValueError):
            read_manifest(path)
    rows = read_manifest(save_manifest(tmp_path, [manifest_row("glyph.png")]))
    assert rows[0]["index"] == 1


def test_manifest_blocks_duplicate_pixels_and_related_pages_cross_split(tmp_path):
    glyph_image().save(tmp_path / "first.png")
    image = glyph_image()
    ImageDraw.Draw(image).point((0, 0), fill=30)
    image.save(tmp_path / "second.png")
    for second in (manifest_row("first.png", "test", group_id="different"),
                   manifest_row("second.png", "test", group_id="different", page_id="page1")):
        first = manifest_row("first.png", page_id="page1")
        with pytest.raises(ValueError, match="leakage"):
            read_manifest(save_manifest(tmp_path, [first, second]))


@pytest.mark.parametrize("group_key", ["writer_id", "original_id", "parent_id", "specimen_id", "session_id", "capture_id"])
def test_manifest_keeps_declared_provenance_groups_together(tmp_path, group_key):
    glyph_image().save(tmp_path / "first.png")
    second_image = glyph_image()
    ImageDraw.Draw(second_image).point((0, 0), fill=30)
    second_image.save(tmp_path / "second.png")
    first = manifest_row("first.png", group_id="one", **{group_key: "shared-origin"})
    second = manifest_row("second.png", "test", group_id="two", **{group_key: "shared-origin"})
    with pytest.raises(ValueError, match="leakage"):
        read_manifest(save_manifest(tmp_path, [first, second]))


def test_manifest_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="within"):
        read_manifest(save_manifest(tmp_path, [manifest_row("../outside.png")]))


def test_metrics_detect_errors_and_finite_sample_interval():
    labels = np.array([0, 0, 1, 1])
    predictions = np.zeros((4, 32))
    predictions[np.arange(4), [0, 1, 1, 1]] = 1
    report = classification_metrics(labels, predictions)
    assert report["accuracy"] == 0.75
    assert report["count"] == 4
    assert report["confusion_matrix"][0][:2] == [1, 1]
    assert report["accuracy_ci95"][0] < 0.75 < report["accuracy_ci95"][1]
    perfect = classification_metrics(labels, np.eye(32)[labels])
    assert perfect["accuracy"] == 1
    assert perfect["accuracy_ci95"][0] < 1


def test_missing_model_has_clear_error(tmp_path):
    recognizer = Recognizer(tmp_path)
    assert not recognizer.loaded
    with pytest.raises(RuntimeError, match="No trained model"):
        recognizer.predict(glyph_image())


def test_train_artifact_and_predict_roundtrip(tmp_path, monkeypatch):
    """Exercise actual optimizer, validation selection, artifact and inference paths.

    These deliberately tiny geometric fixtures test plumbing, not OCR accuracy.
    """
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch-cache"))
    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for letter in LETTERS:
            image = glyph_image()
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 80, 20 + letter["index"], 84), fill=20 + split_index * 20)
            draw.point((split_index, 0), fill=0)
            filename = f"{split}-{letter['index']}.png"
            image.save(tmp_path / filename)
            rows.append(manifest_row(filename, split, label=letter["character"], group_id=f"{split}-{letter['index']}"))
    save_manifest(tmp_path, rows)
    artifacts = tmp_path / "artifacts"
    report = train(tmp_path, artifacts, epochs=1, models=["cnn"], batch_size=32)
    assert report["selection"]["locked_before_test"]
    assert report["models"]["cnn"]["test"]["count"] == 32
    assert report["deployment_latency_ms"]["split"] == "validation"
    recognizer = Recognizer(artifacts)
    assert recognizer.loaded
    prediction = recognizer.predict(glyph_image())
    assert prediction["prediction"]["character"] in {row["character"] for row in LETTERS}
    assert len(prediction["top_predictions"]) == 5
    assert 0 <= prediction["prediction"]["score"] <= 1
    assert prediction["preprocessed_image"].startswith("data:image/png;base64,")
    # Corruption is rejected before deserialization.
    (artifacts / "cnn.pt").write_bytes(b"corrupted artifact")
    with pytest.raises(RuntimeError, match="checksum"):
        Recognizer(artifacts)
