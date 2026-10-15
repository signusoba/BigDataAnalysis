"""Generate a clearly labeled, font-disjoint synthetic development corpus.

This is not Arshasb ground truth and does not establish camera OCR accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .letters import LETTERS, label_index, normalize_label
from .preprocessing import preprocess_image

FONT_GROUPS = {
    "Noto Naskh Arabic": ("train", "noto/NotoNaskhArabic-*.ttf"),
    "Noto Sans Arabic": ("train", "noto/NotoSansArabic-*.ttf"),
    "Noto Nastaliq Urdu": ("train", "noto/NotoNastaliqUrdu-*.ttf"),
    "DejaVu Sans": ("validation", "dejavu/DejaVuSans.ttf"),
    "Noto Kufi Arabic": ("test", "noto/NotoKufiArabic-*.ttf"),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_identities(row: dict) -> list[tuple[str, str]]:
    """Check supplied provenance fields, in addition to the declared group_id.

    Callers must precompute connected components of related originals/derivatives
    as group_id. Per-field checks cannot infer missing links or image genealogy.
    """
    return [(key, str(row[key])) for key in
            ("font_family", "group_id", "page_id", "specimen_id", "session_id", "capture_id",
             "writer_id", "original_id", "parent_id")
            if row.get(key) is not None]


def read_manifest(path: Path) -> list[dict]:
    """Validate supplied labels and provenance without guessing missing labels."""
    records = []
    group_splits, hash_splits = {}, {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        label = normalize_label(row.get("label", row.get("character", "")))
        split = row.get("split")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"{path.name}:{line_number}: split must be train, validation, or test.")
        relative = row.get("path", row.get("image"))
        if not relative:
            raise ValueError(f"{path.name}:{line_number}: missing image path.")
        image_path = (path.parent / relative).resolve()
        if not image_path.is_relative_to(path.parent.resolve()) or not image_path.is_file():
            raise ValueError(f"{path.name}:{line_number}: image must exist within the manifest directory.")
        source = row.get("source", "unknown")
        if source != "synthetic_font" and row.get("verified") is not True:
            raise ValueError(f"{path.name}:{line_number}: real images require verified=true and a reviewed label.")
        if source != "synthetic_font" and (not row.get("group_id") or not str(row.get("reviewer", "")).strip()):
            raise ValueError(f"{path.name}:{line_number}: reviewed real images require group_id and reviewer.")
        if source == "synthetic_font" and not row.get("font_family"):
            raise ValueError(f"{path.name}:{line_number}: synthetic images require a font_family split group.")
        actual_hash = sha256_file(image_path)
        if row.get("sha256") and actual_hash != row["sha256"]:
            raise ValueError(f"Image checksum mismatch: {image_path.name}")
        for mapping, key, description in [
            (group_splits, identity, identity[0]) for identity in group_identities(row)
        ] + [(hash_splits, actual_hash, "identical image")]:
            if key is not None and key in mapping and mapping[key] != split:
                raise ValueError(f"Data leakage: {description} occurs in more than one split: {key}")
            if key is not None:
                mapping[key] = split
        records.append({**row, "label": label, "index": label_index(label), "split": split,
                        "image_path": str(image_path), "sha256": actual_hash})
    if not records:
        raise ValueError("Manifest is empty.")
    return records


def generate_corpus(output: Path, samples_per_font: int = 20, font_root: Path = Path("/usr/share/fonts/truetype")) -> dict:
    from fontTools.ttLib import TTFont

    if samples_per_font < 2:
        raise ValueError("Use at least two renderings per font.")
    if (output / "manifest.jsonl").exists():
        # Repeat runs verify instead of silently overwriting reviewed or generated data.
        rows = read_manifest(output / "manifest.jsonl")
        summary_path = output / "dataset_manifest.json"
        if not summary_path.exists():
            raise ValueError("Output contains a manifest without generator provenance; choose a new output directory.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("samples_per_font") != samples_per_font:
            raise ValueError("Existing corpus uses a different sample count; choose a new output directory.")
        if len(rows) != summary["count"]:
            raise ValueError("Existing manifest count does not match its provenance.")
        return summary
    output.mkdir(parents=True, exist_ok=True)
    records, fonts = [], []
    for family, (split, pattern) in FONT_GROUPS.items():
        paths = sorted(font_root.glob(pattern))
        if not paths:
            raise RuntimeError(f"Missing font family {family}; install the fonts-noto-core and fonts-dejavu-core packages.")
        for path in paths:
            with TTFont(path) as font_file:
                cmap = font_file.getBestCmap()
                missing = [letter["character"] for letter in LETTERS if ord(letter["character"]) not in cmap]
                if missing:
                    raise RuntimeError(f"{path.name} does not cover the Persian alphabet: {missing}")
            fonts.append({"family": family, "file": str(path), "sha256": sha256_file(path), "split": split})
            for letter in LETTERS:
                for sample in range(samples_per_font):
                    font_size = 48 + round(sample * 64 / max(samples_per_font - 1, 1))
                    font = ImageFont.truetype(str(path), font_size)
                    # Drawing bounds include all detached marks. No contextual word forms.
                    bounds = font.getbbox(letter["character"], direction="rtl", language="fa")
                    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
                    image = Image.new("L", (max(144, width + 32), max(144, height + 32)), 255)
                    draw = ImageDraw.Draw(image)
                    draw.text(((image.width - width) // 2 - bounds[0], (image.height - height) // 2 - bounds[1]),
                              letter["character"], font=font, fill=0, direction="rtl", language="fa")
                    preprocess_image(image)  # Reject blank/missing glyph renderings before labeling.
                    relative = Path("images") / split / f"{path.stem}_{letter['index']:02d}_{sample:02d}.png"
                    destination = output / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    image.save(destination)
                    records.append({"path": str(relative), "label": letter["character"], "index": letter["index"],
                                    "name": letter["name"], "split": split, "source": "synthetic_font",
                                    "font_family": family, "font_file": path.name,
                                    "font_size": font_size, "group_id": f"font-family:{family}",
                                    "sha256": sha256_file(destination), "verified": False})
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    read_manifest(manifest)  # Cross-split hash/family/group validation.
    counts = {split: sum(row["split"] == split for row in records) for split in ("train", "validation", "test")}
    summary = {"domain": "synthetic isolated printed Persian letters", "real_camera_images": 0,
               "generator": "Pillow RAQM isolated canonical Unicode glyph rendering", "seed": 42,
               "samples_per_font": samples_per_font, "count": len(records), "counts": counts,
               "classes": LETTERS, "fonts": fonts, "manifest_sha256": sha256_file(manifest),
               "split_policy": "Entire font families, including every weight and size, belong to one split.",
               "augmentation_policy": "Training loader only; no flips; no validation/test augmentation.",
               "limitations": ["Synthetic font renders are not captured handwriting or camera ground truth.",
                               "Repeated sizes of the same font are related renderings, not independent real originals."]}
    (output / "dataset_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/letters"))
    parser.add_argument("--samples-per-font", type=int, default=20)
    parser.add_argument("--font-root", type=Path, default=Path("/usr/share/fonts/truetype"),
                        help="Font directory containing noto/ and dejavu/ subdirectories.")
    arguments = parser.parse_args()
    summary = generate_corpus(arguments.output, arguments.samples_per_font, arguments.font_root)
    print(json.dumps({key: summary[key] for key in ("domain", "count", "counts", "manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
