#!/usr/bin/env python3
"""Create a runnable source/data/model ZIP without environment caches or secrets."""

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[1]


def package(output: Path, include_data: bool = True) -> dict:
    paths = set()
    for pattern in ["README.md", ".gitignore", "Makefile", "pyproject.toml", "requirements*.txt", "requirements.lock",
                    "persian_ocr/*.py", "web/*", "scripts/*.py", "scripts/*.sh", "tests/*.py", "docs/**/*.md",
                    "docs/*.json", "docs/screenshots/*.png"]:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    for name in ["cnn.pt", "hog.joblib", "mobilenet.pt", "metrics.json", "model_metadata.json", "selection.json"]:
        path = ROOT / "artifacts" / name
        if path.is_file():
            paths.add(path)
    if include_data:
        for directory in [ROOT / "data/raw/arshasb", ROOT / "data/letters"]:
            if directory.is_dir():
                paths.update(path for path in directory.rglob("*") if path.is_file() and not path.is_symlink())
    output = output.resolve()
    if output in paths:
        raise ValueError("Output must not replace a source or model file")
    output.parent.mkdir(parents=True, exist_ok=True)
    checksums = []
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            if not path.resolve().is_relative_to(ROOT) or path.is_symlink():
                raise ValueError("Refusing to package a path outside the project")
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            checksums.append(f"{hashlib.sha256(content).hexdigest()}  {relative}")
            archive.writestr("Noor-Persian-Letter-Lab/" + relative, content)
        archive.writestr("Noor-Persian-Letter-Lab/SHA256SUMS", "\n".join(checksums) + "\n")
    return {"archive": str(output), "files": len(paths) + 1, "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/Noor-Persian-Letter-Lab.zip")
    parser.add_argument("--source-only", action="store_true", help="Omit sample and generated image data; models remain included")
    args = parser.parse_args()
    import json
    print(json.dumps(package(args.output, not args.source_only), indent=2))
