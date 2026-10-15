#!/usr/bin/env python3
"""Import the supplied Arshasb ZIP or sample tarball without executing its contents."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from zipfile import ZipFile, is_zipfile


MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_EXPANDED_BYTES = 1024 * 1024 * 1024


def safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Archive contains an unsafe path")
    return path


def write_preserving(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"Existing file differs; refusing to overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def import_archive(archive_path: Path, output: Path) -> dict:
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("This importer accepts sample archives up to 256 MiB")
    source_bytes = archive_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    license_bytes = None
    if is_zipfile(archive_path):
        with ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) > 10000 or sum(m.file_size for m in infos) > MAX_EXPANDED_BYTES:
                raise ValueError("ZIP exceeds safe sample-import limits")
            for info in infos:
                safe_relative(info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive links are not supported")
            candidates = [m for m in infos if PurePosixPath(m.filename).name == "Arshasb_samples.tar.gz"]
            if len(candidates) != 1 or candidates[0].file_size > MAX_ARCHIVE_BYTES:
                raise ValueError("Expected exactly one Arshasb_samples.tar.gz inside the ZIP")
            source_bytes = archive.read(candidates[0])
            licenses = [m for m in infos if PurePosixPath(m.filename).name == "LICENSE"]
            if len(licenses) == 1:
                license_bytes = archive.read(licenses[0])

    output = output.resolve()
    entries = []
    with tarfile.open(fileobj=io.BytesIO(source_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 50000 or sum(m.size for m in members) > MAX_EXPANDED_BYTES:
            raise ValueError("Tarball exceeds safe sample-import limits")
        seen = set()
        for member in members:
            path = safe_relative(member.name)
            if not (member.isdir() or member.isfile()):
                raise ValueError("Only ordinary files and directories may be imported")
            if member.isdir():
                continue
            parts = path.parts
            if parts and parts[0] == "Arshasb_samples":
                parts = parts[1:]
            if len(parts) != 2 or not re.fullmatch(r"\d{5}", parts[0]):
                raise ValueError(f"Unexpected dataset layout: {member.name}")
            relative = Path(*parts)
            if relative.suffix.lower() not in {".png", ".xlsx", ".txt"} or relative in seen:
                raise ValueError("Unexpected or duplicate dataset file")
            seen.add(relative)
            target = output / relative
            if not target.resolve().is_relative_to(output):
                raise ValueError("Dataset target escapes the output directory")
            entries.append((member, target))
        # Validate the whole archive before writing any dataset files.
        for member, target in entries:
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Dataset member could not be read")
            write_preserving(target, stream.read())

    if license_bytes:
        write_preserving(output / "SOURCE_LICENSE", license_bytes)
    pages = sorted(p.name for p in output.iterdir() if p.is_dir() and re.fullmatch(r"\d{5}", p.name))
    report = {"source": "Arshasb uploaded sample archive", "source_sha256": source_sha,
              "pages": len(pages), "imported_files": len(entries),
              "label_granularity": "word and line; no character bounding boxes",
              "upstream": "https://github.com/persiandataset/Arshasb"}
    report_path = output / "import_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw/arshasb"))
    args = parser.parse_args()
    print(json.dumps(import_archive(args.archive, args.output), indent=2))


if __name__ == "__main__":
    main()
