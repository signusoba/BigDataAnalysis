import io
from pathlib import Path
import tarfile

import pytest

from scripts.import_arshasb import import_archive, write_preserving


def make_archive(path: Path, name: str, content: bytes = b"example", link: bool = False):
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(name)
        if link:
            info.type = tarfile.SYMTYPE
            info.linkname = "/tmp/outside"
            archive.addfile(info)
        else:
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("name", ["../escaped.txt", "/tmp/escaped.txt", "Arshasb_samples/00001/../../escaped.txt"])
def test_archive_rejects_traversal_before_writing(tmp_path, name):
    archive = tmp_path / "sample.tar.gz"
    make_archive(archive, name)
    with pytest.raises(ValueError, match="unsafe path"):
        import_archive(archive, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_archive_rejects_links(tmp_path):
    archive = tmp_path / "sample.tar.gz"
    make_archive(archive, "Arshasb_samples/00001/page_00001.png", link=True)
    with pytest.raises(ValueError, match="ordinary files"):
        import_archive(archive, tmp_path / "output")


def test_archive_import_is_repeatable(tmp_path):
    archive = tmp_path / "sample.tar.gz"
    make_archive(archive, "Arshasb_samples/00001/fulltext_00001.txt", "سلام".encode())
    first = import_archive(archive, tmp_path / "output")
    second = import_archive(archive, tmp_path / "output")
    assert first == second
    assert first["pages"] == 1
    assert (tmp_path / "output/00001/fulltext_00001.txt").read_text() == "سلام"


def test_import_preserves_existing_modified_files(tmp_path):
    target = tmp_path / "original.txt"
    target.write_bytes(b"user work")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_preserving(target, b"different")
    assert target.read_bytes() == b"user work"
