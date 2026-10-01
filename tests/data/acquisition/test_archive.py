import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from signworld.data.acquisition.transfer.archive import ArchiveError, extract_tar, extract_zip_flat


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return path


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as bundle:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    return path


def test_zip_members_are_extracted_flat_and_only_once(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "a.zip", {"x/one.mp4": b"1", "y/z/two.mp4": b"22"})
    destination = tmp_path / "clips"

    assert extract_zip_flat(archive, destination) == 2
    assert sorted(path.name for path in destination.iterdir()) == ["one.mp4", "two.mp4"]
    assert extract_zip_flat(archive, destination) == 0


def test_zip_with_colliding_names_is_rejected(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "a.zip", {"x/clip.mp4": b"1", "y/clip.mp4": b"2"})

    with pytest.raises(ArchiveError, match="share a name"):
        extract_zip_flat(archive, tmp_path / "clips")


def test_tar_is_extracted_once(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "corpus.tar.gz", {"corpus/annotations.csv": b"id|text"})
    destination = tmp_path / "extracted"

    assert extract_tar(archive, destination) is True
    assert (destination / "corpus" / "annotations.csv").read_bytes() == b"id|text"
    assert extract_tar(archive, destination) is False


def test_tar_member_escaping_the_destination_is_rejected(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "evil.tar.gz", {"../escaped.txt": b"x"})

    with pytest.raises(ArchiveError, match="unsafe"):
        extract_tar(archive, tmp_path / "extracted")

    assert not (tmp_path / "escaped.txt").exists()
