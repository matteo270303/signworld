"""Safe, idempotent extraction of zip and tar archives."""

import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Final

_COPY_BUFFER_BYTES: Final = 8 * 1024 * 1024


class ArchiveError(RuntimeError):
    """An archive is malformed or would write outside its destination."""


def extract_zip_flat(archive: Path, destination: Path) -> int:
    """Extract every file of ``archive`` directly into ``destination``, dropping folders.

    Equivalent to ``unzip -j``. Files already present with the expected size are kept, so an
    interrupted extraction resumes. Returns the number of files written.
    """
    destination.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(archive) as bundle:
        members = [info for info in bundle.infolist() if not info.is_dir()]
        names = [PurePosixPath(info.filename).name for info in members]
        if len(set(names)) != len(names):
            raise ArchiveError(
                f"{archive}: files in different folders share a name; "
                "flat extraction would overwrite them"
            )
        for info, name in zip(members, names, strict=True):
            target = destination / name
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            partial = target.with_name(f"{name}.part")
            with bundle.open(info) as source, partial.open("wb") as sink:
                shutil.copyfileobj(source, sink, _COPY_BUFFER_BYTES)
            partial.replace(target)
            written += 1
    return written


def extract_tar(archive: Path, destination: Path) -> bool:
    """Extract ``archive`` into ``destination`` once; return ``False`` if already done.

    Members that would escape ``destination`` (absolute paths, ``..``, outside links) are
    rejected by the standard library's ``data`` filter.
    """
    marker = destination / f".extracted-{archive.name}"
    if marker.exists():
        return False
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive) as bundle:
            bundle.extractall(destination, filter="data")
    except tarfile.FilterError as error:
        raise ArchiveError(f"{archive}: unsafe member rejected ({error})") from error
    marker.touch()
    return True
