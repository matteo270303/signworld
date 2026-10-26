"""Frame folders to one video: the pipeline reads mp4, some releases ship JPG or PNG frames."""

import subprocess
from pathlib import Path
from typing import Final

FRAME_SUFFIXES: Final = (".jpg", ".png")


class FrameEncodingError(RuntimeError):
    """ffmpeg could not turn the folder into a video."""


def frame_count(folder: Path) -> int:
    return sum(1 for path in folder.iterdir() if path.suffix.lower() in FRAME_SUFFIXES)


def encode_frames(folder: Path, fps: float, target: Path, suffix: str) -> Path:
    """Encode the ``suffix`` frames of ``folder``, in name order, as H.264 at ``fps``.

    Near lossless (crf 14): the crop and the poses are made from the decoded video, so the
    encoding must not be what the model learns. Written under a temporary name and renamed,
    so an interrupted run never leaves a file that looks complete.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.stem}.partial{target.suffix}")
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob", "-i", str(folder / f"*{suffix}"),
        "-c:v", "libx264", "-crf", "14", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(partial),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise FrameEncodingError(f"{folder}: {result.stderr.strip()[:300]}")
    partial.rename(target)
    return target
