"""Contact sheets: frames of a clip with the articulator boxes drawn, for inspection by eye."""

from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image, ImageDraw

from signworld.data.pose.boxes import articulator_boxes
from signworld.data.pose.wholebody import Articulator, PoseTrack
from signworld.data.video import ClipReader

COLOURS: Final[dict[Articulator, tuple[int, int, int]]] = {
    Articulator.BODY: (160, 160, 160),
    Articulator.LEFT_HAND: (37, 99, 235),
    Articulator.RIGHT_HAND: (245, 158, 11),
    Articulator.FACE: (16, 185, 129),
}


def contact_sheet(video: Path, pose: Path, frames: int = 8, columns: int = 4) -> Image.Image:
    """``frames`` evenly spaced pose frames, boxes drawn, laid out ``columns`` per row."""
    track = PoseTrack.load(pose)
    boxes = articulator_boxes(track)
    chosen = np.linspace(0, len(track.frame_indices) - 1, frames).round().astype(int)
    images = ClipReader(video).frames(track.frame_indices[chosen])
    height, width = images.shape[1:3]
    rows = -(-frames // columns)
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for tile, (position, image) in enumerate(zip(chosen, images, strict=True)):
        canvas = Image.fromarray(image)
        draw = ImageDraw.Draw(canvas)
        for column, part in enumerate(Articulator):
            if not boxes.visible[position, column]:
                continue
            x0, y0, x1, y1 = boxes.boxes[position, column] * np.array([width, height] * 2)
            draw.rectangle((x0, y0, x1, y1), outline=COLOURS[part], width=2)
        sheet.paste(canvas, ((tile % columns) * width, (tile // columns) * height))
    return sheet
