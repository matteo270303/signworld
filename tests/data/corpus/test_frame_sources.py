"""CSL-Daily and PHOENIX-2014-T: frame folders become videos, then clips with official splits."""

import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from signworld.data.acquisition.config import AcquisitionConfig
from signworld.data.acquisition.outcome import Status
from signworld.data.acquisition.sources.base import AccessRequiredError
from signworld.data.acquisition.sources.csl_daily import CSLDailySettings, CSLDailySource
from signworld.data.acquisition.sources.phoenix14t import Phoenix14TSettings, Phoenix14TSource
from signworld.data.corpus.builders import CSLDailyManifest, Phoenix14TManifest
from signworld.data.corpus.manifest import clips_of, read_manifest

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def write_frames(folder: Path, count: int, name: str, size: tuple[int, int] = (64, 48)) -> None:
    folder.mkdir(parents=True)
    for index in range(count):
        frame = np.full((size[1], size[0], 3), index * 10, dtype=np.uint8)
        cv2.imwrite(str(folder / name.format(index)), frame)


def make_csl(root: Path) -> None:
    labels = [
        {"name": "S0_P0_T00", "length": 6, "label_char": ["你", "好"], "signer": 3},
        {"name": "S1_P0_T00", "length": 4, "label_char": ["谢", "谢"], "signer": 5},
        {"name": "S2_P0_T00", "length": 4, "label_char": ["再", "见"], "signer": 5},
    ]
    (root / "sentence_label").mkdir(parents=True)
    (root / "sentence_label/labels.pkl").write_bytes(pickle.dumps({"info": labels}))
    (root / "sentence_label/split.txt").write_text(
        "name|split\nS0_P0_T00|train\nS1_P0_T00|dev\n", encoding="utf-8"
    )
    write_frames(root / "frames/S0_P0_T00", 6, "{:06d}.jpg")
    write_frames(root / "frames/S1_P0_T00", 4, "{:06d}.jpg")


def csl_source(config: AcquisitionConfig, root: Path) -> CSLDailySource:
    settings = CSLDailySettings(
        local_root=root,
        labels_file="sentence_label/labels.pkl",
        split_file="sentence_label/split.txt",
        frames_dir="frames",
    )
    return CSLDailySource(settings, config)


def test_csl_without_the_release_explains_how_to_get_it(config: AcquisitionConfig) -> None:
    source = CSLDailySource(CSLDailySettings(), config)

    assert source.media_keys() == []
    with pytest.raises(AccessRequiredError, match="Release Agreement"):
        source.check_access()


@needs_ffmpeg
def test_csl_fetch_encodes_the_frames_and_the_manifest_follows(
    config: AcquisitionConfig, tmp_path: Path
) -> None:
    make_csl(tmp_path / "release")
    source = csl_source(config, tmp_path / "release")

    assert source.media_keys() == ["S0_P0_T00", "S1_P0_T00"]  # S2 has no official split
    assert source.fetch_item("S0_P0_T00").status is Status.DONE
    video = source.layout.raw / "videos" / "S0_P0_T00.mp4"
    assert cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT) == 6
    assert source.fetch_item("S1_P0_T00").status is Status.DONE
    video.unlink()

    clips = {c.clip_id: c for c in clips_of(read_manifest(CSLDailyManifest(source).build()))}

    assert clips["S0_P0_T00"].video_available is False
    assert clips["S1_P0_T00"].video_available is True
    assert clips["S1_P0_T00"].split == "valid"  # the release calls it dev
    assert clips["S0_P0_T00"].caption == "你好"
    assert clips["S0_P0_T00"].end_s == pytest.approx(6 / 30)
    assert clips["S1_P0_T00"].channel_id == "signer:5"
    assert (clips["S1_P0_T00"].caption_language, clips["S1_P0_T00"].sign_language) == ("zh", "csl")


def test_csl_reports_a_missing_frame_folder(config: AcquisitionConfig, tmp_path: Path) -> None:
    make_csl(tmp_path / "release")
    shutil.rmtree(tmp_path / "release/frames/S1_P0_T00")

    outcome = csl_source(config, tmp_path / "release").fetch_item("S1_P0_T00")

    assert outcome.status is Status.UNAVAILABLE


def make_phoenix(root: Path) -> None:
    annotations = root / "annotations/manual"
    annotations.mkdir(parents=True)
    header = "name|video|start|end|speaker|orth|translation\n"
    rows = {
        "train": "a-1|a-1/1/*.png|-1|-1|Signer01|REGEN|es regnet .\n",
        "dev": "b-1|b-1/1/*.png|-1|-1|Signer05|WIND|es weht .\n",
        "test": "c-1|c-1/1/*.png|-1|-1|Signer04|SONNE|die sonne scheint .\n",
    }
    for official, row in rows.items():
        (annotations / f"PHOENIX-2014-T.{official}.corpus.csv").write_text(
            header + row, encoding="utf-8"
        )
        write_frames(root / "frames" / official / row.split("|")[0], 5, "images{:04d}.png")


def phoenix_source(config: AcquisitionConfig, root: Path) -> Phoenix14TSource:
    return Phoenix14TSource(Phoenix14TSettings(local_root=root, frames_dir="frames"), config)


def test_phoenix_without_a_local_copy_still_downloads_the_archive(
    config: AcquisitionConfig,
) -> None:
    assert Phoenix14TSource(Phoenix14TSettings(), config).media_keys() == [
        "phoenix-2014-T.v3.tar.gz"
    ]


@needs_ffmpeg
def test_phoenix_local_copy_is_encoded_and_listed_with_official_splits(
    config: AcquisitionConfig, tmp_path: Path
) -> None:
    make_phoenix(tmp_path / "release")
    source = phoenix_source(config, tmp_path / "release")

    assert source.media_keys() == ["a-1", "b-1", "c-1"]
    for key in source.media_keys():
        assert source.fetch_item(key).status is Status.DONE

    clips = {c.clip_id: c for c in clips_of(read_manifest(Phoenix14TManifest(source).build()))}

    assert {k: c.split for k, c in clips.items()} == {"a-1": "train", "b-1": "valid", "c-1": "test"}
    assert all(c.video_available for c in clips.values())
    assert clips["b-1"].caption == "es weht ."
    assert clips["b-1"].end_s == pytest.approx(5 / 25)
    assert clips["c-1"].channel_id == "signer:Signer04"
    assert (clips["a-1"].caption_language, clips["a-1"].sign_language) == ("de", "gsg")
