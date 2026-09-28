import json
from pathlib import Path

from signworld.acquisition.config import AcquisitionConfig
from signworld.acquisition.ledger import Ledger
from signworld.acquisition.outcome import Outcome, Status
from signworld.acquisition.sources.openasl import COLUMNS, OpenASLSettings, OpenASLSource
from signworld.acquisition.sources.youtube_sl25 import YouTubeSL25Settings, YouTubeSL25Source
from signworld.corpus.builders import OpenASLManifest, YouTubeSL25Manifest
from signworld.corpus.manifest import clips_of, read_manifest


def test_openasl_manifest_keeps_splits_times_and_availability(
    config: AcquisitionConfig, tmp_path: Path
) -> None:
    root = tmp_path / "openASL"
    (root / "data").mkdir(parents=True)
    (root / "raw-video").mkdir()
    (root / "raw-video" / "yidA.mp4").write_bytes(b"")
    rows = [
        "\t".join(COLUMNS),
        'yidA-1\tyidA\t00:00:06.000\t00:00:06.589\tSay "hi"!\tsay hi\t\ttrain',
        "yidB-1\tyidB\t00:01:00.000\t00:01:02.500\tBye.\tbye\t\ttest",
    ]
    (root / "data" / "openasl-v1.0.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    source = OpenASLSource(OpenASLSettings(local_root=root), config)

    clips = clips_of(read_manifest(OpenASLManifest(source).build()))

    assert [(c.clip_id, c.split, c.video_available) for c in clips] == [
        ("yidA-1", "train", True),
        ("yidB-1", "test", False),
    ]
    assert clips[0].caption == 'Say "hi"!'
    assert clips[1].duration_s == 2.5


class _NoDownloads:
    def __enter__(self) -> "_NoDownloads":
        return self

    def __exit__(self, *exception: object) -> None:
        return None


def test_youtube_manifest_keeps_only_the_track_in_the_videos_own_language(
    config: AcquisitionConfig,
) -> None:
    source = YouTubeSL25Source(YouTubeSL25Settings(), config, downloader=_NoDownloads())  # type: ignore[arg-type]
    layout = source.layout
    layout.metadata.mkdir(parents=True)
    (layout.metadata / source.METADATA_FILE).write_text(
        "vidA,ase\nvidB,???\nvidC,dsl\n", encoding="utf-8"
    )
    videos = layout.raw / "videos"
    videos.mkdir(parents=True)
    for video_id in ("vidA", "vidB"):
        (videos / f"{video_id}.info.json").write_text(
            json.dumps({"channel_id": f"UC-{video_id}"}), encoding="utf-8"
        )
    cue = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n\n00:00:03.000 --> 00:00:04.000\nWorld\n"
    # vidA is ASL with an English track and a French translation; vidB has an unknown
    # sign language, so no track counts as its own.
    (videos / "vidA.en.vtt").write_text(cue, encoding="utf-8")
    (videos / "vidA.fr.vtt").write_text(cue, encoding="utf-8")
    (videos / "vidB.en.vtt").write_text(cue, encoding="utf-8")
    ledger = Ledger(layout.ledgers / "shard.jsonl")
    ledger.record(Outcome("vidA", Status.DONE))
    ledger.record(Outcome("vidB", Status.DONE))
    ledger.record(Outcome("vidC", Status.UNAVAILABLE))

    clips = clips_of(read_manifest(YouTubeSL25Manifest(source).build()))

    assert len(clips) == 2
    assert {(c.video_id, c.caption_language) for c in clips} == {("vidA", "en")}
    first = clips[0]
    assert (first.clip_id, first.sign_language, first.channel_id) == (
        "vidA.en.00000",
        "ase",
        "UC-vidA",
    )
