import pytest

from signworld.data.acquisition.adopt import adopt_downloads
from signworld.data.acquisition.config import AcquisitionConfig
from signworld.data.acquisition.ledger import read_ledgers
from signworld.data.acquisition.outcome import Status
from signworld.data.acquisition.sources import build_source


def _youtube(config: AcquisitionConfig, ids: list[str]):
    source = build_source("youtube_sl25", config)
    source.layout.metadata.mkdir(parents=True)
    rows = "".join(f"{video_id},ase\n" for video_id in ids)
    (source.layout.metadata / source.METADATA_FILE).write_text(rows)
    source.layout.raw.joinpath("videos").mkdir(parents=True)
    return source


def test_downloads_on_disk_are_recorded_and_status_counts_them(config: AcquisitionConfig) -> None:
    ids = ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc", "ddddddddddd"]
    source = _youtube(config, ids)
    videos = source.layout.raw / "videos"
    for video_id in ids[:2]:
        (videos / f"{video_id}.mp4").write_bytes(b"x")
        (videos / f"{video_id}.info.json").write_text("{}")
    (videos / f"{ids[2]}.mp4").write_bytes(b"x")  # no info: left to fetch-media
    gone = source.layout.root / "failed_unavailable.txt"
    gone.write_text(f"{ids[3]}\nzzzzzzzzzzz\n")

    result = adopt_downloads(source, gone)

    assert (result.done, result.unavailable, result.without_info) == (2, 1, 1)
    ledger = read_ledgers(source.layout.ledgers)
    assert {key: entry.status for key, entry in ledger.items()} == {
        ids[0]: Status.DONE,
        ids[1]: Status.DONE,
        ids[3]: Status.UNAVAILABLE,
    }


def test_adopting_twice_changes_nothing(config: AcquisitionConfig) -> None:
    source = _youtube(config, ["aaaaaaaaaaa"])
    videos = source.layout.raw / "videos"
    (videos / "aaaaaaaaaaa.mp4").write_bytes(b"x")
    (videos / "aaaaaaaaaaa.info.json").write_text("{}")

    adopt_downloads(source)
    again = adopt_downloads(source)

    assert (again.done, again.already_recorded) == (0, 1)
    assert len(read_ledgers(source.layout.ledgers)) == 1


def test_only_youtube_downloads_can_be_adopted(config: AcquisitionConfig) -> None:
    with pytest.raises(ValueError, match="only defined for YouTube"):
        adopt_downloads(build_source("csl_daily", config))
