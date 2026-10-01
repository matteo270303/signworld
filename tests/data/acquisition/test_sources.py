import zipfile
from pathlib import Path
from unittest import mock

import pytest

from signworld.data.acquisition.config import AcquisitionConfig
from signworld.data.acquisition.ledger import read_ledgers
from signworld.data.acquisition.outcome import Outcome, Status
from signworld.data.acquisition.sharding import Shard
from signworld.data.acquisition.sources import build_source
from signworld.data.acquisition.sources.base import (
    Access,
    AccessRequiredError,
    DatasetSource,
    NoSettings,
)
from signworld.data.acquisition.sources.csl_news import CSLNewsSettings, CSLNewsSource
from signworld.data.acquisition.sources.openasl import (
    COLUMNS,
    OpenASLSettings,
    OpenASLSource,
    read_video_ids,
)
from signworld.data.acquisition.sources.youtube_sl25 import (
    VideoEntry,
    download_order,
    interleave_by_language,
    read_metadata,
)


class _ScriptedSource(DatasetSource[NoSettings]):
    name = "scripted"
    homepage = "https://example.org"
    terms = "test"
    access = Access.PUBLIC
    settings_model = NoSettings

    def __init__(self, config: AcquisitionConfig, script: dict[str, Status | Exception]) -> None:
        super().__init__(NoSettings(), config)
        self.script = script
        self.calls: list[str] = []

    def fetch_metadata(self) -> None:
        return None

    def media_keys(self) -> list[str]:
        return list(self.script)

    def fetch_item(self, key: str) -> Outcome:
        self.calls.append(key)
        result = self.script[key]
        if isinstance(result, Exception):
            raise result
        return Outcome(key, result)


def _no_sleep(seconds: float) -> None:
    return None


def test_settled_items_are_skipped_and_failures_retried(config: AcquisitionConfig) -> None:
    source = _ScriptedSource(
        config, {"a": Status.DONE, "b": Status.FAILED, "c": Status.UNAVAILABLE}
    )
    first = source.fetch_media(Shard.whole())
    source.calls.clear()

    second = source.fetch_media(Shard.whole())

    assert source.calls == ["b"]
    assert (first.remaining, second.remaining) == (1, 0)
    assert second.complete


def test_failures_are_abandoned_after_the_maximum_attempts(config: AcquisitionConfig) -> None:
    source = _ScriptedSource(config, {"a": Status.FAILED})
    for _ in range(config.max_attempts):
        source.fetch_media(Shard.whole())
    source.calls.clear()

    source.fetch_media(Shard.whole())

    assert source.calls == []


def test_refusals_pause_with_doubling_cooldowns_then_stop(config: AcquisitionConfig) -> None:
    script = dict.fromkeys(("a", "b", "c", "d"), Status.BLOCKED)
    source = _ScriptedSource(config, script)
    pauses: list[float] = []

    report = source.fetch_media(Shard.whole(), sleep=pauses.append)

    assert source.calls == ["a", "b", "c"]
    assert pauses == [10.0, 20.0]
    assert report.stopped_early and report.refused
    assert report.remaining == 4


def test_a_success_resets_the_refusal_streak(config: AcquisitionConfig) -> None:
    script = {"a": Status.BLOCKED, "b": Status.BLOCKED, "c": Status.DONE, "d": Status.BLOCKED}
    source = _ScriptedSource(config, script)

    report = source.fetch_media(Shard.whole(), sleep=_no_sleep)

    assert not report.stopped_early
    assert source.calls == ["a", "b", "c", "d"]


def test_limit_caps_the_items_of_a_run(config: AcquisitionConfig) -> None:
    source = _ScriptedSource(config, dict.fromkeys(("a", "b", "c"), Status.DONE))

    report = source.fetch_media(Shard.whole(), limit=2)

    assert source.calls == ["a", "b"]
    assert report.remaining == 1


def test_a_run_stops_when_its_time_budget_is_spent(config: AcquisitionConfig) -> None:
    budgeted = config.model_copy(update={"run_budget_s": 30.0})
    source = _ScriptedSource(budgeted, dict.fromkeys(("a", "b", "c"), Status.DONE))
    clock = iter([0.0, 10.0, 60.0])
    source.fetch_item = lambda key: (  # type: ignore[method-assign]
        source.calls.append(key) or Outcome(key, Status.DONE)
    )

    with mock.patch("signworld.data.acquisition.sources.base.monotonic", lambda: next(clock)):
        report = source.fetch_media(Shard.whole())

    assert source.calls == ["a"]
    assert report.stopped_early and not report.refused
    assert report.remaining == 2


def test_unexpected_error_is_recorded_and_the_shard_continues(config: AcquisitionConfig) -> None:
    source = _ScriptedSource(config, {"a": RuntimeError("boom"), "b": Status.DONE})

    report = source.fetch_media(Shard.whole())

    assert report.counts == {Status.FAILED: 1, Status.DONE: 1}
    assert "RuntimeError: boom" in read_ledgers(source.layout.ledgers)["a"].detail


def test_items_settled_under_another_sharding_are_not_fetched_again(
    config: AcquisitionConfig,
) -> None:
    source = _ScriptedSource(config, {f"key-{index}": Status.DONE for index in range(20)})
    source.fetch_media(Shard.whole())
    source.calls.clear()

    for index in range(4):
        source.fetch_media(Shard(index, 4))

    assert source.calls == []


def test_manual_source_explains_how_to_obtain_the_data(config: AcquisitionConfig) -> None:
    with pytest.raises(AccessRequiredError, match="Release Agreement"):
        build_source("csl_daily", config).fetch_media(Shard.whole())


def test_bobsl_requires_credentials(
    config: AcquisitionConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BOBSL_USERNAME", raising=False)
    monkeypatch.delenv("BOBSL_PASSWORD", raising=False)

    with pytest.raises(AccessRequiredError, match="BOBSL_USERNAME"):
        build_source("bobsl", config).fetch_metadata()


def test_youtube_sl25_metadata_marks_unknown_languages(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text("Bdj5MUf_3Hc,ase\nsIo8170zyMw,???\n", encoding="utf-8")

    assert read_metadata(path) == [
        VideoEntry("Bdj5MUf_3Hc", "ase"),
        VideoEntry("sIo8170zyMw", None),
    ]


def test_youtube_sl25_metadata_rejects_malformed_rows(tmp_path: Path) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text("Bdj5MUf_3Hc,ase,extra\n", encoding="utf-8")

    with pytest.raises(ValueError, match=":1:"):
        read_metadata(path)


def _openasl_copy(root: Path, video_ids: list[str], present: list[str]) -> None:
    rows = ["\t".join(COLUMNS)] + [
        f'{video_id}-{index}\t{video_id}\t00:00:01.000\t00:00:02.000\tSay "hi"\thi\t\ttrain'
        for index, video_id in enumerate(video_ids)
    ]
    annotation = root / "data" / "openasl-v1.0.tsv"
    annotation.parent.mkdir(parents=True)
    annotation.write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "raw-video").mkdir()
    for video_id in present:
        (root / "raw-video" / f"{video_id}.mp4").write_bytes(b"video")


def test_openasl_video_ids_are_unique_and_ordered(tmp_path: Path) -> None:
    _openasl_copy(tmp_path, ["yidA", "yidB", "yidA"], present=[])

    assert read_video_ids(tmp_path / "data" / "openasl-v1.0.tsv") == ["yidA", "yidB"]


def test_openasl_rejects_an_unexpected_header(tmp_path: Path) -> None:
    path = tmp_path / "openasl.tsv"
    path.write_text("vid\tyid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected header"):
        read_video_ids(path)


def test_openasl_local_copy_reports_missing_videos(
    config: AcquisitionConfig, tmp_path: Path
) -> None:
    root = tmp_path / "openASL"
    _openasl_copy(root, ["yidA", "yidB"], present=["yidA"])
    source = OpenASLSource(OpenASLSettings(local_root=root), config)

    report = source.fetch_media(Shard.whole())

    assert report.counts == {Status.DONE: 1, Status.UNAVAILABLE: 1}


class _LocalHub:
    def __init__(self, archive: Path) -> None:
        self.archive = archive

    def fetch(self, path: str, local_dir: Path) -> Path:
        return self.archive


def test_csl_news_archive_is_extracted_and_optionally_removed(
    config: AcquisitionConfig, tmp_path: Path
) -> None:
    archive = tmp_path / "archive_001.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("clips/a.mp4", b"a")
        bundle.writestr("clips/b.mp4", b"b")
    source = CSLNewsSource(
        CSLNewsSettings(keep_archives=False),
        config,
        hub=_LocalHub(archive),  # type: ignore[arg-type]
    )

    outcome = source.fetch_item("archive_001.zip")

    assert outcome.status is Status.DONE
    assert sorted(path.name for path in source.clips_dir.iterdir()) == ["a.mp4", "b.mp4"]
    assert not archive.exists()


def test_youtube_sl25_keys_interleave_languages_and_keep_every_video() -> None:
    entries = [
        VideoEntry("a1", "ase"),
        VideoEntry("a2", "ase"),
        VideoEntry("a3", "ase"),
        VideoEntry("b1", "bfi"),
        VideoEntry("a1", "ase"),
        VideoEntry("u1", None),
    ]

    assert interleave_by_language(entries) == ["a1", "b1", "u1", "a2", "a3"]


def test_first_languages_lead_the_download_order_without_losing_videos() -> None:
    entries = [
        VideoEntry("b1", "bfi"),
        VideoEntry("a1", "ase"),
        VideoEntry("g1", "gsg"),
        VideoEntry("a2", "ase"),
        VideoEntry("b2", "bfi"),
    ]

    order = download_order(entries, first=["ase"])

    assert set(order[:2]) == {"a1", "a2"}
    assert sorted(order) == sorted({entry.video_id for entry in entries})
    assert download_order(entries, first=["ase"]) == order


def test_download_order_spreads_a_language_over_its_channels() -> None:
    # The release lists each channel's videos together; here, 100 per channel.
    entries = [
        VideoEntry(f"c{channel}-{video}", "ase") for channel in range(5) for video in range(100)
    ]

    prefix = download_order(entries, first=["ase"])[:50]

    assert len({video_id.split("-")[0] for video_id in prefix}) == 5
