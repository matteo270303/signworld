import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yt_dlp

from signworld.data.acquisition.config import YouTubeConfig
from signworld.data.acquisition.outcome import Status
from signworld.data.acquisition.transfer.youtube import YouTubeDownloader, classify_error


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("ERROR: [youtube] a: Video unavailable. This video has been removed", Status.UNAVAILABLE),
        (
            "ERROR: [youtube] a: Private video. Sign in if you've been granted access",
            Status.UNAVAILABLE,
        ),
        (
            "ERROR: [youtube] a: Sign in to confirm you\N{RIGHT SINGLE QUOTATION MARK}re not a bot",
            Status.BLOCKED,
        ),
        ("ERROR: [youtube] a: Requested format is not available", Status.BLOCKED),
        ("ERROR: unable to download video data: HTTP Error 429: Too Many Requests", Status.BLOCKED),
        ("ERROR: [download] Got error: The read operation timed out", Status.FAILED),
        ("ERROR: Unable to download API page: [Errno 101] Network is unreachable", Status.BLOCKED),
    ],
)
def test_errors_are_classified_by_what_retrying_can_achieve(message: str, expected: Status) -> None:
    assert classify_error(message) is expected


class _FakeClient:
    def __init__(self, options: dict[str, Any], results: list[dict[str, Any] | Exception]) -> None:
        self.options = options
        self.closed = False
        self._results = results

    def extract_info(self, url: str, download: bool) -> dict[str, Any]:
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        self.closed = True


class _FakeFactory:
    """Builds fake clients that share one script of results, and remembers every client."""

    def __init__(self, *results: dict[str, Any] | Exception) -> None:
        self.results = list(results)
        self.clients: list[_FakeClient] = []

    def __call__(self, options: dict[str, Any]) -> _FakeClient:
        client = _FakeClient(options, self.results)
        self.clients.append(client)
        return client


def _downloader(tmp_path: Path, factory: _FakeFactory, **overrides: Any) -> YouTubeDownloader:
    config = YouTubeConfig(**overrides)
    return YouTubeDownloader(
        config, tmp_path / "videos", with_subtitles=True, client_factory=factory
    )


def test_options_follow_the_configuration(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    factory = _FakeFactory()

    with _downloader(
        tmp_path,
        factory,
        cookies_file=cookies,
        pot_server_home=tmp_path / "server",
        max_height=480,
        ip_version=6,
    ):
        options = factory.clients[0].options
        private_cookies = Path(options["cookiefile"]).read_text(encoding="utf-8")

    assert private_cookies == cookies.read_text(encoding="utf-8")
    assert options["source_address"] == "::"
    assert options["extractor_args"] == {
        "youtubepot-bgutilscript": {"server_home": [str(tmp_path / "server")]}
    }
    assert options["format"].startswith("bv*[vcodec^=avc1][height<=480][fps<=30]/")
    assert options["writesubtitles"] is True
    assert options["writeautomaticsub"] is False


def test_clients_use_a_private_copy_of_the_cookies_removed_at_exit(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("original", encoding="utf-8")
    factory = _FakeFactory()

    with _downloader(tmp_path, factory, cookies_file=cookies):
        copy = Path(factory.clients[0].options["cookiefile"])
        copy.write_text("rotated by yt-dlp", encoding="utf-8")

    assert cookies.read_text(encoding="utf-8") == "original"
    assert not copy.exists()
    assert factory.clients[0].closed


def test_successful_download_writes_a_compact_info_file(tmp_path: Path) -> None:
    info = {
        "id": "abc",
        "channel_id": "UC1",
        "duration": 12.5,
        "formats": ["large"],
        "requested_subtitles": {"sv": {}, "da": {}},
    }

    with _downloader(tmp_path, _FakeFactory(info)) as downloader:
        outcome = downloader.download("abc")

    written = json.loads((tmp_path / "videos" / "abc.info.json").read_text(encoding="utf-8"))
    assert outcome.status is Status.DONE
    assert written["channel_id"] == "UC1"
    assert written["subtitle_languages"] == ["da", "sv"]
    assert "formats" not in written


def test_unavailable_video_keeps_the_client(tmp_path: Path) -> None:
    factory = _FakeFactory(yt_dlp.utils.DownloadError("ERROR: [youtube] abc: Private video"))

    with _downloader(tmp_path, factory) as downloader:
        outcome = downloader.download("abc")

    assert outcome.status is Status.UNAVAILABLE
    assert "Private video" in outcome.detail
    assert len(factory.clients) == 1


def test_refusal_rebuilds_the_client(tmp_path: Path) -> None:
    factory = _FakeFactory(yt_dlp.utils.DownloadError("ERROR: HTTP Error 429: Too Many Requests"))

    with _downloader(tmp_path, factory) as downloader:
        outcome = downloader.download("abc")

    assert outcome.status is Status.BLOCKED
    assert len(factory.clients) == 2
    assert factory.clients[0].closed


def test_token_provider_timeout_is_retried_with_a_fresh_client(tmp_path: Path) -> None:
    timeout = subprocess.TimeoutExpired(["deno", "run", "generate_once.ts", "--version"], 15)
    factory = _FakeFactory(timeout, {"id": "abc"})

    with _downloader(tmp_path, factory) as downloader:
        outcome = downloader.download("abc")

    assert outcome.status is Status.DONE
    assert len(factory.clients) == 2


def test_retry_sleep_matches_the_keyword_yt_dlp_passes(tmp_path: Path) -> None:
    factory = _FakeFactory()

    with _downloader(tmp_path, factory):
        sleep = factory.clients[0].options["retry_sleep_functions"]["http"]

    assert yt_dlp.utils.float_or_none(sleep(n=3)) == 8.0


def test_download_outside_the_context_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="context"):
        _downloader(tmp_path, _FakeFactory()).download("abc")
