"""The YouTube probe tells account refusals from network ones."""

from pathlib import Path
from typing import Any

import yt_dlp

from signworld.acquisition.config import YouTubeConfig
from signworld.acquisition.outcome import Status
from signworld.acquisition.probe import probe


class _Network:
    """A YouTube that refuses anonymous IPv6 and says the cookies were rotated."""

    def __init__(self, options: dict[str, Any]) -> None:
        self._options = options

    def extract_info(self, url: str, download: bool) -> dict[str, Any]:
        assert not download
        if "cookiefile" in self._options:
            self._options["logger"].warning(
                "The provided YouTube account cookies are no longer valid. They have likely "
                "been rotated in the browser as a security measure."
            )
            return {}
        if self._options.get("source_address") == "::":
            raise yt_dlp.utils.DownloadError("ERROR: Sign in to confirm you're not a bot")
        return {}

    def close(self) -> None:
        pass


def test_each_variant_reports_its_own_outcome(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")

    results = probe(YouTubeConfig(cookies_file=cookies), ["a"], pause_s=0, client_factory=_Network)

    by_variant = {result.variant: result for result in results}
    assert by_variant["anonymous-ipv6"].status is Status.BLOCKED
    assert by_variant["anonymous-ipv4"].status is Status.DONE
    assert by_variant["cookies-ipv6"].cookies_rejected
    assert not by_variant["anonymous-ipv4"].cookies_rejected


def test_variants_with_cookies_are_skipped_without_a_cookies_file() -> None:
    results = probe(YouTubeConfig(), ["a", "b"], pause_s=0, client_factory=_Network)

    assert {result.variant for result in results} == {"anonymous-ipv6", "anonymous-ipv4"}
    assert len(results) == 4
