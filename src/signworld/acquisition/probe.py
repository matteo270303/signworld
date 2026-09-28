"""Which way out still reaches YouTube: with or without cookies, over IPv4 or IPv6.

A refusal alone does not say whether YouTube objects to the account or to the network. One
metadata request per video and variant, from the node the probe runs on, separates the two,
and yt-dlp's own warning tells whether YouTube still accepted the cookies.
"""

import logging
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import yt_dlp

from .config import YouTubeConfig
from .outcome import Status
from .transfer import youtube
from .transfer.youtube import ClientFactory, YouTubeDownloader

_ROTATED_COOKIES: Final = "cookies are no longer valid"
_DETAIL_CHARS: Final = 300


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    cookies: bool
    ip_version: Literal[4, 6]


VARIANTS: Final = (
    Variant("cookies-ipv6", cookies=True, ip_version=6),
    Variant("cookies-ipv4", cookies=True, ip_version=4),
    Variant("anonymous-ipv6", cookies=False, ip_version=6),
    Variant("anonymous-ipv4", cookies=False, ip_version=4),
)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    variant: str
    video_id: str
    status: Status
    cookies_rejected: bool
    """yt-dlp warned that YouTube no longer accepts the cookies (rotated or revoked)."""
    detail: str


def probe(
    config: YouTubeConfig,
    video_ids: Sequence[str],
    variants: Sequence[Variant] = VARIANTS,
    pause_s: float = 10.0,
    client_factory: ClientFactory = yt_dlp.YoutubeDL,
) -> list[ProbeResult]:
    """Check every video under every variant, pausing between requests.

    Variants with cookies are skipped when the configuration has no cookies file.
    """
    results = []
    with tempfile.TemporaryDirectory(prefix="signworld-probe-") as scratch:
        for variant in variants:
            if variant.cookies and config.cookies_file is None:
                continue
            settings = config.model_copy(
                update={
                    "cookies_file": config.cookies_file if variant.cookies else None,
                    "ip_version": variant.ip_version,
                }
            )
            downloader = YouTubeDownloader(
                settings, Path(scratch), with_subtitles=False, client_factory=client_factory
            )
            with downloader:
                for video_id in video_ids:
                    with _warnings() as seen:
                        outcome = downloader.check(video_id)
                    results.append(
                        ProbeResult(
                            variant.name,
                            video_id,
                            outcome.status,
                            any(_ROTATED_COOKIES in message.lower() for message in seen),
                            outcome.detail[:_DETAIL_CHARS],
                        )
                    )
                    time.sleep(pause_s)
    return results


class _Collector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextmanager
def _warnings() -> Iterator[list[str]]:
    """The warnings yt-dlp forwards to the downloader's logger while the block runs."""
    collector = _Collector()
    youtube.logger.addHandler(collector)
    try:
        yield collector.messages
    finally:
        youtube.logger.removeHandler(collector)
