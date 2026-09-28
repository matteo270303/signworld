"""YouTube downloads through yt-dlp, reported as outcomes a ledger can act on."""

import json
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Protocol, Self

import yt_dlp

from ..config import YouTubeConfig
from ..outcome import Outcome, Status
from ..pacing import DeadlineExceededError, deadline

logger = logging.getLogger(__name__)

_WATCH_URL: Final = "https://www.youtube.com/watch?v={}"
_POT_PROVIDER_ARGUMENTS: Final = "youtubepot-bgutilscript"
_POT_SCRIPT: Final = "generate_once.ts"
_WARM_UP_TIMEOUT_S: Final = 600.0
_MAX_RETRY_SLEEP_S: Final = 60.0
# Binding to the wildcard address of a family is how yt-dlp restricts connections to it.
_WILDCARD_ADDRESS: Final = {4: "0.0.0.0", 6: "::"}

# Refusals, and network failures on our side: neither says anything about the video.
_BLOCKED_MARKERS: Final = (
    "not a bot",
    "http error 429",
    "too many requests",
    "requested format is not available",
    "rate-limited",
    "network is unreachable",
    "no route to host",
    "temporary failure in name resolution",
    "name or service not known",
    "address family not supported",
    "cannot assign requested address",
)
_UNAVAILABLE_MARKERS: Final = (
    "video unavailable",
    "private video",
    "has been removed",
    "account associated with this video has been terminated",
    "copyright",
    "no longer available",
    "members-only",
    "join this channel",
    "sign in to confirm your age",
    "not available in your country",
    "this live event will begin",
    "premieres in",
)

# Kept from yt-dlp's info dictionary: what splits by channel and clip audits need.
_INFO_FIELDS: Final = (
    "id",
    "channel_id",
    "channel",
    "upload_date",
    "duration",
    "fps",
    "width",
    "height",
    "vcodec",
    "language",
    "title",
)


class YouTubeClient(Protocol):
    """The part of ``yt_dlp.YoutubeDL`` the downloader relies on."""

    def extract_info(self, url: str, download: bool) -> dict[str, Any]: ...

    def close(self) -> None: ...


ClientFactory = Callable[[dict[str, Any]], YouTubeClient]


def classify_error(message: str) -> Status:
    """Map a yt-dlp error to what retrying can achieve.

    "Requested format is not available" counts as a refusal: with a proof-of-origin token
    missing, YouTube serves only storyboards, and every later video would fail the same way.
    """
    text = message.lower().replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
    if any(marker in text for marker in _BLOCKED_MARKERS):
        return Status.BLOCKED
    if any(marker in text for marker in _UNAVAILABLE_MARKERS):
        return Status.UNAVAILABLE
    return Status.FAILED


def _exponential_retry_sleep(n: int) -> float:
    """Pause before retry ``n`` (from 0); yt-dlp passes ``n`` by keyword."""
    return min(2.0**n, _MAX_RETRY_SLEEP_S)


class YouTubeDownloader:
    """Fetches the video stream of YouTube videos, without audio, and their manual subtitles.

    Used as a context manager, so one yt-dlp client serves a whole session. The client reads
    a private copy of the cookies, because yt-dlp writes rotated cookies back to its file and
    parallel jobs must not overwrite the shared original. After a refusal or a stalled
    transfer the client is rebuilt, which also picks up a refreshed cookies file.

    Next to ``<id>.<ext>`` it writes ``<id>.info.json`` with the few fields needed later:
    channel, duration, frame rate, resolution and subtitle languages.
    """

    def __init__(
        self,
        config: YouTubeConfig,
        output_dir: Path,
        *,
        with_subtitles: bool,
        client_factory: ClientFactory = yt_dlp.YoutubeDL,
    ) -> None:
        self._config = config
        self._output_dir = output_dir
        self._with_subtitles = with_subtitles
        self._client_factory = client_factory
        self._scratch: tempfile.TemporaryDirectory[str] | None = None
        self._client: YouTubeClient | None = None

    def __enter__(self) -> Self:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._warm_up_token_provider()
        self._scratch = tempfile.TemporaryDirectory(prefix="signworld-youtube-")
        self._client = self._new_client()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._close_client()
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None

    def download(self, video_id: str) -> Outcome:
        self._require_client()
        try:
            with deadline(self._config.item_timeout_s):
                info = self._extract(video_id)
        except yt_dlp.utils.DownloadError as error:
            outcome = Outcome(video_id, classify_error(str(error)), str(error))
            if outcome.status is Status.BLOCKED:
                self._restart_client()
            return outcome
        except DeadlineExceededError as error:
            self._restart_client()
            return Outcome(video_id, Status.FAILED, str(error))
        self._write_info(video_id, info)
        return Outcome(video_id, Status.DONE)

    def check(self, video_id: str) -> Outcome:
        """Ask for the video's formats without downloading anything: a single page request.

        ``DONE`` means YouTube would serve the video; a missing proof-of-origin token still
        shows up, as "requested format is not available".
        """
        try:
            with deadline(self._config.item_timeout_s):
                self._require_client().extract_info(_WATCH_URL.format(video_id), download=False)
        except yt_dlp.utils.DownloadError as error:
            return Outcome(video_id, classify_error(str(error)), str(error))
        return Outcome(video_id, Status.DONE)

    def _extract(self, video_id: str) -> dict[str, Any]:
        url = _WATCH_URL.format(video_id)
        try:
            return self._require_client().extract_info(url, download=True)
        except subprocess.TimeoutExpired:
            # The token provider probes deno with a 15 s limit, which a file cache evicted
            # from shared storage can exceed at any time; the video itself is not at fault.
            logger.warning("Token provider timed out on %s; warming it up and retrying", video_id)
            self._warm_up_token_provider()
            self._restart_client()
            return self._require_client().extract_info(url, download=True)

    def _require_client(self) -> YouTubeClient:
        if self._client is None:
            raise RuntimeError("YouTubeDownloader.download must run inside its context")
        return self._client

    def options(self, cookies_file: Path | None) -> dict[str, Any]:
        config = self._config
        options: dict[str, Any] = {
            "format": self._format_selector(),
            "outtmpl": {"default": str(self._output_dir / "%(id)s.%(ext)s")},
            "writesubtitles": self._with_subtitles,
            "writeautomaticsub": False,
            "subtitleslangs": ["all", "-live_chat"],
            "subtitlesformat": "vtt",
            "noplaylist": True,
            "continuedl": True,
            "overwrites": False,
            "fixup": "never",
            "socket_timeout": config.socket_timeout_s,
            "retries": config.retries,
            "fragment_retries": config.retries,
            "extractor_retries": config.retries,
            "retry_sleep_functions": dict.fromkeys(
                ("http", "fragment", "extractor"), _exponential_retry_sleep
            ),
            "sleep_interval_requests": config.sleep_requests_s,
            "sleep_interval": config.sleep_min_s,
            "max_sleep_interval": config.sleep_max_s,
            "sleep_interval_subtitles": config.sleep_subtitles_s,
            "quiet": True,
            "no_warnings": False,
            "noprogress": True,
            "logger": _LoggerBridge(logger),
        }
        if cookies_file is not None:
            options["cookiefile"] = str(cookies_file)
        if config.ip_version is not None:
            options["source_address"] = _WILDCARD_ADDRESS[config.ip_version]
        if config.pot_server_home is not None:
            options["extractor_args"] = {
                _POT_PROVIDER_ARGUMENTS: {"server_home": [str(config.pot_server_home)]}
            }
        return options

    def _format_selector(self) -> str:
        config = self._config
        height = f"[height<={config.max_height}]"
        smooth = f"{height}[fps<={config.max_fps}]"
        codecs = ["[vcodec^=avc1]", ""] if config.prefer_h264 else [""]
        videos = [f"bv*{codec}{limit}" for codec in codecs for limit in (smooth, height)]
        return "/".join([*videos, f"b{height}"])

    def _warm_up_token_provider(self) -> None:
        """Load the token script once, without the provider's time limit.

        The provider is skipped if deno does not answer its version probe within 15 s, which
        a cold start from shared storage exceeds; without tokens YouTube refuses every video.
        """
        home = self._config.pot_server_home
        if home is None:
            return
        script = home / "src" / _POT_SCRIPT
        deno = shutil.which("deno")
        if deno is None or not script.is_file():
            logger.warning("No deno on PATH or no %s: YouTube will likely refuse downloads", script)
            return
        subprocess.run(
            [deno, "run", "--allow-all", str(script), "--version"],
            capture_output=True,
            check=False,
            timeout=_WARM_UP_TIMEOUT_S,
        )

    def _new_client(self) -> YouTubeClient:
        return self._client_factory(self.options(self._private_cookies()))

    def _private_cookies(self) -> Path | None:
        source = self._config.cookies_file
        if source is None or self._scratch is None:
            return source
        copy = Path(self._scratch.name) / "cookies.txt"
        shutil.copyfile(source, copy)
        return copy

    def _restart_client(self) -> None:
        self._close_client()
        self._client = self._new_client()

    def _close_client(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _write_info(self, video_id: str, info: dict[str, Any]) -> None:
        summary = {field: info.get(field) for field in _INFO_FIELDS}
        summary["subtitle_languages"] = sorted(info.get("requested_subtitles") or {})
        path = self._output_dir / f"{video_id}.info.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


class _LoggerBridge:
    """The logger interface yt-dlp expects, forwarding to standard logging.

    Errors are logged at debug level: the fetch loop already reports every outcome.
    """

    def __init__(self, target: logging.Logger) -> None:
        self._target = target

    def debug(self, message: str) -> None:
        self._target.debug(message)

    def info(self, message: str) -> None:
        self._target.debug(message)

    def warning(self, message: str) -> None:
        self._target.warning(message)

    def error(self, message: str) -> None:
        self._target.debug(message)
