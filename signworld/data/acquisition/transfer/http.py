"""Resumable HTTP(S) downloads that never leave a truncated file at the final path."""

import logging
import time
from pathlib import Path
from typing import Final

import requests

from ..config import HttpConfig

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS: Final = frozenset({408, 425, 429, 500, 502, 503, 504})
_UNAUTHORIZED_STATUS: Final = frozenset({401, 403})
_TRANSIENT_ERRORS: Final = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


class DownloadError(RuntimeError):
    """A file could not be downloaded, or it failed verification."""


class AuthenticationError(DownloadError):
    """The server rejected the credentials."""


class _RetryableStatusError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")


class HttpDownloader:
    """Streams URLs to disk, resuming partial transfers and publishing only complete files.

    Bytes go to ``<name>.part``, renamed to the final name once the transfer has finished
    and, when the size is known, matches it.
    """

    def __init__(self, config: HttpConfig, session: requests.Session | None = None) -> None:
        self._config = config
        self._session = session or requests.Session()

    def fetch(
        self,
        url: str,
        destination: Path,
        *,
        expected_bytes: int | None = None,
        auth: tuple[str, str] | None = None,
    ) -> Path:
        if destination.exists() and (
            expected_bytes is None or destination.stat().st_size == expected_bytes
        ):
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f"{destination.name}.part")
        self._transfer_with_retries(url, partial, auth)
        received = partial.stat().st_size
        if expected_bytes is not None and received != expected_bytes:
            partial.unlink()
            raise DownloadError(f"{url}: expected {expected_bytes} bytes, received {received}")
        partial.replace(destination)
        logger.info("Downloaded %s (%d bytes)", destination, received)
        return destination

    def _transfer_with_retries(self, url: str, partial: Path, auth: tuple[str, str] | None) -> None:
        for attempt in range(1, self._config.retries + 1):
            try:
                self._transfer(url, partial, auth)
            except (*_TRANSIENT_ERRORS, _RetryableStatusError) as error:
                if attempt == self._config.retries:
                    raise DownloadError(f"{url}: gave up after {attempt} attempts") from error
                delay = self._config.backoff_s * 2 ** (attempt - 1)
                logger.warning(
                    "%s: attempt %d failed (%s); retrying in %.0f s", url, attempt, error, delay
                )
                time.sleep(delay)
            else:
                return

    def _transfer(self, url: str, partial: Path, auth: tuple[str, str] | None) -> None:
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with self._session.get(
            url, headers=headers, auth=auth, stream=True, timeout=self._config.timeout_s
        ) as response:
            status = response.status_code
            if offset and status == requests.codes.requested_range_not_satisfiable:
                return  # the partial file already holds every byte
            if status in _RETRYABLE_STATUS:
                raise _RetryableStatusError(status)
            if status in _UNAUTHORIZED_STATUS:
                raise AuthenticationError(f"{url}: HTTP {status}")
            if not response.ok:
                raise DownloadError(f"{url}: HTTP {status}")
            resumed = offset > 0 and status == requests.codes.partial_content
            with partial.open("ab" if resumed else "wb") as sink:
                for chunk in response.iter_content(chunk_size=self._config.chunk_bytes):
                    sink.write(chunk)
