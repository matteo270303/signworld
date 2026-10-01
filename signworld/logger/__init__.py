"""Process-wide logging setup."""

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Send every logger to stderr with one consistent format."""
    logging.basicConfig(level=level, format=_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", force=True)
