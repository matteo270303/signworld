"""Registry of the dataset sources the pipeline can fetch."""

from typing import Any, Final

from ..config import AcquisitionConfig
from .base import DatasetSource
from .bobsl import BOBSLSource
from .csl_daily import CSLDailySource
from .csl_news import CSLNewsSource
from .openasl import OpenASLSource
from .phoenix14t import Phoenix14TSource
from .youtube_sl25 import YouTubeSL25Source

SOURCES: Final[dict[str, type[DatasetSource[Any]]]] = {
    source.name: source
    for source in (
        YouTubeSL25Source,
        OpenASLSource,
        CSLNewsSource,
        BOBSLSource,
        Phoenix14TSource,
        CSLDailySource,
    )
}


class UnknownSourceError(ValueError):
    """No source is registered under the requested name."""


def build_source(name: str, config: AcquisitionConfig) -> DatasetSource[Any]:
    """Instantiate a source with its settings validated from ``config.sources[name]``."""
    try:
        source = SOURCES[name]
    except KeyError as error:
        available = ", ".join(sorted(SOURCES))
        raise UnknownSourceError(f"unknown source {name!r}; available: {available}") from error
    settings = source.settings_model.model_validate(config.sources.get(name, {}))
    return source(settings, config)
