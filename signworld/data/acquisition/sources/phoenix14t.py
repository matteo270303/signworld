"""RWTH-PHOENIX-Weather 2014 T (Camgöz et al., 2018): German weather forecasts in DGS."""

import logging
from pathlib import PurePosixPath
from urllib.parse import urlparse

from pydantic import PositiveInt

from ..config import FrozenModel
from ..outcome import Outcome, Status
from ..transfer.archive import extract_tar
from .base import Access, DatasetSource

logger = logging.getLogger(__name__)


class Phoenix14TSettings(FrozenModel):
    url: str = "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz"
    size_bytes: PositiveInt = 41_699_758_035


class Phoenix14TSource(DatasetSource[Phoenix14TSettings]):
    """One archive holding the frames together with gloss and translation annotations."""

    name = "phoenix14t"
    homepage = "https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/"
    terms = "See the homepage for the terms of use."
    access = Access.PUBLIC
    settings_model = Phoenix14TSettings

    def fetch_metadata(self) -> None:
        logger.info("%s: annotations ship inside the archive; nothing to fetch", self.name)

    def media_keys(self) -> list[str]:
        return [PurePosixPath(urlparse(self.settings.url).path).name]

    def fetch_item(self, key: str) -> Outcome:
        archive = self.http.fetch(
            self.settings.url, self.layout.raw / key, expected_bytes=self.settings.size_bytes
        )
        extract_tar(archive, self.layout.extracted)
        return Outcome(key, Status.DONE)
