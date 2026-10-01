"""BOBSL (Albanie et al., 2021): BSL-interpreted BBC broadcasts, released under a BBC agreement."""

import os
from urllib.parse import urljoin

from ..config import FrozenModel
from ..outcome import Outcome, Status
from ..transfer.http import AuthenticationError
from .base import Access, AccessRequiredError, DatasetSource


class BOBSLSettings(FrozenModel):
    base_url: str = "https://thor.robots.ox.ac.uk/bobsl/v1.4/"
    metadata_files: tuple[str, ...] = ()
    media_files: tuple[str, ...] = ()
    username_env: str = "BOBSL_USERNAME"
    password_env: str = "BOBSL_PASSWORD"


class BOBSLSource(DatasetSource[BOBSLSettings]):
    """Files listed in the configuration, downloaded with personal BBC credentials."""

    name = "bobsl"
    homepage = "https://www.robots.ox.ac.uk/~vgg/data/bobsl/"
    terms = "BBC BOBSL Terms of Use; a personal password is issued after approval."
    access = Access.CREDENTIALS
    settings_model = BOBSLSettings

    def check_access(self) -> None:
        variables = (self.settings.username_env, self.settings.password_env)
        missing = [variable for variable in variables if not os.environ.get(variable)]
        if missing:
            raise AccessRequiredError(
                f"{self.name}: set {' and '.join(missing)}; credentials are issued by the BBC "
                f"after accepting the terms of use ({self.homepage})."
            )
        if not (self.settings.metadata_files or self.settings.media_files):
            raise AccessRequiredError(
                f"{self.name}: list metadata_files and media_files in the configuration "
                "once the file index is visible with your credentials."
            )

    def fetch_metadata(self) -> None:
        self.check_access()
        for relative in self.settings.metadata_files:
            url = urljoin(self.settings.base_url, relative)
            target = self.http.fetch(url, self.layout.metadata / relative, auth=self._credentials())
            self.provenance.register(target, origin=url)

    def media_keys(self) -> list[str]:
        return list(self.settings.media_files)

    def fetch_item(self, key: str) -> Outcome:
        try:
            self.http.fetch(
                urljoin(self.settings.base_url, key),
                self.layout.raw / key,
                auth=self._credentials(),
            )
        except AuthenticationError as error:
            return Outcome(key, Status.BLOCKED, str(error))
        return Outcome(key, Status.DONE)

    def _credentials(self) -> tuple[str, str]:
        # HTTP Basic is assumed; to be confirmed against the server once access is granted.
        return os.environ[self.settings.username_env], os.environ[self.settings.password_env]
