from pathlib import Path

import pytest
from pydantic import ValidationError

from signworld.acquisition.config import AcquisitionConfig
from signworld.acquisition.sources import SOURCES, UnknownSourceError, build_source

REPOSITORY = Path(__file__).resolve().parents[2]


def test_repository_configuration_is_valid_for_every_source() -> None:
    config = AcquisitionConfig.from_yaml(REPOSITORY / "configs" / "acquisition.yaml")

    for name in SOURCES:
        assert build_source(name, config).name == name


def test_unknown_top_level_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        AcquisitionConfig.model_validate({"data_root": str(tmp_path), "dataroot": "typo"})


def test_misspelled_source_settings_are_rejected(tmp_path: Path) -> None:
    config = AcquisitionConfig.model_validate(
        {"data_root": str(tmp_path), "sources": {"csl_news": {"revison": "typo"}}}
    )

    with pytest.raises(ValidationError):
        build_source("csl_news", config)


def test_unknown_source_is_reported(config: AcquisitionConfig) -> None:
    with pytest.raises(UnknownSourceError, match="available"):
        build_source("nonexistent", config)
