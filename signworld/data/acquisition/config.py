"""Typed configuration of the acquisition pipeline."""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)


class FrozenModel(BaseModel):
    """Immutable settings that reject unknown keys, so a typo in the YAML fails loudly."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class HttpConfig(FrozenModel):
    chunk_bytes: PositiveInt = 8 * 1024 * 1024
    retries: PositiveInt = 5
    timeout_s: PositiveFloat = 60.0
    backoff_s: NonNegativeFloat = 5.0


class RefusalPolicy(FrozenModel):
    """How a shard reacts when the host refuses requests (bot wall, HTTP 429, bad credentials).

    Each consecutive refusal pauses the shard for ``cooldown_s``, doubled every time; at
    ``max_consecutive`` the shard stops, so the scheduler resumes it later.
    """

    max_consecutive: PositiveInt = 3
    cooldown_s: NonNegativeFloat = 600.0


class YouTubeConfig(FrozenModel):
    cookies_file: Path | None = None
    pot_server_home: Path | None = None
    max_height: PositiveInt = 720
    max_fps: PositiveInt = 30
    prefer_h264: bool = True
    ip_version: Literal[4, 6] | None = None
    """Connect only over this IP version; ``None`` lets the system choose."""
    retries: PositiveInt = 5
    socket_timeout_s: PositiveFloat = 30.0
    item_timeout_s: PositiveFloat = 3600.0
    sleep_requests_s: NonNegativeFloat = 0.75
    sleep_min_s: NonNegativeFloat = 10.0
    sleep_max_s: NonNegativeFloat = 20.0
    sleep_subtitles_s: NonNegativeFloat = 5.0

    @model_validator(mode="after")
    def _ordered_sleep_bounds(self) -> Self:
        if self.sleep_max_s < self.sleep_min_s:
            raise ValueError("sleep_max_s must not be smaller than sleep_min_s")
        return self


class AcquisitionConfig(FrozenModel):
    data_root: Path
    max_attempts: PositiveInt = 3
    run_budget_s: PositiveFloat | None = None
    """Stop a run after this long, before a host that limits sustained activity refuses us.

    The scheduler then rests the shard and starts the next run; ``None`` fetches without a
    time limit.
    """
    refusals: RefusalPolicy = Field(default_factory=RefusalPolicy)
    http: HttpConfig = Field(default_factory=HttpConfig)
    youtube: YouTubeConfig = Field(default_factory=YouTubeConfig)
    sources: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        with path.open(encoding="utf-8") as stream:
            return cls.model_validate(yaml.safe_load(stream))
