"""The run configuration: the base file, the ablation overlays and their merge."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from signworld.experiment.train.config import SemanticSettings, load_config, merge
from tests.worldsign import ABLATIONS, BASE


def test_the_base_file_is_arm_a_with_the_documented_sizes() -> None:
    config = load_config(BASE)

    assert config.losses.arm == "A"
    assert config.encoder.levels == (5, 11, 17, 23)
    assert [(s.blocks, s.spatial_scale) for s in config.masking.specs] == [(8, 0.15), (2, 0.7)]
    assert config.semantic.hypotheses == 1
    assert config.physical.enabled and config.pose_encoder.trainable


@pytest.mark.parametrize("arm", ["A0", "A", "B0", "B", "C"])
def test_every_arm_overlay_sets_its_arm(arm: str) -> None:
    assert load_config(BASE, ABLATIONS / f"arm_{arm}.yaml").losses.arm == arm


def test_ablations_lay_over_the_winning_arm() -> None:
    config = load_config(BASE, ABLATIONS / "arm_B.yaml", ABLATIONS / "esp2_no_physical.yaml")

    assert config.losses.arm == "B"
    assert not config.physical.enabled
    assert config.physical.target_dim == 256  # untouched keys keep the base value
    assert not load_config(BASE, ABLATIONS / "esp3_pose_frozen.yaml").pose_encoder.trainable
    assert load_config(BASE, ABLATIONS / "esp4_latent.yaml").semantic.hypotheses == 4


def test_inherits_and_merge(tmp_path: Path) -> None:
    child = tmp_path / "child.yaml"
    child.write_text(f"inherits: {BASE}\nname: child\nlosses: {{arm: C}}\n")

    config = load_config(child)

    assert (config.name, config.losses.arm, config.losses.sigreg_weight) == ("child", "C", 0.05)
    assert merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_queries_must_split_into_the_hypotheses() -> None:
    with pytest.raises(ValidationError):
        SemanticSettings(queries=8, hypotheses=3)
