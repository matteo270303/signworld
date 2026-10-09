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
    assert config.physical.enabled and config.physical.target_dim == config.pose_encoder.output_dim


@pytest.mark.parametrize("arm", ["A0", "A", "V", "B0", "B", "C"])
def test_every_arm_overlay_sets_its_arm(arm: str) -> None:
    assert load_config(BASE, ABLATIONS / f"arm_{arm}.yaml").losses.arm == arm


def test_ablations_lay_over_the_winning_arm() -> None:
    config = load_config(BASE, ABLATIONS / "arm_B.yaml", ABLATIONS / "esp2_no_physical.yaml")

    assert config.losses.arm == "B"
    assert not config.physical.enabled
    assert config.physical.target_dim == 192  # untouched keys keep the base value
    assert load_config(BASE, ABLATIONS / "esp4_latent.yaml").semantic.hypotheses == 4


def test_inherits_and_merge(tmp_path: Path) -> None:
    child = tmp_path / "child.yaml"
    child.write_text(f"inherits: {BASE}\nname: child\nlosses: {{arm: C}}\n")

    config = load_config(child)

    assert (config.name, config.losses.arm, config.losses.sigreg_weight) == ("child", "C", 0.04)
    assert merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_queries_must_split_into_the_hypotheses() -> None:
    with pytest.raises(ValidationError):
        SemanticSettings(queries=8, hypotheses=3)


@pytest.mark.parametrize(
    ("overlay", "check"),
    [
        ("arm_V.yaml", lambda c: c.losses.arm == "V"),
        ("esp2_no_physical.yaml", lambda c: not c.physical.enabled),
        ("esp4_latent.yaml", lambda c: c.semantic.hypotheses == 4),
        ("esp6_video_target.yaml", lambda c: c.physical.target == "video"),
        ("g1_global.yaml", lambda c: c.semantic.trains_encoder),
        ("d4_seed1.yaml", lambda c: c.training.seed == 1),
    ],
)
def test_every_run_of_the_plan_loads_over_an_arm(overlay: str, check) -> None:  # type: ignore[no-untyped-def]
    config = load_config(BASE, ABLATIONS / "arm_A.yaml", ABLATIONS / overlay)

    assert check(config)
    assert config.name.startswith("worldsign-") and config.name != "worldsign-A"


def test_the_literature_weights_of_the_arms() -> None:
    losses = load_config(BASE).losses

    assert losses.vicreg_coefficients == (25.0, 25.0, 1.0)  # Bardes et al., Tab. 7
    assert losses.uniformity_t == 2.0 and losses.uniformity_weight == pytest.approx(1 / 3)
    assert losses.infonce_temperature == 0.07 and losses.infonce_min_temperature == 0.01
    assert losses.false_negative_alpha == 1e-3  # GloFND, bimodal (FastCLIP, CC3M)
