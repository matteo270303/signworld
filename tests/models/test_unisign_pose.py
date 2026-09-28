"""Collaudo "riproduzione dei pesi" for the pose encoder: same outputs as Uni-Sign's own code.

The reference comparison needs a clone of github.com/ZechengLi19/Uni-Sign and a released
checkpoint; set UNISIGN_REFERENCE and UNISIGN_CHECKPOINT, otherwise those tests are skipped.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

from signworld.models.unisign_pose import OUTPUT_CHANNELS, UniSignPoseEncoder
from signworld.pose.wholebody import ARTICULATOR_INDICES, Articulator

REFERENCE = os.environ.get("UNISIGN_REFERENCE")
CHECKPOINT = os.environ.get("UNISIGN_CHECKPOINT")


def _inputs(batch: int = 2, time: int = 16, seed: int = 0) -> dict[Articulator, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        part: torch.rand(batch, time, len(ARTICULATOR_INDICES[part]), 3, generator=generator) * 2
        - 1
        for part in Articulator
    }


def test_output_is_one_feature_per_frame_and_articulator() -> None:
    features = UniSignPoseEncoder().eval()(_inputs())

    assert set(features) == set(Articulator)
    assert all(value.shape == (2, 16, OUTPUT_CHANNELS) for value in features.values())


def test_left_hand_shares_every_weight_with_the_right() -> None:
    encoder = UniSignPoseEncoder()

    assert encoder.gcn_modules["left"] is encoder.gcn_modules["right"]
    assert encoder.fusion_gcn_modules["left"] is encoder.fusion_gcn_modules["right"]
    assert encoder.proj_linear["left"] is encoder.proj_linear["right"]


@pytest.mark.skipif(not (REFERENCE and CHECKPOINT), reason="Uni-Sign reference not configured")
def test_outputs_match_the_reference_implementation() -> None:
    sys.path.insert(0, str(REFERENCE))
    from stgcn_layers.stgcn_block import (  # type: ignore[import-not-found]  # noqa: PLC0415 (reference clone is only on sys.path here)
        get_stgcn_chain,
    )

    state = torch.load(Path(str(CHECKPOINT)), map_location="cpu", weights_only=True)["model"]
    ours = UniSignPoseEncoder.from_checkpoint(Path(str(CHECKPOINT))).eval()

    reference: dict[str, dict[str, torch.nn.Module]] = {}
    for part in ("body", "right", "face_all"):
        adjacency = state[f"gcn_modules.{part}.layer0_0.gcn.A"].float()
        spatial, dim = get_stgcn_chain(64, "spatial", (1, 2), adjacency.clone(), True)
        temporal, _ = get_stgcn_chain(dim, "temporal", (5, 2), adjacency.clone(), True)
        projection = torch.nn.Linear(3, 64)
        for name, module in (
            ("proj_linear", projection),
            ("gcn_modules", spatial),
            ("fusion_gcn_modules", temporal),
        ):
            prefix = f"{name}.{part}."
            own = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
            module.load_state_dict({k: v.float() for k, v in own.items()})
            module.eval()
        reference[part] = {"proj": projection, "spatial": spatial, "temporal": temporal}
    reference["left"] = reference["right"]

    inputs = _inputs(batch=3, time=32, seed=1)
    with torch.no_grad():
        expected = {}
        spatial_out = {
            part: reference[part]["spatial"](
                reference[part]["proj"](inputs[Articulator(part)]).permute(0, 3, 1, 2)
            )
            for part in reference
        }
        anchors = {"left": -2, "right": -1, "face_all": 0}
        for part in reference:
            x = spatial_out[part]
            if part in anchors:
                x = x + spatial_out["body"][..., anchors[part]][..., None]
            expected[part] = reference[part]["temporal"](x).mean(-1).transpose(1, 2)
        actual = ours(inputs)

    for part in Articulator:
        cosine = torch.nn.functional.cosine_similarity(actual[part], expected[part], dim=-1)
        assert cosine.min() > 0.999
        assert (actual[part] - expected[part]).abs().max() < 1e-3
