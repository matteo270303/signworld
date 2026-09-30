"""The video branch end to end on tiny real V-JEPA 2.1 modules, and the budget at full size."""

import copy

import pytest
import torch
from torch import nn

from signworld.worldmodel.budget import VideoBudget
from signworld.worldmodel.config import FusionSettings, load_config
from signworld.worldmodel.losses import Objective, physical_energy
from signworld.worldmodel.masking import TokenGrid
from signworld.worldmodel.physical import set_rope_grid
from signworld.worldmodel.video_branch import assemble

from .conftest import ABLATIONS, BASE, GRID, meta_modules, needs_hub, tiny_config


def _batch(size: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    frames = torch.randint(0, 256, (size, 4, 64, 64, 3), generator=generator, dtype=torch.uint8)
    boxes = torch.tensor([0.1, 0.1, 0.6, 0.7]).expand(size, GRID.steps, 4, 4).clone()
    visible = torch.ones(size, GRID.steps, 4, dtype=torch.bool)
    return frames, boxes, visible


@needs_hub
def test_one_training_step_reaches_only_the_trainable_parts() -> None:
    encoder, predictor = meta_modules()
    config = tiny_config()
    branch = assemble(encoder, predictor, config, GRID)
    frames, boxes, visible = _batch()
    generator = torch.Generator().manual_seed(0)

    physical = branch.physical(
        frames, boxes, visible, step=10, total_steps=100, generator=generator
    )
    predicted = branch.semantic(frames)
    target = torch.randn(2, GRID.steps, 4, 8)
    objective = Objective(config.losses, config.semantic, physical=True)
    terms = objective.semantic_terms(predicted, torch.randn(2, 512), ["a", "b"], generator)
    confidence = torch.ones(2, GRID.steps, 4)
    terms["e_fis"] = physical_energy(
        physical.predictions, target, confidence, physical.context_lambda
    )
    objective.combine(terms).total.backward()

    assert len(physical.predictions) == 2  # both mask kinds
    assert physical.predictions[0].masked.shape == (2, GRID.steps, 4, 8)
    assert physical.predictions[0].visible.shape == (2, GRID.steps, 4, 8)
    assert predicted.shape == (2, 1, 512)
    frozen = branch.backbone.encoder.patch_embed.proj.weight
    assert not frozen.requires_grad and frozen.grad is None
    adapters = branch.backbone.adapters[0].up[0]
    assert adapters.grad is not None  # B receives gradient at step 0
    assert branch.semantic_predictor.queries.grad is not None
    assert branch.physical_predictor is not None
    assert all(
        p.grad is not None for p in branch.physical_predictor.fusion.parameters() if p.requires_grad
    )


@needs_hub
@pytest.mark.parametrize("kind", ["linear", "residual"])
def test_linear_and_residual_fusion_reproduce_the_released_predictor(kind: str) -> None:
    encoder, predictor = meta_modules()
    released = copy.deepcopy(predictor)
    released.predictor_proj = nn.Identity()
    released.predictor_proj_context = nn.Identity()
    set_rope_grid(released, GRID.rows)
    config = tiny_config(fusion=FusionSettings(kind=kind))
    branch = assemble(encoder, predictor, config, GRID)
    frames, _, _ = _batch(1)
    mask = branch.masks(1, torch.Generator().manual_seed(3))[0]
    physical = branch.physical_predictor
    assert physical is not None

    with torch.no_grad():
        levels = branch.backbone.context_levels(frames, mask.context)
        ours = physical.tokens(levels, mask)
        last = encoder.norms_block[-1](levels[-1])
        target, context = released(last, [mask.context], [mask.target], mask_index=0)

    assert torch.allclose(ours[0, mask.target[0]], target[0], atol=1e-5)
    assert torch.allclose(ours[0, mask.context[0]], context[0], atol=1e-5)


@needs_hub
def test_the_rope_grid_follows_the_clip() -> None:
    _, predictor = meta_modules()
    set_rope_grid(predictor, 16)
    grids = {m.grid_size for m in predictor.modules() if hasattr(m, "separate_positions")}
    assert grids == {16}


@needs_hub
def test_esp2_has_no_physical_level_and_esp4_has_four_hypotheses() -> None:
    encoder, predictor = meta_modules()
    branch = assemble(encoder, predictor, tiny_config(ABLATIONS / "esp2_no_physical.yaml"), GRID)
    assert not branch.has_physical_level
    encoder, predictor = meta_modules()
    branch = assemble(encoder, predictor, tiny_config(ABLATIONS / "esp4_latent.yaml"), GRID)
    assert branch.semantic(_batch(1)[0]).shape == (1, 4, 512)


class _Attention(nn.Module):
    """The shapes of Meta's RoPE attention, and the attributes set_rope_grid looks for."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.grid_size = 24

    def separate_positions(self) -> None: ...


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = _Attention(width)
        self.mlp = nn.ModuleDict(
            {"fc1": nn.Linear(width, 4 * width), "fc2": nn.Linear(4 * width, width)}
        )


def _full_size_stubs() -> tuple[nn.Module, nn.Module]:
    """ViT-L (24 x 1,024) and its predictor (12 x 384) with Meta's layer shapes, on meta."""
    encoder = nn.Module()
    encoder.blocks = nn.ModuleList(_Block(1024) for _ in range(24))
    encoder.norms_block = nn.ModuleList(nn.LayerNorm(1024) for _ in range(4))
    encoder.hierarchical_layers = [5, 11, 17, 23]
    encoder.embed_dim = 1024
    predictor = nn.Module()
    predictor.predictor_blocks = nn.ModuleList(_Block(384) for _ in range(12))
    predictor.predictor_embed = nn.Linear(1024, 384)
    predictor.predictor_norm = nn.LayerNorm(384)
    predictor.predictor_proj = nn.Linear(384, 1664)
    predictor.predictor_proj_context = nn.Linear(384, 1664)
    return encoder, predictor


def test_the_full_size_budget_matches_the_document() -> None:
    """Trainable parameters per component at full size, against §4.8."""
    config = load_config(BASE)
    with torch.device("meta"):
        encoder, predictor = _full_size_stubs()
        branch = assemble(encoder, predictor, config, TokenGrid())

    budget = VideoBudget.of(branch)
    assert budget.encoder_lora == 7_077_888  # §4.8: 7.08 M
    assert budget.predictor_lora == 1_327_104  # 1.33 M
    assert budget.fusion == pytest.approx(4.60e6, rel=0.01)  # 4 LayerNorms + MLP
    assert budget.physical_head == 256 * 384 + 256  # 0.10 M
    assert budget.encoder_norms == pytest.approx(0.10e6, rel=0.02)
    assert budget.semantic_predictor == pytest.approx(7.67e6, rel=0.01)
