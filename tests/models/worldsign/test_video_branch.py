"""The video branch end to end on tiny real V-JEPA 2.1 modules, and the budget at full size."""

import copy

import pytest
import torch
from torch import nn

from signworld.experiment.train.budget import VideoBudget
from signworld.experiment.train.config import FusionSettings, load_config
from signworld.loss.worldsign import Objective, physical_energy
from signworld.models.worldsign.backbone import compile_blocks
from signworld.models.worldsign.masking import TokenGrid
from signworld.models.worldsign.physical import set_rope_grid
from signworld.models.worldsign.video_branch import assemble
from tests.worldsign import (
    ABLATIONS,
    BASE,
    GRID,
    full_size_stubs,
    meta_modules,
    needs_hub,
    slow,
    tiny_config,
)


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
    target = torch.randn(2, GRID.steps, 8)
    objective = Objective(config.losses, config.semantic, physical=True)
    terms = objective.semantic_terms(predicted, torch.randn(2, 256), ["a", "b"], generator)
    confidence = torch.ones(2, GRID.steps)
    terms["e_fis"] = physical_energy(
        physical.predictions, target, confidence, physical.context_lambda
    )
    objective.combine(terms).total.backward()

    assert len(physical.predictions) == 2  # both mask kinds
    assert physical.predictions[0].state.shape == (2, GRID.steps, 8)
    assert physical.predictions[0].box_tokens.shape == (2, GRID.steps, 4)
    assert predicted.shape == (2, 1, 256)
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
    physical.eval()  # the dropout of the blocks (0.1) is a training-time draw

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
    assert branch.semantic(_batch(1)[0]).shape == (1, 4, 256)


def test_the_full_size_budget_matches_the_document() -> None:
    """Trainable parameters per component at full size, against §4.8."""
    config = load_config(BASE)
    with torch.device("meta"):
        encoder, predictor = full_size_stubs()
        branch = assemble(encoder, predictor, config, TokenGrid())

    budget = VideoBudget.of(branch)
    assert budget.encoder_lora == 7_077_888  # §4.8: 7.08 M
    assert budget.predictor_lora == 1_327_104  # 1.33 M
    assert budget.fusion == pytest.approx(4.60e6, rel=0.01)  # 4 LayerNorms + MLP
    assert budget.physical_head == 4 * 384 * 192 + 192  # 0.30 M
    assert budget.encoder_norms == pytest.approx(0.10e6, rel=0.02)
    assert budget.semantic_predictor == pytest.approx(7.60e6, rel=0.01)


@needs_hub
def test_compiled_blocks_are_bound_to_their_block_and_follow_a_copy() -> None:
    encoder, _ = meta_modules()
    compile_blocks(encoder.blocks)
    copied = copy.deepcopy(encoder)
    for original, clone in zip(encoder.blocks, copied.blocks, strict=True):
        assert original.forward.__self__ is original
        assert clone.forward.__self__ is clone  # the copy runs its own weights


def _step_gradients(compiled: bool) -> tuple[list[float], dict[str, torch.Tensor]]:
    """Two steps of the physical and semantic passes (different masks): losses, gradients."""
    encoder, predictor = meta_modules()
    config = tiny_config()
    settings = config.encoder.model_copy(update={"compile_blocks": compiled})
    branch = assemble(encoder, predictor, config.model_copy(update={"encoder": settings}), GRID)
    assert config.encoder.activation_checkpointing
    assert config.physical.dropout > 0
    frames, boxes, visible = _batch()
    losses = []
    for step in range(2):
        torch.manual_seed(step)  # the predictor's dropout, replayed by the checkpointing
        generator = torch.Generator().manual_seed(step)
        physical = branch.physical(frames, boxes, visible, step, 100, generator)
        total = branch.semantic(frames).square().mean()
        for prediction in physical.predictions:
            total = total + prediction.state.square().mean()
        total.backward()
        losses.append(float(total.detach()))
    gradients = {n: p.grad.clone() for n, p in branch.named_parameters() if p.grad is not None}
    return losses, gradients


@slow
@needs_hub
def test_compiled_blocks_give_the_eager_losses_and_gradients() -> None:
    eager_losses, eager = _step_gradients(compiled=False)
    losses, compiled = _step_gradients(compiled=True)
    assert losses == pytest.approx(eager_losses, rel=1e-5)
    assert set(compiled) == set(eager)
    largest = max(float(gradient.norm()) for gradient in eager.values())
    for name, gradient in eager.items():
        if gradient.norm() < 1e-3 * largest:  # zero but for rounding: biases before a BatchNorm
            continue
        error = (compiled[name] - gradient).norm() / gradient.norm()
        assert error < 1e-3, name
