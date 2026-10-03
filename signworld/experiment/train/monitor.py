"""The fail-fast monitor of a training run (§4.13.3-§4.13.5).

Four cadences, in steps of 128 clips (the document's steps of 1,024 clips times 8):

* **every step**: energies, loss spikes over ``spike_sigma`` from the running mean;
* **frequent** (800): an extra pass on a few clips of the current batch reads collapse (the
  pose target ``s``, ŷ, the encoder's mean token), both predictors, the read-out per step and
  per mask coverage, the dynamics, keypoint errors against interpolation and constant
  velocity, localisation, the queries, the text head, the LoRA and the gradient of every term
  on its level (shares and conflicts);
* **validation** (4,000): every measure of the validation clips that the test also reads
  (``measures``: retrieval, losses, physical read-outs, keypoint errors, geometry, alignment
  and uniformity, noise test, hubness, modality gap, 2x2 table, leak test, attention), R@1
  on as many training clips with the gap, and on a fixed probe batch the pose target's
  isotropy, kinematic content per articulator and speed (CKA);
* **rare** (16,000): temporal order ω, the plausibility tests and the drift of the video
  encoder.

With ``diagnostics.cadence = epoch`` the trainer calls the frequent, validation and rare
readings at the end of each epoch instead, and judges there the stops due within it.
Everything is read at step 0 too, as the reference. Every reading goes to the metrics log;
the rules compare it with its threshold or with the step-0 reference and log an alarm. Only a
leak or a non-finite loss stops a run by itself; the gate run also stops at a failed F1-F3.
The readings of a level start with the stage it enters (gerarchia §7): the pose and semantic
readings from stage P, the physical ones from F0, the LoRA's from F.
"""

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from signworld.loss.worldsign import POSE_TERMS
from signworld.metrics.measures import collect, model_measures, pose_target_measures, split_measures
from signworld.metrics.readings import (
    Spread,
    gradient_norms,
    keypoint_errors,
    linear_cka,
    lora_ratios,
    physical_readings,
    query_cosine,
    ridge_r2,
    semantic_readings,
    shares_and_cosines,
    sigreg_ratio,
    term_gradients,
    text_head_spearman,
)
from signworld.models.worldsign.lora import adapter_parameters
from signworld.models.worldsign.masking import token_roles
from signworld.models.worldsign.model import StepRandomness, WorldSign, WorldSignBatch
from signworld.models.worldsign.plausibility import as_measures, plausibility_tests
from signworld.models.worldsign.readout import membership

from .config import DiagnosticsSettings, WorldSignConfig
from .curriculum import POSE, VIDEO_LORA, Stage, families
from .distributed import SINGLE, Distributed
from .validation import RetrievalScores, retrieval

PARTS = ("body", "left", "right", "face")


class RunStoppedError(RuntimeError):
    """A leak, or a failed programmed stop of the gate run: the run ends here."""


@dataclass(frozen=True, slots=True)
class Alarm:
    name: str
    step: int
    value: float
    threshold: float
    meaning: str
    stop: bool = False


@dataclass(frozen=True, slots=True)
class StopReport:
    """A programmed stop of the gate run (§4.13.5)."""

    name: str
    step: int
    criteria: dict[str, tuple[bool, str]]

    @property
    def passed(self) -> bool:
        return all(ok for ok, _ in self.criteria.values())


@dataclass
class _Rule:
    metric: str
    """A reading name; ``*`` matches any suffix."""
    kind: str
    """``below``, ``above``: against the threshold; ``below_ratio``, ``above_ratio``: the
    reading over its step-0 reference against the threshold."""
    threshold: float
    meaning: str
    readings: int = 1
    """Consecutive readings beyond the threshold before the alarm."""
    stop: bool = False


def _rules(s: DiagnosticsSettings) -> list[_Rule]:
    """The alarms of §4.13.3 that a reading can decide by itself."""
    return [
        _Rule(
            "leak_change",
            "above",
            s.leak_tolerance,
            "masked content changes the predictions: leak",
            stop=True,
        ),
        _Rule("s_rank", "below_ratio", s.collapse_ratio, "the pose target collapses"),
        _Rule("s_std", "below_ratio", s.collapse_ratio, "the pose target shrinks"),
        _Rule("y_rank", "below_ratio", s.collapse_ratio, "ŷ collapses"),
        _Rule(
            "encoder_rank", "below_ratio", s.collapse_ratio, "the video encoder's output collapses"
        ),
        _Rule("gamma_masked", "below", s.gamma_min, "the physical predictor answers the mean"),
        _Rule("gamma_sem", "below", s.gamma_min, "the semantic predictor answers the mean"),
        _Rule("dynamics_margin", "below", 0.0, "no dynamics learnt: the baseline wins"),
        _Rule("keypoint_margin", "below", 0.0, "decoded keypoints do not beat interpolation"),
        _Rule("localization_increase", "below", s.localization_min, "predictions not localised"),
        _Rule("order_cosine", "above", s.order_max, "the semantic predictor ignores the order"),
        _Rule("encoder_drift_r2", "below", s.drift_min, "the encoder flattens onto the pose"),
        _Rule("text_head_spearman", "below", s.spearman_min, "the text head alters the semantics"),
        _Rule(
            "video_cos_physical_semantic",
            "below",
            s.conflict_cosine,
            "pose and meaning conflict on the video LoRA",
            s.conflict_readings,
        ),
        _Rule(
            "y_cos_sem_sigreg",
            "below",
            s.conflict_cosine,
            "alignment and SIGReg conflict: anisotropic target",
            s.conflict_readings,
        ),
        _Rule(
            "pose_cos_anchor_sigreg_posa",
            "below",
            s.conflict_cosine,
            "SIGReg fights the anchor on the pose target",
            s.conflict_readings,
        ),
        _Rule(
            "video_share_max",
            "above",
            s.share_max,
            "one term dominates the video gradient",
            s.conflict_readings,
        ),
        _Rule(
            "lora_ratio_max", "above", s.lora_ratio_max, "a LoRA moves its layer by more than 10 %"
        ),
        _Rule("modality_gap", "above", s.modality_gap_max, "ŷ and ẽ live in separate spaces"),
        _Rule("hubness", "above_ratio", s.hubness_growth_max, "predictions pulled to the centre"),
        _Rule("noise_drop", "below", s.noise_drop_min, "the model does not look at the video"),
        _Rule("s_sigreg", "above_ratio", s.sigreg_growth_max, "the pose target leaves N(0, I)"),
        _Rule("pose_r2_drop_*", "above", s.pose_r2_drop_max, "the pose target loses kinematics"),
        _Rule("excluded_part1", "above", s.excluded_hands_max, "the left hand is mostly excluded"),
        _Rule("excluded_part2", "above", s.excluded_hands_max, "the right hand is mostly excluded"),
        _Rule("query_cosine", "above", s.query_cosine_max, "the queries collapsed"),
        _Rule("attention_on_boxes_ratio", "below", 1.0, "the queries look at the background"),
        _Rule("lora_zero_blocks", "above", 0.0, "LoRA blocks without gradient"),
        _Rule("loss_spikes", "above", 3.0, "repeated loss spikes: instability"),
    ]


@dataclass
class _References:
    values: dict[str, float] = field(default_factory=dict)
    latent: Tensor | None = None
    features: Tensor | None = None
    residual: Tensor | None = None


class Monitor:
    def __init__(
        self,
        model: WorldSign,
        config: WorldSignConfig,
        probe: Sequence[WorldSignBatch],
        log: Any,
        collective: Distributed = SINGLE,
        *,
        total_steps: int = 1,
    ) -> None:
        self.model = model
        self.config = config
        self.settings = config.diagnostics
        self.probe = list(probe)
        self.log = log
        self.collective = collective
        self.total_steps = total_steps
        self.device = collective.device
        self.references = _References()
        self.history: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.streaks: dict[str, int] = defaultdict(int)
        self.rules = _rules(self.settings)
        self.alarms: list[Alarm] = []
        self.last_validation: dict[str, float] = {}
        """Every reading of the latest validation, for the run's tables."""
        self._loss_mean = 0.0
        self._loss_var = 0.0
        self._loss_count = 0
        self._spikes: list[int] = []

    # ------------------------------------------------------------ cadences

    def due(self, step: int, every: int) -> bool:
        return step % every == 0

    def after_step(self, step: int, total: float, parts: dict[str, float]) -> None:
        """Every step: spikes over ``spike_sigma`` from the running mean of the loss."""
        for name, value in parts.items():
            self.history[f"term_{name}"].append((step, value))
        bound = self.settings.spike_sigma * math.sqrt(self._loss_var)
        stable = self._loss_count > 20 and self._loss_var > 0  # noqa: PLR2004 (a stable estimate)
        if stable and abs(total - self._loss_mean) > bound:
            self._spikes.append(step)
            self.log.write("spike", step=step, loss=total, mean=self._loss_mean)
        self._loss_count += 1
        rate = max(0.01, 1 / self._loss_count)
        delta = total - self._loss_mean
        self._loss_mean += rate * delta
        self._loss_var = (1 - rate) * (self._loss_var + rate * delta * delta)
        window = self.settings.frequent_every * self.settings.conflict_readings
        self._spikes = [s for s in self._spikes if s > step - window]

    def lora_gradients(self) -> dict[str, float]:
        """After the training backward: gradient norm of every LoRA block of the video encoder."""
        norms = gradient_norms(self.model.video.backbone.encoder)
        trainable = [
            p for p in adapter_parameters(self.model.video.backbone.encoder) if p.requires_grad
        ]
        if not trainable or not norms:
            return {}
        values = list(norms.values())
        ordered = sorted(values)
        median = ordered[len(ordered) // 2]
        return {
            "lora_zero_blocks": float(sum(v == 0 for v in values)),
            "lora_grad_max_over_median": max(values) / median if median > 0 else float("inf"),
        }

    def frequent(
        self, step: int, batch: WorldSignBatch, stage: Stage, extra: dict[str, float] | None = None
    ) -> dict[str, float]:
        """The frequent readings, on the first ``diagnostic_clips`` clips of the batch.

        The extra pass runs under the training's autocast: in float32, with gradient, the whole
        model on a GPU's batch does not fit in memory.
        """
        model = self.model
        clips = batch.take(self.settings.diagnostic_clips)
        record: dict[str, Any] = {}
        randomness = StepRandomness.at(self.config.training.seed + 1, step, self.collective.rank)
        was_training = model.training
        model.train(False)
        readings: dict[str, float] = dict(extra or {})
        bf16 = self.config.training.precision == "bf16"
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=bf16):
                terms = model.loss(
                    clips,
                    step,
                    self.total_steps,
                    randomness,
                    pose=stage.pose,
                    physical=stage.physical,
                    semantic=stage.semantic,
                    record=record,
                )
            readings |= self._gradient_readings(terms.parts, record)
        with torch.no_grad():
            readings |= self._level_readings(clips, record)
            if stage.physical and model.pose is not None:
                readings["localization_increase"] = self.localization(
                    clips.take(self.settings.small_clips), randomness
                )
        readings |= self._lora_readings()
        readings["loss_spikes"] = float(len(self._spikes))
        model.train(was_training)
        return self._record("frequent", step, readings)

    def validation(
        self,
        step: int,
        batches: Iterable[WorldSignBatch],
        languages: Sequence[str],
        train_batches: Iterable[WorldSignBatch] | None = None,
    ) -> RetrievalScores:
        """Every validation measure; returns the scores that decide.

        The measures of the test (``measures.split_measures``, with the bootstrap interval of
        T2V R@1 and the physical read-outs as ``val_*``), the leak test and the attention on
        the first clips, R@1 both ways on ``train_batches`` with the gap to validation, and
        the probe readings.
        """
        bf16 = self.config.training.precision == "bf16"
        seed = self.config.training.seed + 2
        cached = list(batches)
        seen = collect(
            self.model,
            cached,
            self.device,
            self.collective,
            bf16=bf16,
            seed=seed,
            desc=f"[val] step {step}",
        )
        scores, readings = split_measures(
            seen, languages, self.config, bootstrap=("t2v_r1",), physical_prefix="val_"
        )
        if cached:
            readings |= model_measures(
                self.model, cached[0], self.device, self.settings.small_clips
            )
        if train_batches is not None:
            train = collect(
                self.model,
                train_batches,
                self.device,
                self.collective,
                bf16=bf16,
                physical=False,
                noise=False,
                seed=seed,
                desc=f"[val train subset] step {step}",
            ).tensors
            subset = retrieval(train["predicted"], train["texts"], train["rows"])
            for direction, value in (("t2v", subset.t2v[1]), ("v2t", subset.v2t[1])):
                readings[f"train_{direction}_r1"] = value
                readings[f"gap_{direction}_r1"] = value - readings[f"{direction}_r1"]
        with torch.no_grad():
            readings |= self._probe_readings(step)
        self._record("validation", step, readings)
        self.last_validation = readings
        return scores

    def rare(self, step: int, batches: Iterable[WorldSignBatch]) -> dict[str, float]:
        """Temporal order ω and the plausibility tests on validation clips; drift of the video
        encoder on the probe batch."""
        readings: dict[str, float] = {}
        iterator = iter(batches)
        first = next(iterator, None)
        with torch.no_grad():
            if first is not None:
                clips = first.take(self.settings.order_clips).to(self.device)
                forward = self.model.video.semantic(clips.frames).float().mean(1)
                backward = self.model.video.semantic(clips.frames.flip(1)).float().mean(1)
                readings["order_cosine"] = float(
                    functional.cosine_similarity(forward, backward).mean()
                )
            drift = self._encoder_drift()
            if drift is not None:
                readings["encoder_drift_r2"] = drift
        if first is not None and self.model.pose is not None:
            chosen, wanted = [first], self.settings.plausibility_clips
            while sum(len(b.videos) for b in chosen) < wanted:
                following = next(iterator, None)
                if following is None:
                    break
                chosen.append(following)
            extra = sum(len(b.videos) for b in chosen) - wanted
            if extra > 0:
                chosen[-1] = chosen[-1].take(len(chosen[-1].videos) - extra)
            tests = plausibility_tests(
                self.model, chosen, self.device, self.settings.plausibility_masks, self.collective
            )
            readings |= as_measures(tests)
        return self._record("rare", step, readings)

    # ------------------------------------------------------------ readings

    def _gradient_readings(
        self, parts: dict[str, Tensor], record: dict[str, Any]
    ) -> dict[str, float]:
        """Shares and cosines of every term's gradient on the weights of its level: the video
        LoRA (the physical level, and the semantic one in the «global» ablation) and the pose
        encoder; the conflict between alignment and SIGReg on ŷ."""
        weight = self.config.losses.sigreg_weight
        weights = {k: (weight if k.startswith("sigreg") else 1 - weight) for k in parts}
        found = families(self.model)
        video = [p for p in found[VIDEO_LORA] if p.requires_grad]
        pose = [p for p in found[POSE] if p.requires_grad]
        out: dict[str, float] = {}
        video_terms = {k: v for k, v in parts.items() if k not in POSE_TERMS}
        if video and video_terms:
            grads = term_gradients(video_terms, weights, video)
            if grads:
                out |= shares_and_cosines(grads, "video")
                shares = [v for k, v in out.items() if k.startswith("video_share_")]
                out["video_share_max"] = max(shares)
                physical = [grads[k] for k in ("e_fis",) if k in grads]
                semantic = [grads[k] for k in grads if k != "e_fis"]
                if physical and semantic:
                    a, b = torch.stack(physical).sum(0), torch.stack(semantic).sum(0)
                    denominator = a.norm() * b.norm()
                    out["video_cos_physical_semantic"] = (
                        float(a @ b / denominator) if float(denominator) > 0 else float("nan")
                    )
        pose_terms = {k: v for k, v in parts.items() if k in POSE_TERMS}
        if pose and pose_terms:
            out |= shares_and_cosines(term_gradients(pose_terms, weights, pose), "pose")
        predicted = record.get("predicted")
        semantic_term = next((k for k in ("e_sem", "infonce") if k in parts), None)
        if (
            predicted is not None
            and semantic_term
            and "sigreg_sem" in parts
            and predicted.requires_grad
        ):
            a = torch.autograd.grad(parts[semantic_term], predicted, retain_graph=True)[0]
            b = torch.autograd.grad(parts["sigreg_sem"], predicted, retain_graph=True)[0]
            a, b = a.flatten().float(), b.flatten().float()
            denominator = a.norm() * b.norm()
            out["y_cos_sem_sigreg"] = (
                float(a @ b / denominator) if float(denominator) > 0 else float("nan")
            )
        return out

    def _level_readings(self, clips: WorldSignBatch, record: dict[str, Any]) -> dict[str, float]:
        """Collapse of the pose target; the physical read-outs and keypoint errors once the
        physical level runs; the semantic level's readings."""
        out: dict[str, float] = {}
        if "latent" in record:
            latent, confidence = record["latent"].detach(), record["confidence"]
            rows = latent[confidence > 0]
            spread = Spread.of(rows)
            out["s_rank"], out["s_std"] = spread.effective_rank, spread.std
            out["s_isoscore"] = spread.isoscore
            out["s_sigreg"] = sigreg_ratio(rows)
            pose = self.model.pose
            if "physical" in record and pose is not None:
                predictions = record["physical"].predictions
                out |= physical_readings(predictions, latent, confidence)
                out["dynamics_margin"] = out["dynamics_r2"] - out["dynamics_baseline_r2"]
                errors = keypoint_errors(
                    predictions, latent, pose.decoder, clips.keypoints, clips.keypoint_weights
                )
                out |= errors
                baselines = [
                    errors[k]
                    for k in ("keypoint_error_interpolation", "keypoint_error_constant_velocity")
                    if not math.isnan(errors[k])
                ]
                if baselines and not math.isnan(errors["keypoint_error_model"]):
                    out["keypoint_margin"] = min(baselines) - errors["keypoint_error_model"]
        if "predicted" in record:
            predicted, target = record["predicted"].detach(), record["target"].detach()
            names = list(self.model.text.centering.languages)
            out |= semantic_readings(predicted, target, clips.languages, names)
            out["y_rank"] = Spread.of(predicted.flatten(0, 1)).effective_rank
            out["encoder_rank"] = Spread.of(record["encoder_mean"]).effective_rank
            out["query_cosine"] = query_cosine(record["queries"])
            centred = self.model.text.centering(clips.captions, clips.languages)
            out["text_head_spearman"] = text_head_spearman(centred, target)
        return out

    def _lora_readings(self) -> dict[str, float]:
        ratios: dict[str, float] = {}
        ratios |= lora_ratios(self.model.video.backbone.encoder)
        if self.model.video.physical_predictor is not None:
            ratios |= lora_ratios(self.model.video.physical_predictor.predictor)
        return {"lora_ratio_max": max(ratios.values())} if ratios else {}

    def localization(self, clips: WorldSignBatch, randomness: StepRandomness) -> float:
        """How much E_fis grows when a box moves to a masked region without its articulator."""
        video, pose = self.model.video, self.model.pose
        physical = video.physical_predictor
        if physical is None or pose is None:
            return float("nan")
        grid = video.grid
        mask = video.masks(len(clips.frames), randomness.masks)[-1].to(self.device)
        levels = video.backbone.context_levels(clips.frames, mask.context)
        tokens = physical.tokens(levels, mask).view(
            -1, grid.steps, grid.rows, grid.columns, physical.width
        )
        roles = token_roles(mask, grid)
        masked = roles.masked.view(-1, grid.steps, 1, grid.rows, grid.columns)
        members = membership(clips.boxes, clips.box_visible, grid.rows, grid.columns)
        shifted = torch.roll(members, (grid.rows // 2, grid.columns // 2), dims=(-2, -1)) * (
            1 - members
        )
        target = functional.layer_norm(
            pose.target(clips.pose_tokens).float(), (physical.readout.target_dim,)
        )
        errors = []
        for boxes in (members * masked, shifted * masked):
            reading = physical.readout(tokens, boxes).float()
            weight = boxes.sum((-1, -2, -3)) > 0
            errors.append((reading - target).abs().mean(-1)[weight].mean())
        both = (
            bool(torch.isfinite(errors[0]))
            and bool(torch.isfinite(errors[1]))
            and float(errors[0]) > 0
        )
        return float(errors[1] / errors[0] - 1) if both else float("nan")

    def _probe_latent(self) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        pose = self.model.pose
        if pose is None or not self.probe:
            return None
        latents, keypoints, weights, features = [], [], [], []
        for batch in self.probe:
            clips = batch.to(self.device)
            latents.append(pose.target(clips.pose_tokens).float().cpu())
            keypoints.append(clips.keypoints.float().cpu())
            weights.append(clips.keypoint_weights.float().cpu())
            tokens = self.model.video.backbone.tokens(clips.frames).float()
            grid = self.model.video.grid
            features.append(
                tokens.view(len(tokens), grid.steps, -1, tokens.shape[-1]).mean(2).cpu()
            )
        return torch.cat(latents), torch.cat(keypoints), torch.cat(weights), torch.cat(features)

    def _probe_readings(self, step: int) -> dict[str, float]:
        """Isotropy, kinematic content per articulator and speed of the pose target on the fixed
        probe batch. The R² of each articulator is compared with its best so far: a target
        trained from scratch must gain kinematics, and must not lose what it has gained."""
        found = self._probe_latent()
        if found is None:
            return {}
        latent, keypoints, weights, features = found
        out: dict[str, float] = {}
        if self.references.latent is None:
            self.references.latent = latent
            self.references.features = features
            pose_rows = keypoints.flatten(2).flatten(0, 1)
            feature_rows = features.flatten(0, 1)
            everything = torch.ones(len(pose_rows), dtype=torch.bool)
            explained = self._ridge_predict(pose_rows, feature_rows, everything)
            self.references.residual = feature_rows - explained
        reference = self.references.latent
        target = pose_target_measures(latent, keypoints, weights)
        out["probe_isoscore"] = target["isoscore"]
        out["probe_rank"] = target["rank"]
        out["probe_sigreg"] = target["sigreg"]
        out["cka_s"] = linear_cka(latent.flatten(0, 1), reference.flatten(0, 1))
        for name in PARTS:
            for kind in ("position", "velocity"):
                key = f"pose_r2_{kind}_{name}"
                value = out[key] = target[f"r2_{kind}_{name}"]
                best = self.references.values.get(f"best_{key}")
                if best is not None:
                    out[f"pose_r2_drop_{kind}_{name}"] = best - value
                if not math.isnan(value) and (best is None or value > best):
                    self.references.values[f"best_{key}"] = value
        return out

    def _encoder_drift(self) -> float | None:
        """R² with which the adapted encoder predicts the part of its original features the
        keypoints do not explain (§4.13.3). < 0.5: the encoder is flattening onto the pose."""
        found = self._probe_latent()
        residual = self.references.residual
        if found is None or residual is None:
            return None
        features = found[3].flatten(0, 1)
        clips = len(found[0])
        fit = (torch.arange(clips) < clips // 2)[:, None].expand(clips, found[0].shape[1]).flatten()
        return ridge_r2(features, residual, torch.ones(len(features)), fit)

    @staticmethod
    def _ridge_predict(
        features: Tensor, targets: Tensor, fit: Tensor, penalty: float = 1e-3
    ) -> Tensor:
        x = features.double()
        mean, std = x[fit].mean(0), x[fit].std(0).clamp_min(1e-8)
        x = torch.cat([(x - mean) / std, torch.ones(len(x), 1, dtype=x.dtype)], 1)
        gram = x[fit].T @ x[fit]
        ridge = penalty * gram.diagonal()[:-1].mean() * torch.eye(len(gram), dtype=x.dtype)
        ridge[-1, -1] = 0.0
        coefficients = torch.linalg.solve(gram + ridge, x[fit].T @ targets.double()[fit])
        predicted: Tensor = (x @ coefficients).float()
        return predicted

    # ------------------------------------------------------------ alarms and stops

    def _record(self, kind: str, step: int, readings: dict[str, float]) -> dict[str, float]:
        for name, value in readings.items():
            self.history[name].append((step, value))
            if step == 0 or name not in self.references.values:
                self.references.values.setdefault(name, value)
        self.log.write(kind, step=step, **{k: v for k, v in readings.items() if not math.isnan(v)})
        stop = False
        for alarm in self._alarms(step, readings):
            self.alarms.append(alarm)
            self.log.write(
                "alarm",
                step=step,
                name=alarm.name,
                value=alarm.value,
                threshold=alarm.threshold,
                meaning=alarm.meaning,
                stop=alarm.stop,
            )
            stop |= alarm.stop
        if self.collective.any(stop):
            raise RunStoppedError(f"step {step}: {[a.meaning for a in self.alarms if a.stop]}")
        return readings

    def _alarms(self, step: int, readings: dict[str, float]) -> list[Alarm]:
        found = []
        for rule in self.rules:
            names = [
                n
                for n in readings
                if n == rule.metric
                or (rule.metric.endswith("*") and n.startswith(rule.metric[:-1]))
            ]
            for name in names:
                value = readings[name]
                if math.isnan(value):
                    continue
                if rule.kind.endswith("ratio"):
                    reference = self.references.values.get(name)
                    if reference is None or reference == 0 or step == 0:
                        continue
                    value = value / reference
                beyond = (
                    value < rule.threshold
                    if rule.kind.startswith("below")
                    else value > rule.threshold
                )
                self.streaks[name] = self.streaks[name] + 1 if beyond else 0
                if beyond and self.streaks[name] >= rule.readings:
                    found.append(Alarm(name, step, value, rule.threshold, rule.meaning, rule.stop))
        return found

    def _latest(self, name: str) -> float:
        values = self.history.get(name)
        return values[-1][1] if values else float("nan")

    def _falling(self, name: str) -> bool:
        values = [v for _, v in self.history.get(name, [])]
        if len(values) < 4:  # noqa: PLR2004
            return False
        quarter = max(1, len(values) // 4)
        return sum(values[-quarter:]) / quarter < sum(values[:quarter]) / quarter

    def _pose_criteria(self, add: Callable[[str, bool, str], None]) -> None:
        """F1, the pose: the target learnt, spread and kinematic (posa §6)."""
        s = self.settings
        add("L_anchor falling", self._falling("term_anchor"), "first against last quarter")
        add("SIGReg_posa falling", self._falling("term_sigreg_posa"), "first against last quarter")
        rank, start = self._latest("s_rank"), self.references.values.get("s_rank", math.nan)
        add(
            "rank of s > 0.5 x step 0", rank > s.collapse_ratio * start, f"{rank:.1f} / {start:.1f}"
        )
        iso = self._latest("probe_isoscore")
        add(f"IsoScore of s >= {s.isoscore_min}", iso >= s.isoscore_min, f"{iso:.3f}")
        for part in ("left", "right"):
            r2 = self._latest(f"pose_r2_position_{part}")
            add(
                f"position R² of the {part} hand from s >= {s.pose_r2_min}",
                r2 >= s.pose_r2_min,
                f"{r2:.3f}",
            )

    def _semantic_criteria(self, add: Callable[[str, bool, str], None]) -> None:
        """F1, the semantic level on the encoder as released."""
        s = self.settings
        gamma = self._latest("gamma_sem")
        add("gamma_sem > 0.3", gamma > s.gamma_min, f"{gamma:.3f}")
        add(
            "SIGReg_sem falling",
            self._falling("term_sigreg_sem") or "term_sigreg_sem" not in self.history,
            "",
        )
        decision, chance = self._latest("decision"), self._latest("chance")
        add(
            "R@1 held-out > 5 x chance",
            decision > s.chance_multiple * chance,
            f"{decision:.4f} vs chance {chance:.4f}",
        )
        drop = self._latest("noise_drop")
        add("noise test passed", drop >= s.noise_drop_min, f"R@1 drop {drop:.2f}")
        cosine = self._latest("query_cosine")
        add("queries not collapsed", not cosine > s.query_cosine_max, f"{cosine:.3f}")

    def _progress_criteria(
        self, add: Callable[[str, bool, str], None], extrapolated_r1: float | None
    ) -> None:
        """F3, after the LoRA entered: retrieval on track, order, hubness, no conflict."""
        s = self.settings
        decision = self._latest("decision")
        if s.ridge_baseline_r1 is not None:
            add(
                "R@1 > ridge baseline",
                decision > s.ridge_baseline_r1,
                f"{decision:.4f} vs {s.ridge_baseline_r1:.4f}",
            )
        order = self._latest("order_cosine")
        add("omega < 0.95", not order > s.order_max, f"{order:.3f}")
        hub, start = (
            self._latest("hubness"),
            self.references.values.get("hubness", float("nan")),
        )
        add("hubness stable", not hub > s.hubness_growth_max * start, f"{hub:.2f} vs {start:.2f}")
        conflict = any(
            self.streaks[k] >= s.conflict_readings
            for k in ("video_cos_physical_semantic", "y_cos_sem_sigreg")
        )
        add("no stable gradient conflict", not conflict, "")
        if extrapolated_r1 is not None:
            from ..evaluation.gate import GatePolicy  # noqa: PLC0415

            add(
                "extrapolated R@1 compatible with X",
                GatePolicy().on_track(100 * extrapolated_r1),
                f"{100 * extrapolated_r1:.1f} (held-out channel, proxy of OpenASL)",
            )

    def stop_point(self, name: str, step: int, extrapolated_r1: float | None = None) -> StopReport:
        """The criteria of F1, F2 or F3 (gerarchia §7) on the latest readings."""
        s = self.settings
        criteria: dict[str, tuple[bool, str]] = {}

        def add(label: str, ok: bool, detail: str) -> None:
            criteria[label] = (bool(ok), detail)

        if name == "F1":
            if self.model.pose is not None:
                self._pose_criteria(add)
            self._semantic_criteria(add)
        elif name == "F2":
            add("E_fis falling", self._falling("term_e_fis"), "first against last quarter")
            r2 = self._latest("r2_visible")
            add("R² of mostly visible steps > 0.9", r2 > s.visible_r2_min, f"{r2:.3f}")
            margin = self._latest("dynamics_margin")
            add("dynamics beats the baseline", margin > 0, f"margin {margin:.3f}")
            leak = self._latest("leak_change")
            add("no leak", not leak > s.leak_tolerance, f"{leak:.2e}")
        elif name == "F3":
            if self.model.pose is not None:
                add("E_fis falling", self._falling("term_e_fis"), "first against last quarter")
                lora = self._latest("lora_ratio_max")
                add(
                    f"no LoRA moves its layer by more than {s.lora_ratio_max:.0%}",
                    not lora > s.lora_ratio_max,
                    f"{lora:.3f}",
                )
                drift = self._latest("encoder_drift_r2")
                add(f"encoder drift R² >= {s.drift_min}", not drift < s.drift_min, f"{drift:.3f}")
            self._progress_criteria(add, extrapolated_r1)
        report = StopReport(name, step, criteria)
        self.log.write(
            "stop",
            step=step,
            name=name,
            passed=report.passed,
            criteria={k: {"ok": ok, "detail": d} for k, (ok, d) in criteria.items()},
        )
        return report

    def extrapolate(self, metric: str = "decision") -> float | None:
        """The metric at the last planned step, from a log-linear fit of its history."""
        points = [(s, v) for s, v in self.history.get(metric, []) if s > 0 and not math.isnan(v)]
        if len(points) < 2:  # noqa: PLR2004
            return None
        x = torch.tensor([math.log(s) for s, _ in points], dtype=torch.float64)
        y = torch.tensor([v for _, v in points], dtype=torch.float64)
        slope = float(
            ((x - x.mean()) * (y - y.mean())).sum() / (x - x.mean()).pow(2).sum().clamp_min(1e-12)
        )
        intercept = float(y.mean() - slope * x.mean())
        return intercept + slope * math.log(self.total_steps)

    def state(self) -> dict[str, Any]:
        return {"references": self.references, "history": dict(self.history)}

    def load(self, saved: dict[str, Any]) -> None:
        self.references = saved["references"]
        self.history = defaultdict(list, saved["history"])
