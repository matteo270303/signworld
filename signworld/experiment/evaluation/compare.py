"""The ablation tables: runs compared query by query on the same clips (§4.14, §5.3).

Every run is evaluated on the same split with ``train evaluate``, which keeps the rank of each
query's first match; ``compare_runs`` recognises each run's place in the plan from its
configuration (an ESP-1 arm, or an ablation on θ*), picks θ* as the ESP-1 arm with the best
validation metric (the metric that decides, never the test), and for each planned contrast
reads the difference of R@1 with a **paired bootstrap over the queries**: both runs are
resampled on the same queries, so the variability of the clips cancels out. The relation it
finds (≫, >, ≈, <, ≪) is set against the prediction fixed in advance (progetto §5.3), where
one exists. With one seed per run the interval measures the clips, not the training; the
second seed of θ* (D4) gives the spread between runs to read the gaps against.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import torch

from .report import analyse_run

ESP1: Final = ("A0", "A", "V", "B0", "B", "C")


@dataclass(frozen=True, slots=True)
class Contrast:
    first: str
    second: str
    question: str
    prediction: str | None
    """The relation of ``first`` to ``second`` fixed in advance, or None (not fixed)."""


PLAN: Final = (
    Contrast("A", "A0", "what SIGReg adds", "≫"),
    Contrast("A", "V", "the whole distribution (SIGReg) against its second moments (VICReg)", None),
    Contrast("A", "B0", "distributional constraint against pairwise uniformity (H1)", "≈"),
    Contrast("B0", "A0", "what pairwise uniformity adds", ">"),
    Contrast("B", "A", "do SIGReg and L_unif add up", "≈"),
    Contrast("A", "C", "what negatives add at the same batch and data", "≈"),
    Contrast("θ*", "ESP-2", "does the physical level help the semantic one (H3)", ">"),
    Contrast("θ*", "ESP-6", "the pose as the physical target, against the video (H3)", None),
    Contrast("θ*", "G1", "the hierarchy level by level, against end to end", None),
    Contrast("ESP-4", "θ*", "does caption ambiguity need a latent variable (H7)", "≈"),
    Contrast("D4", "θ*", "the spread between two seeds of θ*", "≈"),
)
"""The planned readings; predictions from progetto §5.3 (ESP-4: «small or no gain»)."""


@dataclass(frozen=True, slots=True)
class Result:
    first: str
    second: str
    question: str
    prediction: str | None
    first_r1: float
    second_r1: float
    delta: float
    low: float
    high: float
    relation: str
    verdict: str


def place(config: dict[str, Any]) -> str:
    """The run's place in the plan, from its configuration."""
    if config["training"]["seed"] != 0:
        return "D4"
    if not config["physical"]["enabled"]:
        return "ESP-2"
    if config["physical"].get("target", "pose") == "video":
        return "ESP-6"
    if config["semantic"]["trains_encoder"]:
        return "G1"
    if config["semantic"]["hypotheses"] > 1:
        return "ESP-4"
    arm: str = config["losses"]["arm"]
    return arm


def _hits(ranks: dict[str, list[int]], k: int) -> torch.Tensor:
    """(queries, 2): whether each query's first match is in the top k, both directions."""
    return torch.stack(
        [torch.tensor(ranks["t2v"]) < k, torch.tensor(ranks["v2t"]) < k], dim=1
    ).float()


def paired_bootstrap(
    first: dict[str, list[int]],
    second: dict[str, list[int]],
    k: int = 1,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Difference of the mean R@k of the two directions, with a 95 % paired interval."""
    a, b = _hits(first, k), _hits(second, k)
    if a.shape != b.shape:
        raise ValueError("the two runs were not evaluated on the same queries")
    difference = (a - b).mean(dim=1)
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(difference), (samples, len(difference)), generator=generator)
    means = difference[draws].mean(dim=1)
    return (
        float(difference.mean()),
        float(means.quantile(0.025)),
        float(means.quantile(0.975)),
    )


def relation(delta: float, low: float, high: float, second: float) -> str:
    """≈ when the interval holds 0; > or <; ≫ or ≪ when also beyond half the second's R@1."""
    if low <= 0 <= high:
        return "≈"
    large = abs(delta) >= 0.5 * max(second, 1e-12)
    if delta > 0:
        return "≫" if large else ">"
    return "≪" if large else "<"


def verdict(prediction: str | None, found: str) -> str:
    if prediction is None:
        return "no prediction fixed in advance"
    agrees = found == prediction or (prediction == ">" and found == "≫")
    return "as predicted" if agrees else "against the prediction"


def compare_runs(runs: Sequence[tuple[Path, Path]], k: int = 1, seed: int = 0) -> dict[str, Any]:
    """``runs``: (run directory, its evaluation JSON) pairs, all on the same split."""
    loaded: dict[str, dict[str, Any]] = {}
    for directory, evaluation in runs:
        config = json.loads((directory / "config.json").read_text())
        report = json.loads(evaluation.read_text())
        name = place(config)
        if name in loaded:
            raise ValueError(f"two runs take the place {name}: {directory}")
        validation = analyse_run(directory)["validation"]
        loaded[name] = {
            "directory": str(directory),
            "rows": report["rows"],
            "ranks": report["ranks"],
            "decision": validation.get("best", {}).get("decision", math.nan),
        }
    rows = {tuple(r["rows"]) for r in loaded.values()}
    if len(rows) > 1:
        raise ValueError("the runs were not evaluated on the same clips in the same order")
    arms = {name: r for name, r in loaded.items() if name in ESP1}
    theta = max(arms, key=lambda name: arms[name]["decision"]) if arms else None
    if theta is not None:
        loaded["θ*"] = loaded[theta]
    results = []
    for contrast in PLAN:
        if contrast.first not in loaded or contrast.second not in loaded:
            continue
        a, b = loaded[contrast.first], loaded[contrast.second]
        delta, low, high = paired_bootstrap(a["ranks"], b["ranks"], k, seed=seed)
        second_r1 = float(_hits(b["ranks"], k).mean())
        found = relation(delta, low, high, second_r1)
        results.append(
            Result(
                contrast.first,
                contrast.second,
                contrast.question,
                contrast.prediction,
                float(_hits(a["ranks"], k).mean()),
                second_r1,
                delta,
                low,
                high,
                found,
                verdict(contrast.prediction, found),
            )
        )
    return {
        "k": k,
        "theta": theta,
        "theta_chosen_by": "the best validation metric among the ESP-1 arms",
        "runs": {name: r["directory"] for name, r in loaded.items()},
        "results": [asdict(r) for r in results],
    }


def markdown(comparison: dict[str, Any]) -> str:
    k = comparison["k"]
    lines = [
        f"# Ablations: mean R@{k} of the two directions, paired bootstrap over the queries",
        "",
        f"θ* = {comparison['theta']} ({comparison['theta_chosen_by']}).",
        "",
        "| contrast | question | first | second | Δ [95 %] | found | predicted | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in comparison["results"]:
        lines.append(
            f"| {r['first']} vs {r['second']} | {r['question']} | {r['first_r1']:.4f} "
            f"| {r['second_r1']:.4f} | {r['delta']:+.4f} [{r['low']:+.4f}, {r['high']:+.4f}] "
            f"| {r['relation']} | {r['prediction'] or '-'} | {r['verdict']} |"
        )
    return "\n".join(lines) + "\n"


def write_comparison(comparison: dict[str, Any], output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    as_json, as_markdown = output / "comparison.json", output / "comparison.md"
    as_json.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")
    as_markdown.write_text(markdown(comparison))
    return as_json, as_markdown
