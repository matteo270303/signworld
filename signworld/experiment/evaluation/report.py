"""The report of one run: what it is, how it trained, what it measured, what to fix.

``analyse_run`` reads what a run leaves in its directory (``provenance.json``, ``config.json``,
``preflight.json``, ``metrics.jsonl``) and, when given, its final evaluation
(``train evaluate``) and the ridge baseline of its splits (``train ridge-baseline``). It writes
``report.json`` and ``report.md`` with:

1. **identity**: the run's arm and ablation, seed and configuration hash, and every launch
   with its code (commit, uncommitted changes) and machine;
2. **training**: steps, stages, cooldown, each loss term at the start and at the end;
3. **validation and stops**: the metric that decides over time, the programmed stops F1-F3;
4. **evaluation**: retrieval both ways with bootstrap intervals against the ridge baseline,
   the readings of the run's contribution (the distribution of ŷ and ẽ for the loss arms, the
   physical level for the ablations on θ*), plausibility, temporal order, where the first
   results come from, the language probes;
5. **diagnosis**: every alarm that fired, with the triage of §4.13.6: what it means, what to
   check first, what to adjust.
"""

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


@dataclass(frozen=True, slots=True)
class Triage:
    check: str
    adjust: str


TRIAGE: Final[dict[str, Triage]] = {
    "leak_change": Triage(
        "P6 (masked tokens removed), P14 (pose isolated), the leak test itself",
        "fix the masking pipeline before anything else: the run is stopped",
    ),
    "s_": Triage(
        "R² of the keypoints from s per articulator -> the anchor -> SIGReg and rank of s -> "
        "the cosine anchor-SIGReg",
        "lower the pose learning rate; reweigh the pose terms; the two-stage fallback",
    ),
    "pose_": Triage(
        "R² of the keypoints from s on the probe batch; the anchor's loss",
        "lower the pose learning rate or strengthen the anchor",
    ),
    "y_rank": Triage(
        "gamma_sem -> SIGReg_sem (or VICReg / L_unif in its arm) -> the query outputs",
        "expected without a regulariser (A0); otherwise the regulariser's weight",
    ),
    "text_effective_rank": Triage(
        "the text head's Spearman, SIGReg on ẽ",
        "the regulariser on ẽ; the text head's learning rate",
    ),
    "gamma_sem": Triage(
        "R² of ẽ from ŷ in training and validation, per language",
        "ambiguous captions (ESP-4) or under-fitting (capacity, learning rate)",
    ),
    "gamma_masked": Triage(
        "video-pose sync -> R² of the mostly visible steps -> R² by mask coverage",
        "pose or box pipeline; read-out",
    ),
    "dynamics_margin": Triage(
        "R² of the mostly hidden steps against the mean of the visible ones",
        "longer F stage, physical predictor capacity",
    ),
    "keypoint_margin": Triage(
        "keypoint errors of the model, of the decoder floor and of the baselines",
        "if the decoder floor is high, the anchor's decoder; else the physical level",
    ),
    "localization_increase": Triage(
        "E_fis with the box moved to a masked region without the articulator (§4.6)",
        "add the penalty on articulator-like predictions outside the box",
    ),
    "order_cosine": Triage(
        "the 3D-RoPE of the semantic predictor; time-reversed retrieval",
        "position encoding of the semantic predictor",
    ),
    "encoder_drift_r2": Triage(
        "encoder drift -> conflict of gradients -> gradient shares",
        "reduce the weight of E_fis at the next stop",
    ),
    "video_": Triage(
        "the gradient shares and cosines on the video LoRA",
        "rebalance the levels (only G1 shares the LoRA)",
    ),
    "y_cos_sem_sigreg": Triage(
        "PC1: anisotropy of the centred caption targets",
        "centring per language; the text head",
    ),
    "text_head_spearman": Triage(
        "Spearman of the caption similarities before and after the head",
        "lower the text head's learning rate (VL-JEPA: x0.05-0.10 on the Y-encoder)",
    ),
    "lora_": Triage("‖ΔW‖/‖W‖ and the gradient norm per block", "the LoRA learning rate"),
    "modality_gap": Triage(
        "the logistic classifier ŷ against ẽ; the alignment",
        "the regulariser per modality; the alignment weight",
    ),
    "hubness_": Triage(
        "in T2V ŷ pulled to the centre (gamma_sem); in V2T a few ẽ attract the clips",
        "the regulariser's weight; ESP-4 when the captions are ambiguous",
    ),
    "noise_drop": Triage(
        "R@1 with the video replaced by noise", "a text shortcut: the semantic predictor's input"
    ),
    "excluded_part": Triage(
        "share of steps whose hand box is excluded for low confidence",
        "the box threshold (F4: 1.0 instead of 0.3)",
    ),
    "query_cosine": Triage("cosine between the 8 query outputs", "query initialisation; dropout"),
    "attention_on_boxes_ratio": Triage(
        "the queries' attention on the articulators against the background",
        "more crop and jitter; fewer free parameters",
    ),
    "loss_spikes": Triage(
        "bf16 overflow -> gradient share per term -> SIGReg's values",
        "SIGReg in fp32; a lower learning rate",
    ),
}
"""Alarm (name or prefix) -> what to check first and what to adjust (progetto §4.13.6)."""

META: Final = frozenset({"kind", "time", "step", "epoch", "stage", "reason", "name"})


def triage(name: str) -> Triage | None:
    """The triage of an alarm, by its name or the longest prefix listed."""
    found = [key for key in TRIAGE if name == key or name.startswith(key)]
    return TRIAGE[max(found, key=len)] if found else None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _identity(config: dict[str, Any] | None, provenance: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if config is not None:
        out |= {
            "name": config["name"],
            "arm": config["losses"]["arm"],
            "physical": config["physical"]["enabled"],
            "physical_target": config["physical"].get("target", "pose"),
            "semantic_trains_encoder": config["semantic"]["trains_encoder"],
            "hypotheses": config["semantic"]["hypotheses"],
            "seed": config["training"]["seed"],
            "batch_size": config["training"]["batch_size"],
            "epochs": config["training"]["epochs"],
            "index": config["data"]["index"],
        }
    if provenance is not None:
        out["config_sha256"] = provenance["config_sha256"]
        out["launches"] = [
            {
                "started": segment["started"],
                "resumed": segment["resumed"],
                "commit": segment["code"]["commit"],
                "dirty": segment["code"]["dirty"],
                "source_sha256": segment["code"]["source_sha256"][:12],
                "code_change_allowed": segment["code_change_allowed"],
                "gpus": segment["environment"]["gpus"],
                "host": segment["environment"]["host"],
                "job": segment["environment"]["job"],
                "torch": segment["environment"]["torch"],
            }
            for segment in provenance["segments"]
        ]
    return out


def _quarters(values: list[tuple[int, float]]) -> dict[str, float]:
    """Mean of the first and of the last quarter of a series."""
    finite = [v for _, v in values if isinstance(v, int | float) and math.isfinite(v)]
    if len(finite) < 4:  # noqa: PLR2004 (a quarter needs one value)
        return {"first": math.nan, "last": math.nan}
    quarter = len(finite) // 4
    return {
        "first": sum(finite[:quarter]) / quarter,
        "last": sum(finite[-quarter:]) / quarter,
    }


def _training(records: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [r for r in records if r["kind"] == "step"]
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for record in steps:
        for key, value in record.items():
            if key not in META and isinstance(value, int | float) and not key.startswith("lr_"):
                series[key].append((record["step"], float(value)))
    stages = [(r["step"], r["stage"]) for r in records if r["kind"] == "stage"]
    cooldown = next((r for r in records if r["kind"] == "cooldown"), None)
    return {
        "last_step": steps[-1]["step"] if steps else 0,
        "stages": stages,
        "cooldown": None
        if cooldown is None
        else {"step": cooldown["step"], "why": cooldown["reason"]},
        "spikes": sum(1 for r in records if r["kind"] == "spike"),
        "terms": {name: _quarters(values) for name, values in sorted(series.items())},
    }


def _validation(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in records if r["kind"] == "validation" and "decision" in r]
    if not rows:
        return {}
    best = max(rows, key=lambda r: r["decision"])
    keep = (
        "decision",
        "t2v_r1",
        "v2t_r1",
        "t2v_r1_low",
        "t2v_r1_high",
        "v2t_r1_low",
        "v2t_r1_high",
    )
    return {
        "curve": [(r["step"], r["decision"]) for r in rows],
        "best": {"step": best["step"], **{k: best[k] for k in keep if k in best}},
        "last": {"step": rows[-1]["step"], **{k: rows[-1][k] for k in keep if k in rows[-1]}},
    }


def _stops(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": r["name"],
            "step": r["step"],
            "passed": r["passed"],
            "failed": [k for k, v in r["criteria"].items() if not v["ok"]],
        }
        for r in records
        if r["kind"] == "stop"
    ]


def _alarms(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["kind"] == "alarm":
            grouped[record["name"]].append(record)
    out = []
    for name, fired in grouped.items():
        found = triage(name)
        out.append(
            {
                "name": name,
                "count": len(fired),
                "first_step": fired[0]["step"],
                "last_step": fired[-1]["step"],
                "last_value": fired[-1]["value"],
                "threshold": fired[-1]["threshold"],
                "meaning": fired[-1]["meaning"],
                "stopped": any(r.get("stop") for r in fired),
                "check": None if found is None else found.check,
                "adjust": None if found is None else found.adjust,
            }
        )
    return sorted(out, key=lambda a: a["first_step"])


def _pick(measures: dict[str, float], prefixes: Iterable[str]) -> dict[str, float]:
    return {k: v for k, v in measures.items() if k.startswith(tuple(prefixes))}


def _evaluation(
    evaluation: dict[str, Any] | None, ridge: dict[str, Any] | None, split: str | None
) -> dict[str, Any]:
    if evaluation is None:
        return {}
    m = evaluation["measures"]
    out: dict[str, Any] = {
        "retrieval": _pick(m, ("t2v_", "v2t_", "tolerant_", "chance", "decision")),
        "distribution": _pick(
            m,
            ("y_dist_", "text_dist_", "y_sigreg", "text_sigreg", "y_isoscore", "text_isoscore"),
        )
        | _pick(m, ("y_effective_rank", "text_effective_rank", "alignment", "uniformity_")),
        "hubness": _pick(m, ("hubness_", "modality_gap", "noise_")),
        "physical": _pick(
            m,
            ("r2", "gamma_", "dynamics_", "keypoint_", "token_", "pose_", "leak_", "e_fis"),
        ),
        "order_cosine": m.get("order_cosine"),
        "plausibility": evaluation["plausibility"],
        "language_probes": evaluation["language_probes"],
        "gate": evaluation["gate"],
    }
    if ridge is not None and split is not None and split in ridge["splits"]:
        baseline = ridge["splits"][split]
        out["ridge_baseline"] = {
            "t2v_r1": baseline["t2v_r1"]["estimate"],
            "v2t_r1": baseline["v2t_r1"]["estimate"],
            "gallery": baseline["gallery"],
        }
    return out


def analyse_run(
    run: Path,
    evaluation: Path | None = None,
    ridge: Path | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Everything the run's directory and its evaluation tell (module docstring)."""
    records = _records(run / "metrics.jsonl")
    preflight = _read_json(run / "preflight.json") or []
    return {
        "run": str(run),
        "identity": _identity(_read_json(run / "config.json"), _read_json(run / "provenance.json")),
        "preflight_failures": [a for a in preflight if a.get("status") == "fail"],
        "training": _training(records),
        "validation": _validation(records),
        "stops": _stops(records),
        "alarms": _alarms(records),
        "evaluation": _evaluation(
            None if evaluation is None else _read_json(evaluation),
            None if ridge is None else _read_json(ridge),
            split,
        ),
        "evaluation_split": split,
    }


def _number(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def _table(rows: dict[str, Any]) -> list[str]:
    lines = ["| reading | value |", "|---|---|"]
    lines += [f"| {k} | {_number(v)} |" for k, v in rows.items()]
    return lines


def markdown(report: dict[str, Any]) -> str:
    """The report as Markdown, section by section."""
    identity = report["identity"]
    lines = [f"# Run {identity.get('name', report['run'])}", "", "## Identity", ""]
    lines += _table({k: v for k, v in identity.items() if k != "launches"})
    for launch in identity.get("launches", []):
        lines.append(
            f"- {launch['started']}: commit {launch['commit']} (dirty {launch['dirty']}, "
            f"source {launch['source_sha256']}), {len(launch['gpus'])} GPU {launch['gpus'][:1]}, "
            f"host {launch['host']}, torch {launch['torch']}"
            + (", **code changed on resume**" if launch["code_change_allowed"] else "")
        )
    if report["preflight_failures"]:
        lines += [
            "",
            "**Preflight failures:** " + ", ".join(a["code"] for a in report["preflight_failures"]),
        ]
    training = report["training"]
    lines += [
        "",
        "## Training",
        "",
        f"Steps: {training['last_step']}; stages: {training['stages']}; "
        f"cooldown: {training['cooldown']}; loss spikes: {training['spikes']}",
        "",
    ]
    lines += ["| term | first quarter | last quarter |", "|---|---|---|"]
    lines += [
        f"| {name} | {_number(q['first'])} | {_number(q['last'])} |"
        for name, q in training["terms"].items()
    ]
    validation = report["validation"]
    if validation:
        lines += [
            "",
            "## Validation",
            "",
            f"Best: {validation['best']}",
            "",
            f"Last: {validation['last']}",
        ]
    if report["stops"]:
        lines += ["", "## Programmed stops", ""]
        lines += [
            f"- {s['name']} at step {s['step']}: {'passed' if s['passed'] else 'FAILED'}"
            + (f" (failed: {', '.join(s['failed'])})" if s["failed"] else "")
            for s in report["stops"]
        ]
    lines += ["", "## Diagnosis", ""]
    if not report["alarms"]:
        lines.append("No alarm fired.")
    else:
        lines += [
            "| alarm | steps | count | meaning | check first | adjust |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {a['name']}{' (stop)' if a['stopped'] else ''} "
            f"| {a['first_step']}-{a['last_step']} | {a['count']} | {a['meaning']} "
            f"| {a['check'] or '-'} | {a['adjust'] or '-'} |"
            for a in report["alarms"]
        ]
    evaluation = report["evaluation"]
    if evaluation:
        lines += ["", f"## Evaluation ({report['evaluation_split']})", ""]
        if "ridge_baseline" in evaluation:
            lines.append(f"Ridge baseline on the same split: {evaluation['ridge_baseline']}")
            lines.append("")
        for block in ("retrieval", "distribution", "hubness", "physical"):
            if evaluation[block]:
                lines += ["", f"### {block.capitalize()}", "", *_table(evaluation[block])]
        lines += ["", f"Temporal order ω: {_number(evaluation['order_cosine'])}"]
        if evaluation["plausibility"]:
            lines += [
                "",
                "### Plausibility",
                "",
                "| manipulation | expected | increase | share | passed |",
                "|---|---|---|---|---|",
            ]
            lines += [
                f"| {p['name']} | {p['expected']} | {_number(p['mean_increase'])} | "
                f"{_number(p['share_increased'])} | {p['passed']} |"
                for p in evaluation["plausibility"]
            ]
        lines += [
            "",
            f"Language probes: {evaluation['language_probes']}",
            f"Gate: {evaluation['gate']}",
        ]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    as_json, as_markdown = directory / "report.json", directory / "report.md"
    as_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    as_markdown.write_text(markdown(report))
    return as_json, as_markdown
