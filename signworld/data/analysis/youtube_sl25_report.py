"""Report on YouTube-SL-25: figures, tables and a diagnostic summary.

    python -m signworld.data.analysis.youtube_sl25_report \\
        --root /path/to/youtube-sl-25 --out reports/youtube_sl25

``--root`` holds ``youtube-sl-25-metadata.csv``, ``videos/`` and the downloader's
``failed_*.txt`` lists (and any ``failed_backup*`` directories). Writes ``report.md``,
``summary.json``, ``languages.csv``, ``anomalies.csv``, ``probe.parquet`` (a cache: later runs
probe only new or changed files) and the figures. Needs numpy, scipy, pyarrow, matplotlib and
``ffprobe``; the plotting import is local so the statistics stay usable without matplotlib.
"""

import argparse
import csv
import json
import logging
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from signworld.data.analysis import youtube_sl25 as sl

logger = logging.getLogger(__name__)

PRIMARY, SECONDARY, MUTED, ALERT = "#2b6a99", "#d98324", "#8c8c8c", "#b83b3b"
TEMPERATURES = (1.0, 0.7, 0.5, 0.3, 0.0)


# ── gathering ─────────────────────────────────────────────────────────────────────────────


class Study:
    """Everything the figures and the text read, computed once."""

    def __init__(self, root: Path, out: Path, workers: int, reuse_probe: bool) -> None:
        self.root, self.out = root, out
        out.mkdir(parents=True, exist_ok=True)
        rows = sl.read_release(root / "youtube-sl-25-metadata.csv")
        self.rows = rows
        self.repeated = sl.duplicate_ids(rows)
        self.language_of: dict[str, str] = {}
        for video_id, language in rows:
            self.language_of.setdefault(video_id, language)
        self.release_counts = dict(
            sorted(Counter(self.language_of.values()).items(), key=lambda kv: (-kv[1], kv[0]))
        )
        self.languages = list(self.release_counts)

        cache = out / "probe.parquet"
        if reuse_probe and cache.is_file():
            import pyarrow.parquet as pq

            self.probe = pq.read_table(cache).to_pylist()
        else:
            self.probe = sl.probe_directory(root / "videos", cache, workers)
        self.stray = [r["video_id"] for r in self.probe if r["video_id"] not in self.language_of]
        good = [r for r in self.probe if r["ok"] and r["video_id"] in self.language_of]
        bad = {r["video_id"] for r in self.probe if not r["ok"]}
        self.good = good
        self.failures = sl.read_failures([root, *sorted(root.glob("failed_backup*"))])
        self.state = sl.classify(
            self.language_of, {r["video_id"] for r in good}, bad, self.failures
        )
        self.language = np.array([self.language_of[r["video_id"]] for r in good])
        column = lambda name: np.array([r[name] for r in good], dtype=float)  # noqa: E731
        self.seconds = column("duration_s")
        self.width, self.height = column("width").astype(int), column("height").astype(int)
        self.fps, self.bit_rate, self.size = column("fps"), column("bit_rate"), column("size_bytes")
        self.bias = sl.retrieval_bias(self.languages, self.state, self.language_of)
        self.durations = sl.duration_summary(self.seconds)
        self.by_language = sl.duration_by_language(self.seconds, self.language)
        self.estimated = sl.estimated_release_hours(
            self.seconds, self.language, self.release_counts
        )
        self.anomalies = sl.anomalies(self.probe)
        self.duplicates = sl.near_duplicates(self.probe)
        self.subtitles = sl.subtitle_stats(
            root / "videos", {r["video_id"]: r["duration_s"] for r in good}
        )

    # one row per language, for the CSV and the tables
    def language_table(self) -> list[dict[str, object]]:
        table = []
        total = sum(self.release_counts.values())
        for lang, count in self.release_counts.items():
            mask = self.language == lang
            seconds = self.seconds[mask]
            states = Counter(self.state[v] for v, languages in self.language_of.items()
                             if languages == lang)  # fmt: skip
            resolved = states["downloaded"] + states["unavailable"]
            low, high = sl.wilson(states["downloaded"], resolved)
            estimate = self.estimated.get(lang, (float("nan"),) * 3)
            table.append(
                {
                    "language": lang,
                    "iso_639_3": bool(sl.ISO_639_3.match(lang)),
                    "release_videos": count,
                    "release_share": count / total,
                    **{state: states[state] for state in sl.STATES},
                    "availability": states["downloaded"] / resolved if resolved else float("nan"),
                    "availability_low": low,
                    "availability_high": high,
                    "hours": seconds.sum() / 3600,
                    "mean_min": seconds.mean() / 60 if seconds.size else float("nan"),
                    "median_min": np.median(seconds) / 60 if seconds.size else float("nan"),
                    "est_release_hours": estimate[0],
                    "est_release_hours_low": estimate[1],
                    "est_release_hours_high": estimate[2],
                }
            )
        return table


# ── figures ───────────────────────────────────────────────────────────────────────────────


def _axes(rows: int = 1, cols: int = 1, size: tuple[float, float] = (7, 4.5)):
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100}
    )  # fmt: skip
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False)
    return fig, axes


def _save(fig, out: Path, name: str) -> str:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    return name


def figure_composition(study: Study) -> str:
    languages = study.languages[::-1]
    fig, axes = _axes(size=(7.5, max(4.0, 0.22 * len(languages) + 1)))
    ax = axes[0][0]
    release = [study.release_counts[lang] for lang in languages]
    got = [sum(1 for v, lang2 in study.language_of.items()
               if lang2 == lang and study.state[v] == "downloaded") for lang in languages]  # fmt: skip
    ax.barh(languages, release, color=MUTED, alpha=0.45, label="release")
    ax.barh(languages, got, color=PRIMARY, label="downloaded")
    ax.set_xscale("log")
    ax.set_xlabel("videos (log scale)")
    ax.set_title("Videos per sign language")
    ax.legend(loc="lower right")
    return _save(fig, study.out, "01_composition.png")


def figure_balance(study: Study) -> str:
    counts = list(study.release_counts.values())
    fig, axes = _axes(1, 2, (9.5, 4.2))
    x, y = sl.lorenz(counts)
    axes[0][0].plot(x, y, color=PRIMARY)
    axes[0][0].plot([0, 1], [0, 1], color=MUTED, ls="--", label="perfect balance")
    axes[0][0].set(xlabel="share of languages (smallest first)", ylabel="share of videos",
                   title=f"Lorenz curve, Gini = {sl.gini(counts):.2f}")  # fmt: skip
    axes[0][0].legend()
    slope, r2 = sl.zipf_fit(counts)
    rank = np.arange(1, len(counts) + 1)
    axes[0][1].loglog(rank, sorted(counts, reverse=True), "o", color=PRIMARY, ms=4)
    axes[0][1].loglog(rank, np.exp(np.polyval(
        [slope, np.log(sorted(counts, reverse=True)[0])], np.log(rank))), color=ALERT)  # fmt: skip
    axes[0][1].set(xlabel="language rank", ylabel="videos",
                   title=f"Rank-frequency, slope {slope:.2f} (R² {r2:.2f})")  # fmt: skip
    return _save(fig, study.out, "02_balance.png")


def figure_hours(study: Study) -> str:
    table = [t for t in study.language_table() if t["downloaded"]]
    fig, axes = _axes(1, 2, (10, 4.5))
    videos = np.array([t["downloaded"] for t in table], dtype=float)
    hours = np.array([t["hours"] for t in table], dtype=float)
    ax = axes[0][0]
    ax.loglog(videos / videos.sum(), hours / hours.sum(), "o", color=PRIMARY, ms=4)
    low = min((videos / videos.sum()).min(), (hours / hours.sum()).min())
    ax.plot([low, 1], [low, 1], color=MUTED, ls="--")
    for t, v, h in zip(table[:6], videos[:6], hours[:6], strict=False):
        ax.annotate(t["language"], (v / videos.sum(), h / hours.sum()), fontsize=8)
    ax.set(xlabel="share of downloaded videos", ylabel="share of downloaded hours",
           title="Videos vs hours: off the diagonal = longer or shorter videos")  # fmt: skip
    ax = axes[0][1]
    enough = [t for t in table if t["downloaded"] >= 20]
    enough.sort(key=lambda t: t["median_min"])
    ax.barh([t["language"] for t in enough], [t["median_min"] for t in enough], color=PRIMARY)
    ax.set(xlabel="median duration (min)", title="Median duration per language (≥ 20 videos)")
    ax.tick_params(axis="y", labelsize=7)
    return _save(fig, study.out, "03_hours_vs_videos.png")


def figure_retrieval(study: Study) -> str:
    table = [t for t in study.language_table() if t["downloaded"] + t["unavailable"] >= 10]
    table.sort(key=lambda t: t["availability"])
    fig, axes = _axes(1, 2, (11, max(4.5, 0.2 * len(table) + 1)))
    ax = axes[0][0]
    values = np.array([t["availability"] for t in table])
    error = np.array([values - [t["availability_low"] for t in table],
                      [t["availability_high"] for t in table] - values])  # fmt: skip
    ax.errorbar(values, [t["language"] for t in table], xerr=error, fmt="o", ms=3, color=PRIMARY,
                ecolor=MUTED)  # fmt: skip
    ax.set(xlabel="downloaded / (downloaded + unavailable), 95 % Wilson interval",
           title="Is a video still available? By language")  # fmt: skip
    ax.tick_params(axis="y", labelsize=7)
    ax = axes[0][1]
    counts = Counter(study.state.values())
    ax.bar(sl.STATES, [counts[s] for s in sl.STATES], color=[PRIMARY, ALERT, MUTED, SECONDARY,
                                                              SECONDARY, "#c9c9c9"])  # fmt: skip
    for position, state in enumerate(sl.STATES):
        ax.text(position, counts[state], f"{counts[state]:,}", ha="center", va="bottom", fontsize=8)
    ax.set(title="Retrieval state of the release")
    return _save(fig, study.out, "04_retrieval.png")


def figure_durations(study: Study) -> str:
    seconds = study.seconds[study.seconds > 0]
    minutes = seconds / 60
    fig, axes = _axes(1, 3, (14, 4.2))
    bins = np.logspace(np.log10(minutes.min()), np.log10(minutes.max()), 60)
    ax = axes[0][0]
    ax.hist(minutes, bins=bins, color=PRIMARY, density=True, alpha=0.8)
    grid = np.logspace(np.log10(minutes.min()), np.log10(minutes.max()), 300)
    mu, sigma = study.durations["lognormal_mu"] - math.log(60), study.durations["lognormal_sigma"]
    pdf = np.exp(-((np.log(grid) - mu) ** 2) / (2 * sigma**2)) / (grid * sigma * math.sqrt(2 * math.pi))
    ax.plot(grid, pdf, color=ALERT, label=f"log-normal fit (KS {study.durations['lognormal_ks']:.2f})")
    ax.set(xscale="log", xlabel="duration (min)", ylabel="density", title="Video duration")
    ax.legend()
    ax = axes[0][1]
    ordered = np.sort(minutes)
    ax.plot(ordered, np.arange(1, ordered.size + 1) / ordered.size, color=PRIMARY, label="videos")
    ax.plot(ordered, np.cumsum(ordered) / ordered.sum(), color=SECONDARY, label="hours")
    ax.set(xscale="log", xlabel="duration (min)", ylabel="cumulative share",
           title="Where the hours are: videos vs hours ECDF")  # fmt: skip
    ax.legend()
    ax = axes[0][2]
    top = [lang for lang in study.languages if (study.language == lang).sum() >= 30][:15]
    ax.boxplot([study.seconds[study.language == lang] / 60 for lang in top], tick_labels=top,
               showfliers=False)  # fmt: skip
    ax.set(yscale="log", ylabel="duration (min)", title="Duration, 15 largest languages")
    ax.tick_params(axis="x", rotation=60)
    return _save(fig, study.out, "05_durations.png")


def figure_technical(study: Study) -> str:
    fig, axes = _axes(2, 2, (10, 7))
    labels = [label for _, _, label in sl.HEIGHT_BUCKETS]
    buckets = Counter(sl.height_bucket(int(h)) for h in study.height)
    axes[0][0].bar(labels, [buckets[label] for label in labels], color=PRIMARY)
    axes[0][0].set(title="Frame height (px)")
    axes[0][0].tick_params(axis="x", rotation=30)
    fps = study.fps[np.isfinite(study.fps)]
    rounded = Counter(np.round(fps).astype(int))
    common = sorted(rounded)
    axes[0][1].bar([str(f) for f in common], [rounded[f] for f in common], color=PRIMARY)
    axes[0][1].set(title="Frame rate (fps, rounded)", yscale="log")
    aspects = Counter(sl.aspect_class(int(w), int(h)) for w, h in zip(study.width, study.height, strict=True))
    axes[1][0].bar(list(aspects), list(aspects.values()), color=PRIMARY)
    axes[1][0].set(title="Aspect ratio", yscale="log")
    bpp = sl.bits_per_pixel(study.bit_rate, study.width, study.height, study.fps)
    bpp = bpp[np.isfinite(bpp) & (bpp > 0)]
    axes[1][1].hist(bpp, bins=np.logspace(np.log10(bpp.min()), np.log10(bpp.max()), 60), color=PRIMARY)
    axes[1][1].set(xscale="log", title="Bits per pixel per frame (detail proxy)")
    return _save(fig, study.out, "06_technical.png")


def figure_heatmap(study: Study) -> str:
    top = [lang for lang in study.languages if (study.language == lang).sum() >= 30][:20]
    labels = [label for _, _, label in sl.HEIGHT_BUCKETS]
    grid = np.zeros((len(top), len(labels)))
    for row, lang in enumerate(top):
        heights = study.height[study.language == lang]
        for h in heights:
            grid[row, labels.index(sl.height_bucket(int(h)))] += 1
        grid[row] /= max(grid[row].sum(), 1)
    fig, axes = _axes(size=(6.5, 0.3 * len(top) + 1.5))
    ax = axes[0][0]
    image = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(top)), yticklabels=top,
           title="Resolution mix per language (row share)")  # fmt: skip
    ax.grid(False)
    fig.colorbar(image, ax=ax)
    return _save(fig, study.out, "07_language_resolution.png")


def figure_temperature(study: Study) -> str:
    counts = list(study.release_counts.values())
    grid = np.linspace(0, 1, 21)
    rows = sl.temperature_sampling(counts, grid)
    fig, axes = _axes(1, 2, (10, 4.2))
    axes[0][0].plot(grid, [r["effective_languages"] for r in rows], color=PRIMARY)
    axes[0][0].set(xlabel="sampling temperature t  (p ∝ nᵗ)", ylabel="effective languages",
                   title=f"Balance bought by flattening ({len(counts)} languages)")  # fmt: skip
    top = study.languages[:6]
    width = 0.8 / len(TEMPERATURES)
    for k, t in enumerate(TEMPERATURES):
        p = np.array(counts, dtype=float) ** t
        p /= p.sum()
        axes[0][1].bar(np.arange(len(top)) + k * width, p[: len(top)], width, label=f"t={t:g}")
    axes[0][1].set(xticks=np.arange(len(top)) + 0.4 - width / 2, xticklabels=top,
                   ylabel="sampling probability", title="Mix of the six largest languages")  # fmt: skip
    axes[0][1].legend()
    return _save(fig, study.out, "08_sampling_temperature.png")


def figure_subtitles(study: Study) -> str:
    fig, axes = _axes(1, 2, (9, 4))
    axes[0][0].hist([r["coverage"] for r in study.subtitles if np.isfinite(r["coverage"])],
                    bins=40, color=PRIMARY)  # fmt: skip
    axes[0][0].set(xlabel="share of the video covered by cues", title="Caption coverage")
    axes[0][1].hist([r["words_per_s"] for r in study.subtitles if np.isfinite(r["words_per_s"])],
                    bins=40, color=PRIMARY)  # fmt: skip
    axes[0][1].set(xlabel="words per second of caption", title="Caption density")
    return _save(fig, study.out, "09_subtitles.png")


# ── text ──────────────────────────────────────────────────────────────────────────────────


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _p(value: float) -> str:
    return "<1e-10" if value < 1e-10 else f"{value:.1g}"


def diagnostics(study: Study) -> list[tuple[str, str]]:
    """Findings worth acting on, as ``(severity, sentence)``; thresholds are ours."""
    counts = list(study.release_counts.values())
    total = sum(counts)
    flags: list[tuple[str, str]] = []
    top_language, top_count = next(iter(study.release_counts.items()))
    if top_count / total > 0.3:
        flags.append(("alto", f"`{top_language}` è il {top_count / total:.0%} del rilascio: una "
                      "media sul corpus intero descrive soprattutto quella lingua."))  # fmt: skip
    gini = sl.gini(counts)
    flags.append(("alto" if gini > 0.6 else "info",
                  f"Gini sui video per lingua {gini:.2f}; equivalgono a {sl.effective_classes(counts):.1f} "
                  f"lingue ugualmente frequenti su {len(counts)}."))  # fmt: skip
    small = [lang for lang, n in study.release_counts.items() if n < 100]
    if small:
        flags.append(("medio", f"{len(small)} lingue hanno meno di 100 video nel rilascio: troppo "
                      "poche per addestrare o valutare da sole, utili solo in un mix."))  # fmt: skip
    odd = [lang for lang in study.languages if lang != sl.UNKNOWN and not sl.ISO_639_3.match(lang)]
    if odd:
        flags.append(("medio", f"Codici lingua non ISO 639-3: {', '.join(odd)}. Vanno normalizzati "
                      "prima di raggruppare per lingua."))  # fmt: skip
    unknown = study.release_counts.get(sl.UNKNOWN, 0)
    if unknown:
        flags.append(("medio", f"{unknown} video ({unknown / total:.1%}) hanno lingua `???`."))
    if study.repeated:
        flags.append(("medio", f"{len(study.repeated)} ID compaiono più volte nel CSV (con lingue "
                      "diverse o uguali); qui conta la prima occorrenza."))  # fmt: skip
    bias = study.bias
    if bias.p_value < 0.01 and bias.cramers_v > 0.1:
        flags.append(("alto", f"La disponibilità dipende dalla lingua (chi² p={_p(bias.p_value)}, "
                      f"Cramér V={bias.cramers_v:.2f}): il sottoinsieme scaricato non rappresenta "
                      "il rilascio."))  # fmt: skip
    flags.append(("info", f"Distanza di variazione totale tra mix di lingue del rilascio e dello "
                  f"scaricato: {bias.total_variation:.3f} (KL {bias.kl_nats:.3f} nat)."))  # fmt: skip
    untried = sum(1 for s in study.state.values() if s == "untried")
    if untried:
        flags.append(("medio", f"{untried} ID non sono ancora stati tentati: i numeri sullo "
                      "scaricato cambieranno a download finito."))  # fmt: skip
    estimate = sum(v[0] for v in study.estimated.values())
    if estimate:
        ratio = estimate / sl.CLAIMED_HOURS
        flags.append(("alto" if abs(ratio - 1) > 0.25 else "info",
                      f"Ore stimate sull'intero rilascio (durata media scaricata × conteggio): "
                      f"{estimate:,.0f} h contro le {sl.CLAIMED_HOURS:,.0f} dichiarate "
                      f"({ratio:.0%})."))  # fmt: skip
    top1 = study.durations["hours_share_top1pct"]
    if top1 > 0.1:
        flags.append(("medio", f"L'1 % dei video più lunghi contiene il {top1:.0%} delle ore: "
                      "poche registrazioni lunghe pesano molto."))  # fmt: skip
    if study.by_language["p"] < 0.01:
        flags.append(("info", f"La durata dei video varia con la lingua (Kruskal-Wallis p="
                      f"{_p(study.by_language['p'])}, ε²={study.by_language['epsilon_squared']:.2f})."))  # fmt: skip
    if study.duplicates:
        flags.append(("medio", f"{len(study.duplicates)} gruppi di ID con stessa dimensione e "
                      "durata: probabili ricaricamenti dello stesso video (rischio di leakage "
                      "tra split)."))  # fmt: skip
    broken = sum(1 for s in study.state.values() if s == "corrupt")
    if broken:
        flags.append(("alto", f"{broken} file presenti ma non validi (vedi anomalies.csv)."))
    if study.stray:
        flags.append(("info", f"{len(study.stray)} file in videos/ non corrispondono a ID del CSV."))
    return flags


def write_report(study: Study, figures: list[str]) -> str:
    counts = list(study.release_counts.values())
    total = sum(counts)
    table = study.language_table()
    d = study.durations
    slope, r2 = sl.zipf_fit(counts)
    states = Counter(study.state.values())
    heights = Counter(sl.height_bucket(int(h)) for h in study.height)
    codecs = Counter(r["video_codec"] for r in study.good)
    lines = ["# YouTube-SL-25: analisi del dataset", ""]

    lines += ["## Diagnostiche", ""]
    lines += [f"- **{severity}** · {text}" for severity, text in diagnostics(study)]

    lines += ["", "## 1. Com'è fatto", "",
              f"Il rilascio elenca **{len(study.language_of):,}** video unici in "
              f"**{len(counts)}** lingue dei segni. Ogni riga del CSV ha solo `video_id` e "
              "`language`: nessuna durata, split, canale, firmatario o data. Le didascalie non "
              "sono nel CSV; si scaricano da YouTube insieme al video.", "",
              _table(["lingua", "video", "quota", "scaricati", "ore scaricate", "durata media (min)"],
                     [[t["language"], f"{t['release_videos']:,}", f"{t['release_share']:.1%}",
                       f"{t['downloaded']:,}", f"{t['hours']:.0f}", f"{t['mean_min']:.1f}"]
                      for t in table[:15]]),
              "", f"![composizione]({figures[0]})"]  # fmt: skip

    lines += ["", "## 2. Bilanciamento", "",
              _table(["indice", "valore"],
                     [["Gini (video per lingua)", f"{sl.gini(counts):.3f}"],
                      ["entropia normalizzata", f"{sl.normalised_entropy(counts):.3f}"],
                      ["lingue effettive (inverso di Simpson)", f"{sl.effective_classes(counts):.1f}"],
                      ["quota della prima lingua", f"{counts[0] / total:.1%}"],
                      ["quota delle prime 5", f"{sum(counts[:5]) / total:.1%}"],
                      ["pendenza Zipf (log-log)", f"{slope:.2f} (R² {r2:.2f})"],
                      ["lingue con < 100 video", sum(n < 100 for n in counts)],
                      ["lingue con < 500 video", sum(n < 500 for n in counts)]]),
              "", "Campionando con `p ∝ nᵗ` il mix cambia così:", "",
              _table(["t", "lingue effettive", "quota della prima", "entropia norm."],
                     [[f"{r['temperature']:g}", f"{r['effective_languages']:.1f}",
                       f"{r['top_share']:.1%}", f"{r['entropy_norm']:.2f}"]
                      for r in sl.temperature_sampling(counts, TEMPERATURES)]),
              "", f"![bilanciamento]({figures[1]})", f"![temperatura]({figures[7]})"]  # fmt: skip

    lines += ["", "## 3. Cosa è stato recuperato e con che bias", "",
              _table(["stato", "ID", "quota"],
                     [[s, f"{states[s]:,}", f"{states[s] / total:.1%}"] for s in sl.STATES]),
              "", f"Tra gli ID con esito noto (scaricato o non disponibile), la disponibilità "
              f"dipende dalla lingua: chi² = {study.bias.chi2:.0f}, p = {_p(study.bias.p_value)}, "
              f"Cramér V = {study.bias.cramers_v:.2f}. Distanza di variazione totale del mix "
              f"di lingue: {study.bias.total_variation:.3f}. Le cause di fallimento sono unite su "
              "più run e alcune vecchie possono essere sbagliate (cookie scaduti): gli ID `unavailable` di run vecchi possono essere ritentati con cookie validi, e questo bias va rimisurato a download finito.", "",
              f"![recupero]({figures[3]})"]  # fmt: skip

    lines += ["", "## 4. Durate", "",
              f"{int(d['videos']):,} video validi, **{d['hours']:,.0f} h**. Mediana "
              f"{d['p50_min']:.1f} min, media {d['mean_min']:.1f} min, 5°–95° percentile "
              f"{d['p05_min']:.1f}–{d['p95_min']:.1f} min, 99° {d['p99_min']:.0f} min. "
              f"Il log-normale ha μ={d['lognormal_mu']:.2f}, σ={d['lognormal_sigma']:.2f} sui "
              f"secondi (KS={d['lognormal_ks']:.2f}: valori alti indicano che non è una buona "
              f"descrizione). L'1 % dei video più lunghi vale il {d['hours_share_top1pct']:.0%} "
              f"delle ore, il 10 % il {d['hours_share_top10pct']:.0%}.", "",
              f"![durate]({figures[4]})", f"![ore e video]({figures[2]})"]  # fmt: skip
    estimates = sorted(((lang, v) for lang, v in study.estimated.items()),
                       key=lambda kv: -kv[1][0])[:10]  # fmt: skip
    lines += ["", "Ore stimate sull'intero rilascio per le 10 lingue maggiori "
              "(assume che la disponibilità non dipenda dalla durata):", "",
              _table(["lingua", "ore stimate", "intervallo 95 %"],
                     [[lang, f"{v[0]:,.0f}", f"{v[1]:,.0f}–{v[2]:,.0f}"] for lang, v in estimates])]

    lines += ["", "## 5. Qualità tecnica", "",
              f"Altezza: " + ", ".join(f"{label} {heights[label]:,}" for _, _, label in
                                       sl.HEIGHT_BUCKETS) + ". "
              f"Senza audio: {sum(not r['has_audio'] for r in study.probe):,}. "
              f"Codec video: " + ", ".join(f"{c} {n:,}" for c, n in codecs.most_common(4)) + ". "
              "Il downloader limita a 720p, quindi la distribuzione dell'altezza è troncata "
              "dall'alto.", "", f"![tecnica]({figures[5]})", f"![risoluzione]({figures[6]})"]  # fmt: skip

    lines += ["", "## 6. Anomalie", "",
              f"{len(study.anomalies)} file anomali (`anomalies.csv`), {len(study.duplicates)} "
              "gruppi di probabili duplicati."]
    if study.duplicates:
        lines += ["", "Primi gruppi di duplicati: " + "; ".join(", ".join(g) for g in study.duplicates[:5])]

    lines += ["", "## 7. Didascalie", ""]
    if study.subtitles:
        coverage = np.array([r["coverage"] for r in study.subtitles])
        lines += [f"{len(study.subtitles):,} file di sottotitoli. Copertura mediana del video "
                  f"{np.nanmedian(coverage):.0%}; parole al secondo mediane "
                  f"{np.nanmedian([r['words_per_s'] for r in study.subtitles]):.2f}.", "",
                  f"![didascalie]({figures[8]})"]  # fmt: skip
    else:
        lines += ["Nessun file `.vtt`/`.srt` accanto ai video: lo script di download usato qui "
                  "scarica solo i video. Senza testo non si possono misurare la copertura, la "
                  "lunghezza dei segmenti o la lingua scritta, che sono le diagnostiche più "
                  "utili per capire per quali task il corpus è adatto. Si ottengono con "
                  "`--write-subs` di yt-dlp (o con `YouTubeDownloader(with_subtitles=True)` del "
                  "repository); rilanciando questo report vengono analizzati da soli."]

    lines += ["", "## 8. Per quali task, e cosa non si può dire", "",
              "Dalla struttura dei dati, non dal paper: il corpus offre video con etichetta di "
              "lingua dei segni e, dove le didascalie sono presenti, testo con tempi. Questo "
              "regge (a) identificazione della lingua dei segni, (b) pre-addestramento "
              "multilingue su video, (c) traduzione o retrieval segno→testo se le didascalie sono "
              "allineate. Il CSV non dà split, canale, firmatario né data, per cui: dividere per "
              "ID video evita solo il leakage più ovvio; i ricaricamenti dello stesso contenuto e "
              "i video dello stesso canale possono finire su lati diversi dello split. Per "
              "analizzare l'andamento nel tempo o per canale servono i metadati di YouTube "
              "(`--write-info-json`)."]
    text = "\n".join(lines) + "\n"
    (study.out / "report.md").write_text(text)
    return text


def write_tables(study: Study) -> None:
    table = study.language_table()
    with (study.out / "languages.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (study.out / "anomalies.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "reason"])
        writer.writeheader()
        writer.writerows(study.anomalies)
    summary = {
        "release_ids": len(study.language_of),
        "languages": len(study.release_counts),
        "states": dict(Counter(study.state.values())),
        "gini": sl.gini(list(study.release_counts.values())),
        "durations": study.durations,
        "duration_by_language": study.by_language,
        "retrieval_bias": {"chi2": study.bias.chi2, "p": study.bias.p_value,
                           "cramers_v": study.bias.cramers_v,
                           "total_variation": study.bias.total_variation},  # fmt: skip
        "anomalies": len(study.anomalies),
        "duplicate_groups": len(study.duplicates),
        "subtitle_files": len(study.subtitles),
    }
    (study.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="YouTube-SL-25 directory.")
    parser.add_argument("--out", type=Path, required=True, help="Directory for the report.")
    parser.add_argument("--workers", type=int, default=32, help="ffprobe threads.")
    parser.add_argument("--reuse-probe", action="store_true",
                        help="Trust probe.parquet without looking for new or changed files.")  # fmt: skip
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    study = Study(arguments.root, arguments.out, arguments.workers, arguments.reuse_probe)
    figures = [
        figure_composition(study), figure_balance(study), figure_hours(study),
        figure_retrieval(study), figure_durations(study), figure_technical(study),
        figure_heatmap(study), figure_temperature(study),
    ]  # fmt: skip
    figures.append(figure_subtitles(study) if study.subtitles else "")
    write_tables(study)
    write_report(study, figures)
    print(f"report: {arguments.out / 'report.md'}")


if __name__ == "__main__":
    main()
