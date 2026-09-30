"""Where the results of the collaudo and of the preliminary controls live: ``collaudo/``.

Every test has its own folder at the repository root, named here, with a ``README.md`` that
says why the test exists, what it checks and how; its results sit next to it, one file per
dataset (``<dataset>.json``) or per model. A new test adds its folder here and its README
there, or ``tests/test_collaudo.py`` fails.
"""

from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2] / "collaudo"

TESTS: Final[dict[str, str]] = {
    "sonda-youtube": "YouTube refusals: account or network (acquisition)",
    "durate-didascalie": "caption durations and frame spacing T/32 (§4.13.1)",
    "duplicati-fra-split": "same caption and duration across splits (§4.13.1)",
    "contaminazione-benchmark": "corpus clips overlapping benchmark evaluation clips (§3.9, P2)",
    "prompt-embeddinggemma": "A2: pinned prompt and re-encoded captions (§4.13.1)",
    "pc1-geometria-testo": "PC1: caption target geometry per MRL dimension (§4.12.1)",
    "metriche-sintetiche": "every metric on synthetic cases with a known answer (§4.13.1)",
    "contenuto-checkpoint": "PC5/PC6: sections, shapes and norms of the checkpoints",
    "selezione-frame": "A3: the frame selection reproduces the stored indices",
    "allineamento-video-posa": "A3: poses re-estimated on the decoded frames",
    "qualita-posa": "A4: shoulders, articulator boxes, normalised keypoints",
    "pc5-encoder-posa": "PC5: which pose latent is the physical-level target (§4.4.3)",
    "spettro-posa": "S-JEPA spectrum and fixed whitening (§4.4.3)",
    "isotropizzazione-posa": "whitening, RBIG, SINF and a flow on the S-JEPA latent (§4.4.3)",
    "letture-video": "frozen video features read out: hands and text, per run",
    "pc2-baseline": "PC2: the retrieval floor a trained model must beat",
    "pc3-encoder-video": "PC3: V-JEPA 2.1-L against V-JEPA 2-L",
    "pc4-risoluzione": "PC4: 256 or 384 pixels",
    "frame-selezionati-vs-contigui": "motion-selected against contiguous frames (§4.13.1)",
    "riproduzione-pesi-video": "A1: video encoder weights and input path (§4.13.1)",
    "riproduzione-pesi-posa": "A1: S-JEPA reloads and reproduces its latents (§4.13.1)",
    "pc6-predictor": "PC6: V-JEPA 2.1 predictor and multi-level input (§4.12.1)",
}


def result_path(test: str, name: str, root: Path = ROOT) -> Path:
    """``collaudo/<test>/<name>``; the test must be one of ``TESTS``."""
    if test not in TESTS:
        raise KeyError(f"{test!r} is not a collaudo test; add it to signworld.collaudo.TESTS")
    return root / test / name
