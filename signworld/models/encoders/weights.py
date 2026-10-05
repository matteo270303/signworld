"""The released V-JEPA 2.1 files a run needs, checked before the model is built.

Only the weights are ever downloaded. The code of Meta's repository (``hub_repo``) is not:
when it is missing the error names the folder, and the user puts it there.
"""

import logging
from pathlib import Path
from typing import Final

from signworld.data.acquisition.config import HttpConfig
from signworld.data.acquisition.provenance import verify_sha256
from signworld.data.acquisition.transfer.http import HttpDownloader

logger = logging.getLogger(__name__)

# The released distilled ViT-L/16 at 384 px (the file is kept as ``vjepa2_1_vitl_384.pt``).
VJEPA2_1_VITL_384_URL: Final = (
    "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
)


def ensure_encoder_files(
    hub_repo: Path, checkpoint: Path, *, url: str = VJEPA2_1_VITL_384_URL, sha256: str | None = None
) -> None:
    """Fail when the hub code is absent; download the weights when they are.

    Raises:
        FileNotFoundError: ``hub_repo`` does not exist.
        DownloadError: the weights could not be fetched.
        ChecksumMismatchError: the file is not the pinned one.
    """
    if not (hub_repo / "hubconf.py").is_file():
        raise FileNotFoundError(
            f"{hub_repo}: V-JEPA 2 code not found; it is not downloaded automatically. "
            "Clone facebookresearch/vjepa2 there by hand."
        )
    if not checkpoint.is_file():
        logger.info("%s is missing; downloading %s", checkpoint, url)
        HttpDownloader(HttpConfig()).fetch(url, checkpoint)
    if sha256 is not None:
        verify_sha256(checkpoint, sha256)
