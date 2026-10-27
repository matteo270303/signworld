"""Indice ridotto per i job di debug: poche clip per split, estratte con un seme fisso.

Uso (dalla radice del repo): python slurm/make_debug_index.py
Le clip restano quelle dell'indice vero, quindi i canali restano separati fra gli split.
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SOURCE = Path("data/training/youtube_sl25/index.parquet")
TARGET = SOURCE.with_name("index.debug.parquet")
TRAIN_CLIPS = 2048  # 16 passi da 128 per epoca
HELD_OUT_CLIPS = 256  # per ogni altro split


def main() -> None:
    if not SOURCE.is_file():
        sys.exit(f"manca {SOURCE}: serve lo stage 0")
    table = pq.read_table(SOURCE)
    rng = np.random.default_rng(0)
    parts = []
    for split in sorted(set(table.column("split").to_pylist())):
        rows = table.filter(pc.equal(table.column("split"), split))
        size = min(rows.num_rows, TRAIN_CLIPS if split == "train" else HELD_OUT_CLIPS)
        parts.append(rows.take(np.sort(rng.choice(rows.num_rows, size, replace=False))))
        print(f"{split}: {size} di {rows.num_rows}")
    pq.write_table(pa.concat_tables(parts), TARGET)
    print(f"scritto {TARGET}")


if __name__ == "__main__":
    main()
