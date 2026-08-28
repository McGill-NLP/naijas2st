"""Sample N unique wav recordings for a human-evaluation survey.

Groups all wavs under ``SOURCE_DIR`` by filename, picks one
representative per filename, then randomly samples ``N_SAMPLES``
unique files (seeded) and copies them into ``TARGET_DIR`` with names
prefixed by their grandparent directory for traceability.
"""

import random
import shutil
from pathlib import Path
from collections import defaultdict

# --- CONFIG ---
SOURCE_DIR = Path("SOURCE_DIR")
TARGET_DIR = Path("TARGET_DIR")
N_SAMPLES = 100
RANDOM_SEED = 42  # set to None for non-reproducible


def main():
    """Sample ``N_SAMPLES`` unique wavs and copy them into ``TARGET_DIR``.

    Workflow:
        1. Seed Python's RNG with ``RANDOM_SEED`` (if set) for
           reproducibility, and create ``TARGET_DIR``.
        2. Walk ``SOURCE_DIR/**/*.wav`` and group every wav by basename
           into ``groups[name] -> [paths]`` so duplicate-named
           recordings collapse to a single bucket per filename.
        3. For each bucket, ``random.choice`` one representative wav
           and collect the picks into ``unique_files``.
        4. Sanity-check that there are at least ``N_SAMPLES`` unique
           filenames; raise ``ValueError`` otherwise.
        5. ``random.sample`` ``N_SAMPLES`` unique entries and
           ``shutil.copy2`` each one into ``TARGET_DIR`` with its
           grand-parent directory name prefixed onto the filename so
           the speaker context is preserved.

    Inputs:
        ``SOURCE_DIR`` (recursive wav tree), ``RANDOM_SEED``.

    Outputs:
        ``N_SAMPLES`` wav files in ``TARGET_DIR``.

    Returns:
        None.

    Raises:
        ValueError: If fewer than ``N_SAMPLES`` unique filenames exist
            under ``SOURCE_DIR``.
    """
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # --- COLLECT + GROUP BY FILENAME ---
    groups = defaultdict(list)

    for f in SOURCE_DIR.glob("**/*.wav"):
        if f.is_file():
            groups[f.name].append(f)  # group by filename only

    # --- PICK ONE FILE PER NAME ---
    unique_files = []
    for name, files in groups.items():
        chosen = random.choice(files)  # pick one instance of that filename
        unique_files.append(chosen)

    print(f"Found {len(unique_files)} unique filenames")

    if len(unique_files) < N_SAMPLES:
        raise ValueError(f"Only {len(unique_files)} unique filenames available, need {N_SAMPLES}")

    # --- SAMPLE 100 UNIQUE FILENAMES ---
    selected_files = random.sample(unique_files, N_SAMPLES)

    # --- COPY ---
    for f in selected_files:
        dest = TARGET_DIR / f"{f.parents[1].name}_{f.name}"
        shutil.copy2(f, dest)

    print(f"Copied {N_SAMPLES} unique .wav files to {TARGET_DIR}")


if __name__ == "__main__":
    main()
