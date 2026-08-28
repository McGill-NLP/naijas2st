"""Driver that delegates SeamlessM4T unit extraction to a subprocess worker.

Streams the HF dataset, pairs LRL utterances with their English
counterparts by shared base text ID, and shells out to
``extract_units_worker.py`` for each audio to get discrete units. The
output is a JSONL of speech-to-speech (LRL units, English units) pairs.
"""

import json
import subprocess
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm

DATASET = "McGill-NLP/NaijaS2ST"
SPLIT = "train"
OUT_PATH = "./sts_ready/train.jsonl"

def is_english(text_id):
    """Return ``True`` if the text ID belongs to an English recording.

    Args:
        text_id (str | None): Text ID to test.

    Returns:
        bool: ``True`` when ``text_id`` is non-empty and starts with ``"E"``.
    """
    return text_id and text_id.startswith("E")

def is_african(text_id):
    """Return ``True`` if the text ID belongs to a non-English (LRL) recording.

    Args:
        text_id (str | None): Text ID to test.

    Returns:
        bool: ``True`` when ``text_id`` is non-empty and does not start
        with ``"E"``.
    """
    return text_id and not text_id.startswith("E")

def extract_base_id(text_id):
    """Drop the leading language-letter from a text ID.

    Args:
        text_id (str): Text ID (e.g. ``"YTE_0123"``).

    Returns:
        str: Stripped ID (e.g. ``"TE_0123"``).
    """
    return text_id[1:]

def get_accent(user_id):
    """Classify an English speaker as ``north``/``south``/``unknown``.

    Args:
        user_id (str | None): Speaker user ID.

    Returns:
        str: One of ``"north"``, ``"south"`` or ``"unknown"``.
    """
    if user_id is None:
        return "unknown"
    if user_id.startswith("EN"):
        return "north"
    if user_id.startswith("EY"):
        return "south"
    return "unknown"

def run_worker(audio):
    """Invoke the unit-extractor worker subprocess on one audio sample.

    Args:
        audio (dict): HuggingFace audio dict
            (``{"array": ..., "sampling_rate": ...}``) serialisable to
            JSON.

    Returns:
        list[int]: Discrete unit IDs returned by the worker.

    Raises:
        RuntimeError: If the subprocess exits non-zero.
    """
    p = subprocess.Popen(
        ["python", "extract_unit_workers.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = p.communicate(json.dumps({"audio": audio}))
    if p.returncode != 0:
        raise RuntimeError(err)
    return json.loads(out)["units"]


def main():
    """Build a JSONL of LRL/English speech-to-speech unit-pair training data.

    Workflow:
        1. Stream ``DATASET`` once into two dicts:
           ``english[base] = [examples]`` and
           ``african[base] = [examples]``, keying on
           :func:`extract_base_id` so an LRL utterance and its English
           counterpart share a key.
        2. For each base ID present in both sides:
            - Pick up to two English recordings to mark
              ``accent_pair`` (``"two_accents"`` when both exist,
              ``"single_accent"`` otherwise).
            - For every LRL recording with that base, call
              :func:`run_worker` twice (once for the LRL audio,
              once for the English audio) to extract SeamlessM4T units
              via a subprocess.
            - Write a JSONL record with ``src_units`` (LRL),
              ``tgt_units`` (English), ``src_lang``, ``tgt_lang``
              (``"eng"``), the base ID and the accent pairing.
            - Print a progress marker every 100 records.
        3. Print a final count of records written.

    Inputs:
        ``DATASET`` HF identifier, ``SPLIT`` to stream,
        ``extract_units_worker.py`` available on ``PATH`` /
        ``PYTHONPATH``.

    Outputs:
        One JSONL ``OUT_PATH`` of STS pairs ready for SeamlessM4T
        speech-to-speech training.

    Returns:
        None.
    """
    print(f"Streaming [{SPLIT}]...")
    ds = load_dataset(DATASET, split=SPLIT, streaming=True)

    english = defaultdict(list)
    african = defaultdict(list)

    # -------- Pass 1: index ----------
    for ex in tqdm(ds):
        tid = ex["text_id"]
        base = extract_base_id(tid)
        if is_english(tid):
            english[base].append(ex)
        elif is_african(tid):
            african[base].append(ex)

    # -------- Pass 2: build STS ----------
    written = 0
    with open(OUT_PATH, "w") as f:
        for base, af_list in african.items():
            if base not in english:
                continue

            en_list = english[base]

            # Accent-aware English pairing
            if len(en_list) >= 2:
                en_a, en_b = en_list[:2]
                accent_pair = "two_accents"
            else:
                en_a = en_b = en_list[0]
                accent_pair = "single_accent"

            for af in af_list:
                src_units = run_worker(af["audio"])
                tgt_units = run_worker(en_a["audio"])

                record = {
                    "id": base,
                    "src_lang": af["language"],
                    "tgt_lang": "eng",
                    "src_units": src_units,
                    "tgt_units": tgt_units,
                    "accent_pair": accent_pair
                }

                f.write(json.dumps(record) + "\n")
                written += 1

                if written % 100 == 0:
                    print(f"Wrote {written} STS pairs")

    print(f" Done. Wrote {written} STS pairs → {OUT_PATH}")


if __name__ == "__main__":
    main()
