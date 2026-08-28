"""Build SeamlessM4T manifest for LRL -> English speech translation training.

Streams the McGill-NLP African/Celtic HF dataset twice:

1. First pass collects metadata, pairing each LRL (Hausa/Yoruba/Igbo)
   utterance to its English counterpart using a shared base text ID and
   the preferred Nigerian English accent per language.
2. Second pass downloads only the matched audio files locally.
3. Third pass uses SeamlessM4T's ``UnitSpeechTokenizer`` to extract
   discrete unit codes for the target English audio.

Final output is a JSONL manifest of ``{source, target}`` records ready
for SeamlessM4T fine-tuning.
"""

import argparse
import json
import logging
import gc
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm

from seamless_communication.cli.m4t.finetune.dataset import UnitSpeechTokenizer

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("NaijaS2ST")

# =========================
# CONFIG
# =========================
DATASET = "McGill-NLP/NaijaS2ST"
MODEL = "seamlessM4T_v2_large" # Using medium for better OOM safety on Mila scratch
BASE_PATH = Path("./audio_dataset/NaijaS2ST_reformatted/")
SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AFRICAN_LANGS = {"H": "hausa", "Y": "yoruba", "I": "igbo"}
AFRICAN_ACCENT_PREF = {"hausa": "north", "yoruba": "south", "igbo": "south"}
AFRICAN_LANG_CODES = {"hausa": "hau", "yoruba": "yor", "igbo": "ibo"}

def extract_base_id(text_id):
    """Return the language-agnostic base text ID (everything after the first char).

    Args:
        text_id (str | None): Full text ID like ``"HTE_0123"`` or
            ``"ETE_0123"``.

    Returns:
        str | None: ``"TE_0123"`` for the example above, or ``None``
        if the input is empty/too short.
    """
    return text_id[1:] if text_id and len(text_id) > 1 else None

def get_accent(user_id, text_id):
    """Classify an English recording's accent as ``north`` or ``south``.

    Args:
        user_id (str | None): Speaker ID; ``EN``-prefixed users are
            Northern speakers.
        text_id (str): Text ID; English recordings start with ``E``.

    Returns:
        str: ``"north"``, ``"south"``, or ``"unknown"`` if ``user_id``
        is empty/None.
    """
    if not user_id: return "unknown"
    if user_id.startswith("EN"):
        return "north"
    elif user_id.startswith("H") and text_id.startswith("E"):
        return "north"
    else:
        return "south"

def load_audio_tensor(audio_dict, target_sr=SR):
    """Convert a HuggingFace audio dict to a mono ``[1, T]`` float tensor.

    Args:
        audio_dict (dict): ``{"array": np.ndarray, "sampling_rate": int}``.
        target_sr (int): Desired output sample rate in Hz.

    Returns:
        torch.Tensor: Tensor of shape ``[1, T]`` at ``target_sr``.
    """
    wav = torch.tensor(audio_dict["array"], dtype=torch.float32)
    orig_sr = audio_dict["sampling_rate"]
    
    if len(wav.shape) == 1: 
        wav = wav.unsqueeze(0)
    
    if orig_sr != target_sr:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_sr, target_sr)
        wav = resampler(wav)
        
    return wav

def main(split, out_path):
    """Build the LRL -> English SeamlessM4T training manifest for one split.

    Three-pass pipeline:

        Pass 1 (metadata):
            Stream the HF dataset once and build ``english_meta``
            (``{base_id: {accent: (uid, tid, text)}}``) plus
            ``african_meta`` (``{base_id: {lang: [src records]}}``).
            ``base_id`` strips the leading language letter so English
            and LRL utterances of the same content share a key.

        Filter pairing:
            Restrict to ``base_id``s that exist on both sides and
            collect every audio key (``(text_id, user_id)``) needed
            for the second pass.

        Pass 2 (audio download):
            Stream the dataset again and, every time an example's key
            is in the needed set, write a 16 kHz mono wav to
            ``<BASE_PATH>/<user_id>/<text_id>.wav`` (skip if already on
            disk). Resampling is done on the fly via
            :func:`load_audio_tensor`. Each key is removed from the
            needed set after handling so the loop can short-circuit
            when nothing is left.

        Pass 3 (unit extraction + manifest write):
            Instantiate a ``UnitSpeechTokenizer`` on ``DEVICE``. For
            every matched base ID, pick the target English wav by
            preferred accent (``AFRICAN_ACCENT_PREF[lang]``, falling
            back to any available accent), extract discrete units with
            ``extractor.encode``, and write one JSONL record per LRL
            source with ``source.units = None`` and the extracted
            ``target.units``. Free GPU memory after each language
            block. Errors during unit extraction are logged and
            skipped.

    Args:
        split (str): HF dataset split to process (e.g. ``"train"``,
            ``"test"``).
        out_path (str | os.PathLike): Destination JSONL manifest path.

    Returns:
        None. Writes one JSONL record per source/target pair to
        ``out_path`` and downloads/caches matched wavs under
        ``BASE_PATH``.
    """
    # Pass 1: Collect Metadata
    logger.info("Pass 1: Collecting IDs for matching...")
    english_meta = defaultdict(dict) # {base_id: {accent: (user_id, text_id, text)}}
    african_meta = defaultdict(lambda: defaultdict(list)) # {base_id: {lang: [src_data]}}
    all_needed_ids = set()

    ds = load_dataset(DATASET, split=split, streaming=True, trust_remote_code=True)
    for ex in tqdm(ds, desc="Scanning Metadata"):
        tid = ex["text_id"]
        base = extract_base_id(tid)
        uid = ex["user_id"]
        
        if tid.startswith("E"):
            accent = get_accent(uid, tid)
            english_meta[base][accent] = (uid, tid, ex["text"])
        else:
            lang = AFRICAN_LANGS.get(tid[0])
            if lang:
                african_meta[base][lang].append({"text_id": tid, "user_id": uid, "text": ex["text"]})

    # Filter only matched pairs
    matched_base_ids = [b for b in african_meta if b in english_meta]
    for b in matched_base_ids:
        # Add target ID
        for lang in african_meta[b]:
            pref = AFRICAN_ACCENT_PREF[lang]
            tgt_uid, tgt_tid, _ = english_meta[b].get(pref, list(english_meta[b].values())[0])
            all_needed_ids.add((tgt_tid, tgt_uid))
            # Add source IDs
            for src in african_meta[b][lang]:
                all_needed_ids.add((src["text_id"], src["user_id"]))

    # Pass 2: Download Audio Once (The Speed Fix)
    logger.info(f"Pass 2: Downloading {len(all_needed_ids)} audio files in a single stream...")
    ds_stream = load_dataset(DATASET, split=split, streaming=True, trust_remote_code=True)
    for ex in tqdm(ds_stream, desc="Downloading Audio"):
        key = (ex["text_id"], ex["user_id"])
        if key in all_needed_ids:
            audio_path = BASE_PATH / key[1] / f"{key[0]}.wav"
            if not audio_path.exists():
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                wav = load_audio_tensor(ex["audio"], SR)
                torchaudio.save(audio_path, wav, SR)
            all_needed_ids.remove(key) # Optimization: stop checking once found
        if not all_needed_ids: break

    # Pass 3: Unit Extraction (The OOM-Safe Part)
    logger.info("Pass 3: Extracting units and writing manifest...")
    extractor = UnitSpeechTokenizer(device=torch.device(DEVICE))
    
    with open(out_path, "w") as fout:
        for base_id in tqdm(matched_base_ids, desc="Writing Manifest"):
            for lang, src_list in african_meta[base_id].items():
                # 1. Get Target Info
                pref = AFRICAN_ACCENT_PREF[lang]
                tgt_uid, tgt_tid, tgt_text = english_meta[base_id].get(pref, list(english_meta[base_id].values())[0])
                tgt_path = BASE_PATH / tgt_uid / f"{tgt_tid}.wav"
                print(f"Processing {tgt_path} for target {tgt_uid}_{tgt_tid} with text: {tgt_text}")
                if not tgt_path.exists(): continue

                # Extract units from target English
                try:
                    tgt_wav, _ = torchaudio.load(tgt_path)
                    units = extractor.encode(tgt_wav.to(DEVICE), SR).tolist()
                except Exception as e:
                    logger.error(f"Unit extraction failed for {tgt_tid}: {e}")
                    continue

                # 2. Write pairs
                for src in src_list:
                    src_path = BASE_PATH / src["user_id"] / f"{src['text_id']}.wav"
                    if not src_path.exists(): continue

                    record = {
                        "source": {
                            "id": f"{src['user_id']}_{src['text_id']}",
                            "text": src["text"],
                            "lang": AFRICAN_LANG_CODES[lang],
                            "audio_local_path": src_path.as_posix(),
                            "units": None
                        },
                        "target": {
                            "id": f"{tgt_uid}_{tgt_tid}",
                            "text": tgt_text,
                            "lang": "eng",
                            "audio_local_path": tgt_path.as_posix(),
                            "units": units
                        }
                    }
                    fout.write(json.dumps(record) + "\n")
                
                # Memory Cleanup after each language block
                del units
                gc.collect()
                torch.cuda.empty_cache()

    logger.info(f"Complete: Manifest saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    main(args.split, args.out)