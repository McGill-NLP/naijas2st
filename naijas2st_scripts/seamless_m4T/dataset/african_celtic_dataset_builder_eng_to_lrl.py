"""Build SeamlessM4T manifest for English -> LRL speech translation training.

Counterpart of ``NaijaS2ST_builder.py`` in the reverse
direction: English audio is the source and the matched LRL recording
is the target. Same three-pass logic (metadata, audio download, unit
extraction) but the resulting JSONL has English source records with
discrete units extracted from the LRL target audio.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("accented_to_low_resource")

# =========================
# CONFIG
# =========================
DATASET = "McGill-NLP/NaijaS2ST"
BASE_PATH = Path("./audio_dataset/NaijaS2ST_reformatted/")
SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AFRICAN_LANGS = {"H": "hausa", "Y": "yoruba", "I": "igbo"}
AFRICAN_ACCENT_PREF = {"hausa": "north", "yoruba": "south", "igbo": "south"}
AFRICAN_LANG_CODES = {"hausa": "arb", "yoruba": "swh", "igbo": "swh"}

def extract_base_id(text_id):
    """Return the language-agnostic base text ID (everything after the first char).

    Args:
        text_id (str | None): Full text ID like ``"ETE_0123"``.

    Returns:
        str | None: ``"TE_0123"`` for the example above, or ``None``
        if input is empty/too short.
    """
    return text_id[1:] if text_id and len(text_id) > 1 else None

def get_accent(user_id, text_id):
    """Classify an English recording's accent as ``north`` or ``south``.

    Args:
        user_id (str | None): Speaker ID.
        text_id (str): Text ID.

    Returns:
        str: ``"north"``, ``"south"`` or ``"unknown"``.
    """
    if not user_id: return "unknown"
    if user_id.startswith("EN") or (user_id.startswith("H") and text_id.startswith("E")):
        return "north"
    return "south"

def load_audio_tensor(audio_dict, target_sr=SR):
    """Convert a HuggingFace audio dict to a mono ``[1, T]`` tensor at ``target_sr``.

    Args:
        audio_dict (dict): ``{"array": np.ndarray, "sampling_rate": int}``.
        target_sr (int): Desired sample rate in Hz.

    Returns:
        torch.Tensor: Tensor of shape ``[1, T]`` at ``target_sr``.
    """
    wav = torch.tensor(audio_dict["array"], dtype=torch.float32)
    orig_sr = audio_dict["sampling_rate"]
    if len(wav.shape) == 1: wav = wav.unsqueeze(0)
    if orig_sr != target_sr:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_sr, target_sr)
        wav = resampler(wav)
    return wav

def main(split, out_path):
    """Build the English -> LRL SeamlessM4T training manifest for one split.

    Three-pass pipeline (mirror of the LRL -> English variant):

        Pass 1 (metadata):
            Stream the HF dataset once and split records into
            ``english_meta`` (``{base_id: {accent: (uid, tid, text)}}``)
            and ``african_meta`` (``{base_id: {lang: [tgt records]}}``).
            Keep only base IDs that appear on both sides as the
            ``matched_base_ids`` set.

        Pass 2 (audio download):
            Stream the dataset again; every time an example's
            ``(text_id, user_id)`` belongs to a matched pair, write a
            16 kHz mono wav to
            ``<BASE_PATH>/<user_id>/<text_id>.wav`` (skip if already
            present). Each key is removed from the queue after writing
            so the loop can break early when the queue is empty.

        Pass 3 (unit extraction + manifest write):
            Initialise ``UnitSpeechTokenizer`` on ``DEVICE``. For each
            matched base ID and each African target recording,
            extract discrete units from the **African** audio (not
            English) so the model learns to predict LRL units from
            English speech. Pair the units with the English source
            recording for the matched accent
            (``AFRICAN_ACCENT_PREF[lang]``, falling back to any
            available accent). Write JSONL records with
            ``source.units = None`` (English source) and the
            extracted ``target.units`` (LRL target). Free GPU memory
            after every inner-loop iteration to keep VRAM stable on
            long runs.

    Args:
        split (str): HF dataset split to process.
        out_path (str | os.PathLike): Destination JSONL manifest path.

    Returns:
        None. Writes one JSONL record per source/target pair to
        ``out_path`` and downloads/caches matched wavs under
        ``BASE_PATH``.
    """
    logger.info("Pass 1: Collecting IDs for matching (English -> African)...")
    english_meta = defaultdict(dict) # {base_id: {accent: (uid, tid, text)}}
    african_meta = defaultdict(lambda: defaultdict(list)) # {base_id: {lang: [data_list]}}
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

    # Identify pairs where English exists for the requested African accent
    matched_base_ids = [b for b in english_meta if b in african_meta]

    # Pre-calculate which audio files we need to download/verify
    for b in matched_base_ids:
        # Add English (Source) IDs
        for accent_data in english_meta[b].values():
            all_needed_ids.add((accent_data[1], accent_data[0]))
        # Add African (Target) IDs
        for lang in african_meta[b]:
            for tgt in african_meta[b][lang]:
                all_needed_ids.add((tgt["text_id"], tgt["user_id"]))

    # Pass 2: Audio Download (The "Speed Fix" block remains the same)
    logger.info(f"Pass 2: Ensuring {len(all_needed_ids)} audio files exist...")
    ds_stream = load_dataset(DATASET, split=split, streaming=True, trust_remote_code=True)
    for ex in tqdm(ds_stream, desc="Downloading Audio"):
        key = (ex["text_id"], ex["user_id"])
        if key in all_needed_ids:
            audio_path = BASE_PATH / key[1] / f"{key[0]}.wav"
            if not audio_path.exists():
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                wav = load_audio_tensor(ex["audio"], SR)
                torchaudio.save(audio_path, wav, SR)
            all_needed_ids.remove(key)
        if not all_needed_ids: break

    # Pass 3: Unit Extraction (Target is now the African Language)
    logger.info("Pass 3: Extracting units for African languages and writing manifest...")
    extractor = UnitSpeechTokenizer(device=torch.device(DEVICE))
    
    with open(out_path, "w") as fout:
        for base_id in tqdm(matched_base_ids, desc="Writing Manifest"):
            # For every language version of this text...
            for lang, tgt_list in african_meta[base_id].items():
                
                # We need the English source that matches the preferred accent for this lang
                pref = AFRICAN_ACCENT_PREF[lang]
                src_uid, src_tid, src_text = english_meta[base_id].get(pref, list(english_meta[base_id].values())[0])
                src_path = BASE_PATH / src_uid / f"{src_tid}.wav"
                
                if not src_path.exists(): continue

                # Iterate through all available Target recordings in the African language
                for tgt in tgt_list:
                    tgt_path = BASE_PATH / tgt["user_id"] / f"{tgt['text_id']}.wav"
                    if not tgt_path.exists(): continue

                    # EXTRACT UNITS FROM TARGET (African Lang)
                    try:
                        tgt_wav, _ = torchaudio.load(tgt_path)
                        # Ensure mono/correct SR for encoder
                        units = extractor.encode(tgt_wav.to(DEVICE), SR).tolist()
                    except Exception as e:
                        logger.error(f"Unit extraction failed for {tgt['text_id']}: {e}")
                        continue

                    record = {
                        "source": {
                            "id": f"{src_uid}_{src_tid}",
                            "text": src_text,
                            "lang": "eng",
                            "audio_local_path": src_path.as_posix(),
                            "units": None
                        },
                        "target": {
                            "id": f"{tgt['user_id']}_{tgt['text_id']}",
                            "text": tgt["text"],
                            "lang": AFRICAN_LANG_CODES[lang],
                            "audio_local_path": tgt_path.as_posix(),
                            "units": units
                        }
                    }
                    fout.write(json.dumps(record) + "\n")
                    
                    # Cleanup inside the inner loop to be safe with VRAM
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