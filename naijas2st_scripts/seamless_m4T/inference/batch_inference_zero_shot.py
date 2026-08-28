# -*- coding: utf-8 -*-
"""Zero-shot SeamlessM4Tv2 speech-to-speech inference (LRL audio -> English audio).

Walks ``<INPUT_ROOT>/<language>/<user>/recordings/*.wav`` for Yoruba,
Igbo and Hausa, batches the audio, and generates English speech using
``facebook/seamless-m4t-v2-large``. Preserves the per-language
directory structure under ``OUTPUT_ROOT``.
"""

import os
import torch
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model

INPUT_ROOT = "./naijas2st_scripts/test"
OUTPUT_ROOT = "RESULTS/naijas2st/sts_seamless_v1_zero_shot_lrl_to_eng"
BATCH_SIZE = 4
TARGET_SR = 16000
TGT_LANG = "eng"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ---------------- LOAD MODEL ----------------
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-large")
model = SeamlessM4Tv2Model.from_pretrained(
    "facebook/seamless-m4t-v2-large",
    torch_dtype=torch.float32,
    device_map="auto"
)
model.eval()

def load_audio(path):
    """Load a wav file as mono and resample to ``TARGET_SR``.

    Args:
        path (str | os.PathLike): Path to the wav file.

    Returns:
        torch.Tensor: 1-D tensor of audio samples at ``TARGET_SR``.
    """
    audio, sr = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(audio, sr, TARGET_SR)
    return audio.squeeze(0)


def collect_audio_files(root):
    """List all African-language wav files under ``root``.

    Args:
        root (str | os.PathLike): Directory containing ``yoruba/``,
            ``igbo/`` and ``hausa/`` subtrees with
            ``<user>/recordings/*.wav``.

    Returns:
        list[dict]: One dict per discovered wav with keys
        ``path``/``lang``/``user``/``filename``.
    """
    file_list = []

    for lang in ["yoruba", "igbo", "hausa"]:
        lang_dir = os.path.join(root, lang)
        if not os.path.exists(lang_dir):
            continue

        for user in os.listdir(lang_dir):
            user_dir = os.path.join(lang_dir, user)

            recordings_dir = os.path.join(user_dir, "recordings")
            if not os.path.exists(recordings_dir):
                continue

            for f in os.listdir(recordings_dir):
                if f.endswith(".wav"):
                    full_path = os.path.join(recordings_dir, f)

                    file_list.append({
                        "path": full_path,
                        "lang": lang,
                        "user": user,
                        "filename": f
                    })

    return file_list


def process_batch(batch):
    """Generate English speech for a batch of LRL audio files.

    Args:
        batch (list[dict]): Subset of the ``collect_audio_files``
            records (each with ``path``/``lang``/``user``/``filename``).

    Returns:
        None. Writes one English wav per input under
        ``OUTPUT_ROOT/<lang>/<user>_<filename>``.
    """
    audios = []
    meta = []

    for item in batch:
        try:
            audio = load_audio(item["path"])
            audios.append(audio)
            meta.append(item)
        except Exception as e:
            print(f"Skipping {item['path']} | {e}")

    if len(audios) == 0:
        return

    inputs = processor(
        audio=audios,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            tgt_lang=TGT_LANG,
            generate_speech=True
        )

    for i, out_audio in enumerate(outputs):
        item = meta[i]

        # preserve directory structure
        out_dir = os.path.join(
            OUTPUT_ROOT,
            item["lang"],

        )
        user = item["user"]
        filename = item["filename"]
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{user}_{filename}")

        # Ensure out_audio is 2D (channels × samples)
        if out_audio.dim() == 1:
            out_audio = out_audio.unsqueeze(0)
        elif out_audio.dim() == 3:
            out_audio = out_audio.squeeze(0)

        torchaudio.save(out_path, out_audio.cpu(), TARGET_SR)

        print(f"[{item['lang']}] Saved: {out_path}")


def main():
    """Run zero-shot LRL audio -> English audio batches with SeamlessM4Tv2.

    Workflow:
        1. Call :func:`collect_audio_files` to discover every
           ``<INPUT_ROOT>/<language>/<user>/recordings/*.wav`` for
           Yoruba/Igbo/Hausa and print the total count.
        2. Iterate over the file list in chunks of ``BATCH_SIZE``,
           passing each chunk to :func:`process_batch`. The helper
           loads and resamples audio, processes the whole batch with
           ``SeamlessM4TProcessor`` (padding to the longest sample),
           runs ``model.generate(generate_speech=True, tgt_lang="eng")``
           and writes one translated wav per input to
           ``<OUTPUT_ROOT>/<language>/<user>_<filename>``.
        3. Print progress after every batch.

    Inputs:
        ``<INPUT_ROOT>/<language>/<user>/recordings/*.wav``.

    Outputs:
        Per-input English wav under
        ``<OUTPUT_ROOT>/<language>/<user>_<filename>``.

    Returns:
        None.
    """
    files = collect_audio_files(INPUT_ROOT)
    print(f"Found {len(files)} audio files")

    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i + BATCH_SIZE]
        process_batch(batch)

        print(f"Processed {i + len(batch)} / {len(files)}")


if __name__ == "__main__":
    main()