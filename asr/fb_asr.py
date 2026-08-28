"""Run Meta's Omnilingual ASR (LLM-7B) on FLEURS for a fixed set of languages.

Streams FLEURS test audio for each configured language, writes each
clip to a temporary WAV file, transcribes it with the Omnilingual ASR
pipeline, and saves the per-language transcriptions to JSON.
"""

import json
import os
import tempfile
import soundfile as sf
from itertools import islice
from datasets import load_dataset, Audio
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from huggingface_hub import login


def main():
    """Transcribe FLEURS test audio for each configured language with omniASR.

    Workflow:
        1. Instantiate ``ASRInferencePipeline`` with the ``omniASR_LLM_7B``
           model card (loaded once, reused for every language).
        2. Authenticate with HuggingFace via ``HF_TOKEN`` if set so that
           streaming ``google/fleurs`` does not hit anonymous rate limits.
        3. For each ``(fleurs_code, omnilingual_code)`` pair:
            - Stream the FLEURS test split and cast its ``audio`` column
              to 16 kHz.
            - Create a per-language temp dir for intermediate wavs.
            - Optionally limit to ``max_examples`` items via
              ``itertools.islice``.
            - For every sample, write the resampled audio to a temp WAV,
              call ``pipeline.transcribe([tmp_path], lang=[omnilingual_code])``,
              and append ``{"ID", "transcription"}`` to the language results.
            - On per-sample exceptions, log and continue.
        4. Write ``<results_dir>/<fleurs_code>_transcriptions.json``
           for the full language batch.

    Inputs:
        ``HF_TOKEN`` env var (optional); FLEURS test split for each
        configured language code.

    Outputs:
        ``<results_dir>/<fleurs_code>_transcriptions.json`` per language.

    Returns:
        None.
    """
    pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")

    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        print("Warning: No HF_TOKEN found. You may hit rate limits.")



    language_codes = {
        # "nl_nl":    "nld_Latn", # Dutch, missing
        # "cy_gb":     "cym_Latn",
    #     "sw_ke":   "swh_Latn",
        "yo_ng":    "yor_Latn",
        "ha_ng":     "hau_Latn",
        "ig_ng":      "ibo_Latn",
        "lg_ug":   "lug_Latn",
    }


    max_examples = None
    results_dir = "./RESULTS/asr_LLM-7B"
    os.makedirs(results_dir, exist_ok=True)

    for fleurs_code, omnilingual_code in language_codes.items():
        print(f"\n→ Processing {fleurs_code} ({omnilingual_code})…")

        # Load streaming test dataset
        ds_test = load_dataset("google/fleurs", fleurs_code, trust_remote_code=True, split="test", streaming=True)
        ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16_000))

        tmp_dir = tempfile.mkdtemp(prefix=f"fleurs_{fleurs_code}_")

        test_iter = ds_test if max_examples is None else islice(ds_test, max_examples)

        language_results = []
        for sample in test_iter:
            try:
                arr, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
                file_name = f"{sample['id']}.wav"
                tmp_path = os.path.join(tmp_dir, f"test_{file_name}")
                sf.write(tmp_path, arr, sr)

                sample_id = sample["id"]

                transcription = pipeline.transcribe([tmp_path], lang=[omnilingual_code])
                language_results.append({"ID": sample_id, "transcription": transcription})
                print(f"  Processed sample {sample_id}")
            except Exception as e:
                print(f"  Error processing sample {sample_id}: {e}")

        out_path = os.path.join(results_dir, f"{fleurs_code}_transcriptions.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(language_results, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(language_results)} transcriptions to {out_path}")


if __name__ == "__main__":
    main()
