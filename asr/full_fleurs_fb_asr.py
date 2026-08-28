"""Run Omnilingual ASR (LLM-7B) over all FLEURS languages.

Uses ``omnlingual_to_fleurs.csv`` to map FLEURS language codes to
Omnilingual ASR codes, skips a hard-coded exclude list plus any
languages already processed, and writes per-language transcription
JSONs to the results directory.
"""

import json
import os
import csv
import tempfile
import soundfile as sf
from itertools import islice
from datasets import load_dataset, Audio
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from huggingface_hub import login


def main():
    """Run Omnilingual ASR LLM-7B over the full set of FLEURS languages.

    Workflow:
        1. Instantiate ``ASRInferencePipeline`` with ``omniASR_LLM_7B``.
        2. Authenticate with HuggingFace using ``HF_TOKEN`` if set.
        3. Read ``omnlingual_to_fleurs.csv`` (mapping FLEURS language
           codes to Omnilingual ASR language codes) into ``fleurs_to_omni``.
        4. Determine which languages to skip: the hard-coded
           ``exclude_languages`` set plus any code whose results JSON
           already exists in ``results_dir`` (so the script is resumable).
        5. For each remaining FLEURS code:
            - Stream the FLEURS test split (catching dataset-loading
              failures and continuing to the next language).
            - Cast ``audio`` to 16 kHz; optionally cap to ``max_examples``.
            - For each sample, write a temp WAV, transcribe it with
              Omnilingual ASR, and append ``{"ID", "transcription"}``.
            - On per-sample exceptions, log and continue.
        6. Write ``<results_dir>/<fleurs_code>_transcriptions.json``.

    Inputs:
        ``HF_TOKEN`` env var; ``omnlingual_to_fleurs.csv`` mapping file;
        FLEURS test split for every covered language.

    Outputs:
        One ``<results_dir>/<fleurs_code>_transcriptions.json`` per
        processed language. The directory is the source of truth for
        resumption: pre-existing files signal "already done".

    Returns:
        None.
    """
    pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")

    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        print("Warning: No HF_TOKEN found. You may hit rate limits.")


    exclude_languages = {
        "ga_ie",
        "cy_gb",
        "sw_ke",
        "yo_ng",
        "ha_ng",
        "ig_ng",
        "lg_ug",
    }


    mapping_csv_path = "omnlingual_to_fleurs.csv"

    fleurs_to_omni = {}
    with open(mapping_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fleurs_to_omni[row["fleurs_code"]] = row["omnilingual_asr_code"]


    max_examples = None
    results_dir = "./RESULTS/asr_LLM-7B"
    os.makedirs(results_dir, exist_ok=True)


    # Get list of already processed languages
    already_processed = set()
    for filename in os.listdir(results_dir):
        if filename.endswith("_transcriptions.json"):
            # Extract language code from filename
            lang_code = filename.replace("_transcriptions.json", "")
            already_processed.add(lang_code)

    print(f"Found {len(already_processed)} already processed languages: {already_processed}")


    for fleurs_code, omnilingual_code in fleurs_to_omni.items():

        # Skip excluded languages
        if fleurs_code in exclude_languages:
            print(f"Skipping {fleurs_code} (excluded).")
            continue

        # Skip already processed languages
        if fleurs_code in already_processed:
            print(f"Skipping {fleurs_code} (already processed).")
            continue

        print(f"\n→ Processing {fleurs_code} ({omnilingual_code})…")

        # Load streaming test split
        try:
            ds_test = load_dataset(
                "google/fleurs",
                fleurs_code,
                split="test",
                streaming=True
            )
        except Exception as e:
            print(f"  Could not load dataset for {fleurs_code}: {e}")
            continue

        ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16_000))
        tmp_dir = tempfile.mkdtemp(prefix=f"fleurs_{fleurs_code}_")

        test_iter = ds_test if max_examples is None else islice(ds_test, max_examples)

        language_results = []


        for sample in test_iter:
            sample_id = sample["id"]
            try:
                arr = sample["audio"]["array"]
                sr = sample["audio"]["sampling_rate"]

                # create wav temporary file
                tmp_path = os.path.join(tmp_dir, f"{sample_id}.wav")
                sf.write(tmp_path, arr, sr)

                # ASR transcription
                transcription = pipeline.transcribe(
                    [tmp_path],
                    lang=[omnilingual_code]
                )

                language_results.append({
                    "ID": sample_id,
                    "transcription": transcription
                })

                print(f"  Processed sample {sample_id}")

            except Exception as e:
                print(f"  Error processing sample {sample_id}: {e}")


        out_path = os.path.join(results_dir, f"{fleurs_code}_transcriptions.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(language_results, f, ensure_ascii=False, indent=2)

        print(f"Wrote {len(language_results)} transcriptions to {out_path}")


if __name__ == "__main__":
    main()
