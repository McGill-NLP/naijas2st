"""Run Meta's Omnilingual ASR (LLM-1B) on the local African/Celtic test sets.

Iterates over the per-language directories under
``naijas2st_scripts/test/<language>``, transcribes each audio file
with the Omnilingual ASR pipeline, and writes the transcriptions per
language as JSON. Includes debug prints and per-language pipeline
reinitialisation so failures in one language do not abort the rest.
"""

import json
import os
import tempfile
import soundfile as sf
from pathlib import Path
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import traceback
import sys

language_codes = {
    "pidgin":   "pcm_Latn",
    # "yoruba":    "yor_Latn",
    # "hausa":     "hau_Latn",
    # "igbo":      "ibo_Latn",
    # "english":   "eng_Latn",
}

max_examples = None
data_dir = Path("./naijas2st/test/pidgin")
results_dir = "RESULTS_DIR"
os.makedirs(results_dir, exist_ok=True)

for language, omnilingual_code in language_codes.items():
    print(f"\n→ Processing {language} ({omnilingual_code})…")
    sys.stdout.flush()
    
    # Initialize pipeline fresh for each language
    try:
        print(f"  [DEBUG] Initializing ASR pipeline for {language}...")
        sys.stdout.flush()
        pipeline = ASRInferencePipeline(model_card="omniASR_LLM_1B")
        print(f"  [DEBUG] Pipeline initialized successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"  [ERROR] Failed to initialize pipeline: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        continue
    
    tmp_dir = tempfile.mkdtemp(prefix=f"local_{language}_")

    language_dev_set = data_dir
    language_results = []

    for user in language_dev_set.iterdir():
        print(f"Processing user:{user})")
        # path = user / "recordings/"
        path = user
        print(f"Audio path: {path}")
        for file_path in path.iterdir():
            print("Processing file:", file_path)
            try:
                arr, sr = sf.read(file_path)
                file_name = file_path.name
                tmp_path = os.path.join(tmp_dir, f"test_{file_name}")
                sf.write(tmp_path, arr, sr)

                transcription = pipeline.transcribe([tmp_path], lang=[omnilingual_code])
                language_results.append({"ID": f"{user}_{file_name}", "transcription": transcription})
                print(f"  Processed sample {file_name}")
                print(f"    Transcription: {transcription}")
            except Exception as e:
                print(f"  Error processing sample {file_name}: {e}")            
    
    for file_path in language_dev_set.iterdir():
            print("Processing file:", file_path)
            try:
                arr, sr = sf.read(file_path)
                file_name = file_path.name
                tmp_path = os.path.join(tmp_dir, f"test_{file_name}")
                sf.write(tmp_path, arr, sr)

                transcription = pipeline.transcribe([tmp_path], lang=[omnilingual_code])
                language_results.append({"ID": f"{file_path.stem}", "transcription": transcription})
                print(f"  Processed sample {file_name}")
                print(f"    Transcription: {transcription}")
            except Exception as e:
                print(f"  Error processing sample {file_name}: {e}")      

    out_path = os.path.join(results_dir, f"{language}_transcriptions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(language_results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(language_results)} transcriptions to {out_path}")

