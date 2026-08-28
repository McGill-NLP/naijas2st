"""Step 1: few-shot FLEURS English audio -> LRL text translation with Gemini.

Counterpart of ``step1_few_shot_translation.py`` in the English -> LRL
direction. Uses ``en_us`` audio as the source and asks Gemini for the
target LRL translation, leveraging N parallel few-shot demos.
"""

from google import genai
from datasets import load_dataset, Audio
from huggingface_hub import login
import tempfile
import soundfile as sf
from itertools import islice
import os
import json

def get_fleurs_english_ref_for_one_sample(sample_id):
    """Fetch the English reference transcription for a FLEURS sample.

    Args:
        sample_id (int): FLEURS sample ID to look up.

    Returns:
        str | None: The English transcription string, or ``None`` if
        no matching sample is found.
    """
    ds_en = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
    for sample in ds_en:
        if sample["id"] == sample_id:
            return sample.get("transcription")

def get_fleurs_ref_for_one_sample(sample_id, language_code):
    """Fetch the target-language reference transcription for a FLEURS sample.

    Args:
        sample_id (int): FLEURS sample ID.
        language_code (str): FLEURS language code to load
            (e.g. ``"yo_ng"``).

    Returns:
        str | None: The transcription string for that language, or
        ``None``.
    """
    ds_lang = load_dataset("google/fleurs", language_code, split="test", streaming=True, trust_remote_code=True)
    for sample in ds_lang:
        if sample["id"] == sample_id:
            return sample.get("transcription")



def main():
    """Few-shot FLEURS English audio -> LRL text translation with Gemini.

    Workflow:
        1. Authenticate with HuggingFace via ``HF_TOKEN`` and
           instantiate a Gemini client.
        2. For each ``(language_name, FLEURS_code)`` in
           ``language_codes``:
            - Stream the FLEURS ``en_us`` test split (English source).
            - Load ``few_shot_data/<code>/`` wavs + ``.txt`` files and
              parse the ``english transcription`` and
              ``<language> transcription`` lines into
              ``(wav_path, eng_src, lrl_tgt)`` triples.
            - Upload every demo wav once to the Gemini file store.
            - For each English test sample, write a resampled wav,
              build a multimodal prompt with demos + test audio, and
              call Gemini with retries.
            - Append ``{"id", "prediction", "reference", "source"}``
              where ``reference`` is the FLEURS target-language
              transcription (via
              :func:`get_fleurs_ref_for_one_sample`) and ``source`` is
              the English text.
            - Persist the per-language results JSON after every call.

    Outputs:
        Per-language predictions JSON under
        ``./RESULTS/FLEURS/few_shot_gemini31_eng_to_lrl/<code>.json``.

    Returns:
        None.
    """
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        print("Warning: No HF_TOKEN found. You may hit rate limits.")


    language_codes = {
        # "Irish":    "ga_ie",
        # "Welsh":     "cy_gb",
        # "Swahili":   "sw_ke",
        "Yoruba":    "yo_ng",
        "Hausa":     "ha_ng",
        "Igbo":      "ig_ng",
        # "Luganda":   "lg_ug",
    }


    # Make placeholder later
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    number_of_few_shot_examples = 5
    max_examples = None
    results_dir = "./RESULTS/FLEURS/few_shot_eng_to_lrl_gemini31"
    os.makedirs(results_dir, exist_ok=True)
    fewshot_base_dir = "./fleurs/few_shot_data"

    for language_name, code in language_codes.items():
        print(f"\n→ Processing {language_name} ({code})…")

        # Load streaming test dataset
        ds_test = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
        ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16_000))

        tmp_dir = tempfile.mkdtemp(prefix=f"fleurs_{code}")

        # Load local few-shot examples
        english_code = "en_us"
        fewshot_dir = os.path.join(fewshot_base_dir, english_code)
        fewshot_files = []  # Store file paths instead of uploaded refs
        fewshot_uploaded = []
        
        if os.path.exists(fewshot_dir):
            # Get wav files and sort them
            wav_files = sorted([f for f in os.listdir(fewshot_dir) if f.endswith('.wav')])[:number_of_few_shot_examples]
            
            for wav_file in wav_files:
                wav_path = os.path.join(fewshot_dir, wav_file)
                txt_file = wav_file.replace(f'_en_us.wav', '.txt')
                txt_path = os.path.join(fewshot_dir, txt_file)
                print(txt_path)
                
                if not os.path.exists(txt_path):
                    print(f"  Warning: Missing {txt_file}, skipping {wav_file}")
                    continue
                
                with open(txt_path, 'r', encoding='latin-1') as f:
                    lines = f.read().strip().split('\n')
                lrl_transcription = ""
                english_translation = ""
                for line in lines:
                    if line.lower().startswith(f'{language_name.lower()} transcription:'):
                        lrl_transcription = line.split(':', 1)[1].strip()

                    if line.lower().startswith('english transcription:'):
                        english_translation = line.split(':', 1)[1].strip()

                if not english_translation:
                    print(f"  Warning: No English transcription found in {txt_file}, skipping")
                    continue
                if not lrl_transcription:
                    print(f"  Warning: No {language_name} transcription found in {txt_file}, skipping")
                    continue
                
                # Store file path and translation for later upload
                fewshot_files.append((wav_path, english_translation, lrl_transcription))
                print(f"  ↳ Loaded few-shot example: {wav_file}")
        else:
            print(f"    Warning: Few-shot directory not found: {fewshot_dir}")
            print(f"  ↳ Proceeding without few-shot examples (zero-shot)")

        out_path = os.path.join(results_dir, f"{code}.json")
        results = []

        if fewshot_files:
            print(f"  ↳ Uploading {len(fewshot_files)} few-shot audio files once")
            for wav_path, transcription, translation in fewshot_files:
                try:
                    file_ref = client.files.upload(file=wav_path)
                    fewshot_uploaded.append((file_ref, transcription, translation))
                except Exception as e:
                    print(f"    ↳ failed to upload few-shot {wav_path}: {e}")
        else:
            print("  ↳ No few-shot audio to upload")

        processed_files = set()
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as fp:
                    results = json.load(fp)
                    processed_files = {result["file_name"] for result in results}
                    print(f"  ↳ Loaded {len(results)} existing results from {out_path}")
            except Exception as e:
                print(f"  ↳ Could not load existing results: {e}")

        # Iterate test split as a stream
        test_iter = ds_test if max_examples is None else islice(ds_test, max_examples)
        for sample in test_iter:
            print(f"  ↳ {sample['id']}")
            id = sample["id"]

            if id in processed_files:
                print(f"  ↳ {id} (already processed, skipping)")
                continue
            
            arr, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
            file_name = f"{sample['id']}.wav"
            tmp_path = os.path.join(tmp_dir, f"test_{file_name}")
            sf.write(tmp_path, arr, sr)

            # Build the prompt with few-shot examples using inline data
            # Create multi-turn conversation for few-shot learning
            prompt_parts = [f"You are a translation expert. Here are {len(fewshot_files)} examples of English audio transcribed, then translated into {language_name}. Following these examples, transcribe the last given audio, and use the transcription to provide its {language_name} translation. Return only the {language_name} translation without any additional text"]
                
            for i, (file_ref, transcription, translation) in enumerate(fewshot_uploaded, 1):
                prompt_parts.append(f"Example {i} audio:")
                prompt_parts.append(file_ref)
                prompt_parts.append(f"Transcription: {transcription}\n\n")
                prompt_parts.append(f"\nTranslation: {translation}\n\n")
                
            prompt_parts.append(f"Now transcribe this audio and translate the transcription to {language_name}. Provide only the {language_name} translation, without any additional text or formatting:")

            test_audio = client.files.upload(file=tmp_path)
            prompt_parts.append(test_audio)
            success = False
            number_of_retries = 0
            while not success:
                print(f"    ↳ Attempt {number_of_retries + 1}")
                number_of_retries += 1
                if number_of_retries > 5:
                    print(f"  ↳ failed to process, moving on...")
                    break
                try:
                    resp = client.models.generate_content(
                                model="gemini-3.1-pro-preview",
                                contents=prompt_parts,
                                # config=types.GenerateContentConfig(
                                #     thinking_config=types.ThinkingConfig(thinking_level="low"))
                            )
                    part = resp.candidates[0].content.parts[0]
                    if len(prompt_parts)==23 and part.thought_signature:
                        prompt_parts.append(part.thought_signature)
                    elif len(prompt_parts)>23 and part.thought_signature:
                        prompt_parts[-1]=part.thought_signature

                    results.append({
                                "id": sample["id"],
                                "prediction": resp.text.strip(),
                                "reference": get_fleurs_ref_for_one_sample(sample["id"], code),
                                "source": sample.get("transcription")
                            })
                    success = True
                except Exception as e:
                    success = False
                    print(f"caught error, retrying: {type(e).__name__}")
                finally:
                    try:
                        if hasattr(test_audio, 'name'):
                            client.files.delete(name=test_audio.name)
                            print(f"    ↳ Deleted test file: {test_audio.name}")
                    except Exception as e:
                        print(f"    ↳ Failed to delete test file: {e}")

                    # Save results per language
                    with open(out_path, "w", encoding="utf-8") as fp:
                        json.dump(results, fp, ensure_ascii=False, indent=2)
        

        print(f"  ↳ saved {len(results)} translations to {out_path}")


if __name__ == "__main__":
    main()
