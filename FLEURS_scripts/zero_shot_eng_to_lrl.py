"""Zero-shot FLEURS English audio -> LRL text translation with Gemini.

Streams FLEURS ``en_us`` test audio and asks Gemini to translate each
clip into the target LRL with no in-context demonstrations. Predictions
are saved per language.
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
        sample_id (int): FLEURS sample ID.

    Returns:
        str | None: The English transcription string, or ``None``.
    """
    ds_en = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
    for sample in ds_en:
        if sample["id"] == sample_id:
            return sample.get("transcription")

def get_fleurs_ref_for_one_sample(sample_id, language_code):
    """Fetch the target-language reference transcription for a FLEURS sample.

    Args:
        sample_id (int): FLEURS sample ID.
        language_code (str): FLEURS language code (e.g. ``"yo_ng"``).

    Returns:
        str | None: The transcription string, or ``None``.
    """
    ds_lang = load_dataset("google/fleurs", language_code, split="test", streaming=True, trust_remote_code=True)
    for sample in ds_lang:
        if sample["id"] == sample_id:
            return sample.get("transcription")



def main():
    """Zero-shot FLEURS English audio -> LRL text translation with Gemini.

    Workflow:
        1. Authenticate with HuggingFace via ``HF_TOKEN``, instantiate
           a Gemini client, and create the output directory.
        2. For each ``(language_name, FLEURS_code)`` in
           ``language_codes``:
            - Stream the FLEURS ``en_us`` test split as the English
              source.
            - For each sample, write the audio to a temp WAV, upload
              it to the Gemini file store, and build a minimal prompt
              instructing the model to translate the English audio
              into the target LRL.
            - Call Gemini with up to 5 retries on transient errors.
            - Append ``{"id", "prediction", "reference", "source"}``
              where ``reference`` is the FLEURS target-language
              transcription and ``source`` is the English text.
            - Save the per-language JSON after every utterance.

    Outputs:
        ``./RESULTS/FLEURS/zero_shot_eng_to_lrl/<code>.json`` per language.

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
    results_dir = "./RESULTS/FLEURS/zero_shot_eng_to_lrl_gemini31"
    os.makedirs(results_dir, exist_ok=True)
    fewshot_base_dir = "./fleurs/few_shot_data"

    for language_name, code in language_codes.items():
        print(f"\n→ Processing {language_name} ({code})…")

        # Load streaming test dataset
        ds_test = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
        ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16_000))

        tmp_dir = tempfile.mkdtemp(prefix=f"fleurs_{code}_en")

        out_path = os.path.join(results_dir, f"{code}.json")
        results = []
       
        # Load existing results if the file exists
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
            # Skip if already processed
            file_name = sample["id"]

            if file_name in processed_files:
                print(f"  ↳ {file_name} (already processed, skipping)")
                continue
            
            arr, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
            file_name = f"{sample['id']}.wav"
            tmp_path = os.path.join(tmp_dir, f"test_{file_name}")
            sf.write(tmp_path, arr, sr)

            # Build the prompt with few-shot examples using inline data
            # Create multi-turn conversation for few-shot learning
            prompt_parts = [f"You are a translation expert. Transcribe the given English audio, and use the transcription to provide its {language_name} translation. Return only the {language_name} translation without any additional text"]

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
                    print(f"caught error, retrying: {e}")
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
