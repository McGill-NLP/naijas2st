"""Step 1: few-shot FLEURS LRL audio -> English translation with Gemini 3.1 Pro.

For each configured FLEURS LRL language, streams the test set and runs
Gemini 3.1 Pro speech translation prompted with N parallel
``(LRL audio, LRL transcription, English translation)`` few-shot
demonstrations uploaded once per language. Predictions, references and
sources are written incrementally to per-language JSONs.
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
    """Fetch the English reference transcription for one FLEURS sample.

    Args:
        sample_id (int): FLEURS sample ID to look up.

    Returns:
        str | None: The English transcription string, or ``None`` if
        no sample with that ID is found in the streamed split.
    """
    ds_en = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
    for sample in ds_en:
        if sample["id"] == sample_id:
            return sample.get("transcription")



def main():
    """Few-shot FLEURS LRL audio -> English translation with Gemini 3.1 Pro.

    Workflow (per FLEURS language in ``language_codes``):
        1. Authenticate with HuggingFace via ``HF_TOKEN`` if set and
           instantiate a Gemini client from ``GOOGLE_API_KEY``.
        2. Stream the FLEURS test split for the language, casting
           ``audio`` to 16 kHz; create a per-language temp dir.
        3. Load ``few_shot_data/<code>/`` wavs (first
           ``number_of_few_shot_examples``) and parse the sibling
           ``.txt`` for ``<language> transcription:`` and
           ``english transcription:`` lines into
           ``(wav_path, src, tgt)`` triples.
        4. Upload each demo wav once to the Gemini file store.
        5. For every test sample (optionally capped by
           ``max_examples``):
            - Write the resampled audio to a temp WAV.
            - Build a prompt: instruction, then ``[demo audio, demo
              transcription, demo translation]`` per few-shot, then
              the test audio at the end.
            - Call ``gemini-3.1-pro-preview`` with up-to-5 retries.
              Manage ``thought_signature`` for follow-up reasoning
              turns.
            - Append ``{"id", "prediction", "reference", "source"}``
              where ``reference`` is fetched from FLEURS English via
              :func:`get_fleurs_english_ref_for_one_sample` and
              ``source`` is the LRL FLEURS transcription.
            - In ``finally``, delete the uploaded test audio and
              rewrite ``<results_dir>/<code>.json`` so progress is
              persisted after every utterance.

    Outputs:
        ``./RESULTS/FLEURS/few_shot_gemini31/<code>.json`` per language.

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
    results_dir = "./RESULTS/FLEURS/few_shot_gemini31"
    os.makedirs(results_dir, exist_ok=True)
    fewshot_base_dir = "few_shot_data"

    for language_name, code in language_codes.items():
        print(f"\n→ Processing {language_name} ({code})…")

        # Load streaming test dataset
        ds_test = load_dataset("google/fleurs", code, split="test", streaming=True)
        ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16_000))

        tmp_dir = tempfile.mkdtemp(prefix=f"fleurs_{code}_")

        # Load local few-shot examples
        fewshot_dir = os.path.join(fewshot_base_dir, code)
        fewshot_files = []  # Store file paths instead of uploaded refs
        fewshot_uploaded = []
        
        if os.path.exists(fewshot_dir):
            # Get wav files and sort them
            wav_files = sorted([f for f in os.listdir(fewshot_dir) if f.endswith('.wav')])[:number_of_few_shot_examples]
            
            for wav_file in wav_files:
                wav_path = os.path.join(fewshot_dir, wav_file)
                txt_file = wav_file.replace(f'_{code}.wav', '.txt')
                txt_path = os.path.join(fewshot_dir, txt_file)
                print(txt_path)
                
                
                if not os.path.exists(txt_path):
                    print(f"  Warning: Missing {txt_file}, skipping {wav_file}")
                    continue
                
                with open(txt_path, 'r', encoding='utf-8') as f:
                    lines = f.read().strip().split('\n')
                original_transcription = ""
                english_translation = ""
                for line in lines:
                    print(lines)
                    if line.lower().startswith(f'{language_name.lower()} transcription:'):
                        original_transcription = line.split(':', 1)[1].strip()
                        print(original_transcription)
                        
                    if line.lower().startswith('english transcription:'):
                        english_translation = line.split(':', 1)[1].strip()
                        print(english_translation)
                        
                    
                if not english_translation:
                    print(f"  Warning: No English transcription found in {txt_file}, skipping")
                    continue
                if not original_transcription:
                    print(f"  Warning: No {language_name} transcription found in {txt_file}, skipping")
                    continue
                
                # Store file path and translation for later upload
                fewshot_files.append((wav_path, original_transcription, english_translation))
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

        # Iterate test split as a stream
        test_iter = ds_test if max_examples is None else islice(ds_test, max_examples)
        for sample in test_iter:
            print(f"  ↳ {sample['id']}")
            
            arr, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
            file_name = f"{sample['id']}.wav"
            tmp_path = os.path.join(tmp_dir, f"test_{file_name}")
            sf.write(tmp_path, arr, sr)

            # Build the prompt with few-shot examples using inline data
            # Create multi-turn conversation for few-shot learning
            prompt_parts = [f"You are a translation expert. Here are {len(fewshot_files)} examples of {language_name} audio transcribed, then translated into English. Following these examples, transcribe the last given audio, and use the transcription to provide its English translation. Return only the English translation without any additional text"]
                
            for i, (file_ref, transcription, translation) in enumerate(fewshot_uploaded, 1):
                prompt_parts.append(f"Example {i} audio:")
                prompt_parts.append(file_ref)
                prompt_parts.append(f"Transcription: {transcription}\n\n")
                prompt_parts.append(f"\nTranslation: {translation}\n\n")
                
            prompt_parts.append("Now transcribe this audio and translate the transcription to English. Provide only the English translation, without any additional text or formatting:")

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
                        print("appended thought signature to prompt parts")
                    elif len(prompt_parts)>23 and part.thought_signature:
                        prompt_parts[-1]=part.thought_signature
                        print("updated thought signature in prompt parts")
                    
                    print(f"response: {resp.text.strip()}")
                    results.append({
                                "id": sample["id"],
                                "prediction": resp.text.strip(),
                                "reference": get_fleurs_english_ref_for_one_sample(sample["id"]),
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
