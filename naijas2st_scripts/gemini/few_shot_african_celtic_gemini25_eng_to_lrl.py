"""Few-shot English audio -> LRL text translation with Gemini 2.5 Flash.

Gemini 2.5 sibling of ``few_shot_naijas2st_eng_to_lrl.py``: chooses
the appropriate English accent few-shot directory per target LRL,
uploads demonstration audios once, then translates each English test
utterance into the target LRL. Supports resumption from existing
per-language results JSON.

NOTE: ``source`` and ``reference`` fields for SSA-COMET are added later
by ``reformat_files.py``; they are intentionally not produced here.
"""

from google import genai
import tempfile
import soundfile as sf
import os
import json
from pathlib import Path


def main():
    """Few-shot English audio -> LRL text translation with Gemini 2.5 Flash.

    Workflow (per target LRL in ``accent_dictionary``):
        1. Pick the English accent test directory matching the LRL
           (``english_north_accent`` for Hausa, ``english_south_accent``
           for Yoruba/Igbo/Pidgin).
        2. Load up to ``number_of_few_shot_examples`` few-shot files
           from the accent directory into
           ``(wav_path, english_src, lrl_tgt)`` tuples.
        3. Resume by loading any existing
           ``<results_dir>/<language>.json`` and skipping its file names.
        4. Upload each few-shot wav once to Gemini.
        5. Walk ``english_<accent>/<user>/recordings/*.wav``; for every
           unseen wav, normalise through ``soundfile``, build a
           multimodal prompt with all the demo audios + (transcription,
           translation) lines and the test wav at the end, then call
           ``gemini-2.5-flash`` with up-to-5 retries.
        6. Persist results incrementally: in ``finally``, delete the
           uploaded test wav and rewrite the per-language JSON after
           every utterance.

    Outputs:
        ``RESULTS/naijas2st/few_shot_S2T_eng_to_lrl_gemini25/<lang>.json``.

    Returns:
        None.
    """
    language_list = [
                    # "yoruba",
                    #  "hausa",
                    #  "igbo", 
                    "pidgin"
                     ]

    accent_dictionary = {
                        # "yoruba":"Y",
                        # "hausa":"N",
                        # "igbo":"Y", 
                        "pidgin":"Y"
                        }
    accent_key_dictionary = {"Y": "english_south_accent", "N": "english_north_accent"}
 
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    number_of_few_shot_examples = 5
    max_examples = None
    results_dir = "./RESULTS/naijas2st/few_shot_S2T_eng_to_lrl_gemini25"
    os.makedirs(results_dir, exist_ok=True)
    fewshot_base_dir = Path("./NaijaS2ST_dev_test/few_shot/")
    test_base_dir = Path("./naijas2st_scripts/test/")

    for language, accent in accent_dictionary.items():
        print(f"\n→ Processing English to {language}...")

        test_set = test_base_dir / accent_key_dictionary[accent]
        english_type = accent_key_dictionary[accent]
        print("English accent for this language:", english_type)

        tmp_dir = tempfile.mkdtemp(prefix=f"local_{english_type}_{language}")

        # Load local few-shot examples
        fewshot_dir = fewshot_base_dir / english_type
        fewshot_files = []  # Store file paths instead of uploaded refs
        fewshot_uploaded = []
        
        if os.path.exists(fewshot_dir):
            # Get wav files and sort them
            wav_files = sorted([f for f in os.listdir(fewshot_dir) if f.endswith('.wav')])[:number_of_few_shot_examples]
            
            for wav_file in wav_files:
                wav_path = os.path.join(fewshot_dir, wav_file)
                txt_file = wav_file.replace(f'.wav', '.txt')
                txt_path = os.path.join(fewshot_dir, txt_file)
                print(txt_path)
                
                
                if not os.path.exists(txt_path):
                    print(f"  Warning: Missing {txt_file}, skipping {wav_file}")
                    continue
                
                with open(txt_path, 'r', encoding='utf-8') as f:
                    lines = f.read().strip().split('\n')
                original_english_transcription = ""
                lrl_translation = ""
                for line in lines:
                    print(lines)
                    if line.lower().startswith('english transcription:'):
                        original_english_transcription = line.split(':', 1)[1].strip()
                        print(original_english_transcription)

                    if line.lower().startswith(f'{language.lower()} transcription:'):
                        lrl_translation = line.split(':', 1)[1].strip()
                        print(lrl_translation)

                if not lrl_translation:
                    print(f"  Warning: No {language} transcription found in {txt_file}, skipping")
                    continue
                if not original_english_transcription:
                    print(f"  Warning: No {language} transcription found in {txt_file}, skipping")
                    continue
                
                # Store file path and translation for later upload
                fewshot_files.append((wav_path, original_english_transcription, lrl_translation))
                print(f"  ↳ Loaded few-shot example: {wav_file}")
        else:
            print(f"    Warning: Few-shot directory not found: {fewshot_dir}")
            print(f"  ↳ Proceeding without few-shot examples (zero-shot)")

        out_path = os.path.join(results_dir, f"{language}.json")
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

        if fewshot_files:
            print(f"  ↳ Uploading {len(fewshot_files)} few-shot audio files once")
            for wav_path, original_english_transcription, lrl_translation in fewshot_files:
                try:
                    file_ref = client.files.upload(file=wav_path)
                    fewshot_uploaded.append((file_ref, original_english_transcription, lrl_translation))
                except Exception as e:
                    print(f"    ↳ failed to upload few-shot {wav_path}: {e}")
        else:
            print("  ↳ No few-shot audio to upload")

        for user in test_set.iterdir():
            user_dir = user / "recordings/"
            for file in user_dir.iterdir():
                file_path = file
                file_name = f"{user}_{file_path.stem}"

                # Skip if already processed
                if file_name in processed_files:
                    print(f"  ↳ {file_path.stem} (already processed, skipping)")
                    continue
                
                # file_path = user_dir / file_name
                print(f"  ↳ {file_path.stem}")
                arr, sr = sf.read(file_path)
                tmp_path = os.path.join(tmp_dir, f"test_{file_path.name}")
                sf.write(tmp_path, arr, sr)

                prompt_parts = [f"You are a translation expert. Here are {len(fewshot_files)} examples of English audio transcribed, then translated into {language}. Following these examples, transcribe the last given audio, and use the transcription to provide its exact {language} translation. Return only the {language} translation without any additional text."]
                    
                for i, (file_ref, original_english_transcription, lrl_translation) in enumerate(fewshot_uploaded, 1):
                    prompt_parts.append(f"Example {i} audio:")
                    prompt_parts.append(file_ref)
                    prompt_parts.append(f"Transcription: {original_english_transcription}\n\n")
                    prompt_parts.append(f"\nTranslation: {lrl_translation}\n\n")

                prompt_parts.append(f"Now transcribe this audio and exactly translate the transcription to {language}. Provide only the {language} translation, without any additional text or formatting:")

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
                                    model="gemini-2.5-flash",
                                    contents=prompt_parts,
                                    # config=types.GenerateContentConfig(
                                    #     thinking_config=types.ThinkingConfig(thinking_level="low"))
                                )
                        part = resp.candidates[0].content.parts[0]
                        
                        print(f"response: {resp.text.strip()}")
                        results.append({
                                    "file_name": file_name,
                                    "prediction": resp.text.strip(),
                                    # "reference": lrl_translation,
                                    # "source": original_english_transcription
                                    # "reference": get_fleurs_english_ref_for_one_sample(sample["id"]),
                                    # "source": sample.get("transcription")
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
