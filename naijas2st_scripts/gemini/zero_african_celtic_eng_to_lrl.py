"""Zero-shot English audio -> LRL text translation with Gemini 3.1 Pro.

Counterpart to ``zero_shot_naijas2st.py`` in the English->LRL
direction. Selects the appropriate English accent test directory per
target language and asks Gemini to transcribe and translate each
audio into the target LRL. Supports resumption from existing results.

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
    """Zero-shot English audio -> LRL text translation with Gemini 3.1 Pro.

    Workflow (per target LRL in ``accent_dictionary``):
        1. Pick the English accent test directory aligned with the LRL.
        2. Resume support: if
           ``<results_dir>/<language>.json`` exists, load it and skip
           every ``file_name`` already processed.
        3. Walk ``english_<accent>/<user>/recordings/*.wav``; for every
           unseen wav:
            - Normalise the audio via ``soundfile`` into a temp dir.
            - Upload the normalised wav to the Gemini file store.
            - Build a minimal prompt instructing the model to transcribe
              the English audio and translate it into the target LRL.
            - Call ``gemini-3.1-pro-preview`` with up-to-5 retries.
              Thread the optional ``thought_signature`` returned by the
              reasoning model back into the prompt for follow-up turns.
            - Append ``{"file_name", "prediction"}``; in ``finally``,
              delete the uploaded wav and rewrite the results JSON.

    Outputs:
        ``RESULTS/naijas2st/zero_shot_S2T_eng_to_lrl_gemini31/<lang>.json``.

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

    results_dir = "./RESULTS/naijas2st/zero_shot_S2T_eng_to_lrl_gemini31"
    os.makedirs(results_dir, exist_ok=True)
    test_base_dir = Path("./naijas2st_scripts/test/")

    for language, accent in accent_dictionary.items():
        print(f"\n→ Processing English to {language}...")

        test_set = test_base_dir / accent_key_dictionary[accent]
        english_type = accent_key_dictionary[accent]
        print("English accent for this language:", english_type)


        tmp_dir = tempfile.mkdtemp(prefix=f"local_{english_type}_{language}")

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


        for user in test_set.iterdir():
            user_dir = user / "recordings/"
            # user_dir = user
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

                prompt_parts = [f"You are a translation expert. Transcribe the given English audio, and use the transcription to provide its exact {language} translation. Return only the {language} translation without any additional text."]

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
                        # resp = client.models.generate_content(
                        #     model="gemini-2.5-flash",
                        #     contents=prompt_parts,
                        # )
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

                        print(f"response: {resp.text.strip()}")
                        results.append({
                                    "file_name": file_name,
                                    "prediction": resp.text.strip(),
                                    # "reference": original_english_transcription,
                                    # "source": original_english_transcription
                                    # "reference": get_fleurs_english_ref_for_one_sample(sample["id"]),
                                    # "source": sample.get("transcription"
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
