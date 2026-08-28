"""Zero-shot LRL audio -> English translation with Gemini 2.5 Flash.

Walks the local African/Celtic test set, uploads each audio to the
Gemini file store, and asks the model to transcribe and translate it
to English in one step. No in-context examples are provided.

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
    """Zero-shot LRL audio -> English translation loop using Gemini 2.5 Flash.

    Workflow (per language in ``language_list``):
        1. Set up the Gemini client and ``<results_dir>/<language>.json``
           output path; create a per-language temp dir.
        2. Walk ``naijas2st_scripts/test/<language>/<user>/*.wav``.
           For each test wav:
            - Normalise the audio via ``soundfile`` into the temp dir.
            - Upload the normalised wav to the Gemini file store.
            - Build a minimal prompt instructing the model to transcribe
              the LRL audio and translate it to English (no in-context
              examples).
            - Call ``gemini-2.5-flash`` with up-to-5 retries.
            - Append ``{"file_name", "prediction"}`` to ``results``.
            - In ``finally``, delete the uploaded wav and rewrite
              ``<results_dir>/<language>.json`` after each call so the
              run is resumable on crash.

    Inputs:
        ``GOOGLE_API_KEY``; local test wavs.

    Outputs:
        ``RESULTS/naijas2st/zero_shot_S2T_lrl_to_eng_gemini25/<lang>.json``.

    Returns:
        None.
    """
    language_list = ["pidgin"]
 
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    results_dir = "./RESULTS/naijas2st/zero_shot_S2T_lrl_to_eng_gemini25"
    os.makedirs(results_dir, exist_ok=True)
    test_base_dir = Path("./naijas2st_scripts/test/")

    for language in language_list:
        print(f"\n→ Processing {language}...")

        # Load streaming test dataset
        test_set = test_base_dir / language


        tmp_dir = tempfile.mkdtemp(prefix=f"local_{language}_")
        out_path = os.path.join(results_dir, f"{language}.json")
        results = []

        for user in test_set.iterdir():
            # user_dir = user / "recordings/"
            user_dir = user
            for file in user_dir.iterdir():
                file_path = file
                # file_path = user_dir / file_name
                print(f"  ↳ {file_path.stem}")
                arr, sr = sf.read(file_path)
                tmp_path = os.path.join(tmp_dir, f"test_{file_path.name}")
                sf.write(tmp_path, arr, sr)

                prompt_parts = [f"You are a translation expert. Transcribe the given {language} audio, and use the transcription to provide its exact English translation. Return only the English translation without any additional text."]

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
                        )
                        # resp = client.models.generate_content(
                        #             model="gemini-3.1-pro-preview",
                        #             contents=prompt_parts,
                        #             # config=types.GenerateContentConfig(
                        #             #     thinking_config=types.ThinkingConfig(thinking_level="low"))
                        #         )
                        results.append({
                                    "file_name": f"{user}_{file_path.stem}",
                                    "prediction": resp.text.strip(),
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
