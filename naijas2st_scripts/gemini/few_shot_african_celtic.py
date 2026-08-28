"""Few-shot LRL audio -> English translation with Gemini 3.1 Pro.

Loads N few-shot ``(wav, source transcription, English translation)``
demonstrations from a per-language ``few_shot/`` directory, uploads
them once to the Gemini file store, and uses them as in-context
examples while iterating over the local African/Celtic test set. The
final English prediction per audio is saved into a results JSON.

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
    """Few-shot LRL audio -> English translation loop using Gemini 3.1 Pro.

    Workflow (per language in ``language_list``):
        1. Create a Gemini client from ``GOOGLE_API_KEY``, set up the
           output directory, and locate ``few_shot/<language>`` and
           ``naijas2st_scripts/test/<language>``.
        2. Load the first ``number_of_few_shot_examples`` few-shot
           examples: for each ``*.wav`` find the sibling ``*.txt``
           containing ``<language> transcription:`` and
           ``english transcription:`` lines and parse them into
           ``(wav_path, src_text, tgt_text)`` tuples. If the directory
           is missing, fall back to zero-shot.
        3. Upload every few-shot wav once to the Gemini file store
           (kept in ``fewshot_uploaded`` as
           ``(file_ref, src, tgt)`` triples).
        4. Walk the test set
           ``naijas2st_scripts/test/<language>/<user>/*.wav``. For
           every wav:
            - Round-trip through ``soundfile`` into a temp WAV to
              normalise the header.
            - Build the prompt: a system-style instruction, then for
              each demo append ``["Example i audio:", file_ref,
              "Transcription: ...", "Translation: ..."]``, then a
              closing instruction and the freshly uploaded test wav.
            - Call ``gemini-3.1-pro-preview`` with up-to-5 retries.
              Manage the ``thought_signature`` returned by reasoning
              calls so it can be threaded back into the prompt for
              follow-up turns.
            - Append ``{"file_name", "prediction"}`` to ``results``.
            - In ``finally``, delete the uploaded test wav from the
              Gemini file store and rewrite ``<results_dir>/<lang>.json``
              after every utterance so progress is persisted.

    Inputs:
        ``GOOGLE_API_KEY`` env var; ``few_shot/<lang>`` wav+txt pairs;
        ``naijas2st_scripts/test/<lang>/<user>/*.wav``.

    Outputs:
        ``RESULTS/naijas2st/few_shot_S2T_lrl_to_eng_gemini31/<lang>.json``
        rewritten incrementally as predictions arrive.

    Returns:
        None.
    """
    language_list = ["pidgin"]
 
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    number_of_few_shot_examples = 5
    max_examples = None
    results_dir = "./RESULTS/naijas2st/few_shot_S2T_lrl_to_eng_gemini31"
    os.makedirs(results_dir, exist_ok=True)
    fewshot_base_dir = Path("./few_shot/")
    test_base_dir = Path("./naijas2st_scripts/test/")

    for language in language_list:
        print(f"\n→ Processing {language}...")

        test_set = test_base_dir / language
        tmp_dir = tempfile.mkdtemp(prefix=f"local_{language}_")

        # Load local few-shot examples
        fewshot_dir = fewshot_base_dir / language
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
                original_transcription = ""
                english_translation = ""
                for line in lines:
                    print(lines)
                    if line.lower().startswith(f'{language.lower()} transcription:'):
                        original_transcription = line.split(':', 1)[1].strip()
                        print(original_transcription)
                        
                    if line.lower().startswith('english transcription:'):
                        english_translation = line.split(':', 1)[1].strip()
                        print(english_translation)
                        
                    
                if not english_translation:
                    print(f"  Warning: No English transcription found in {txt_file}, skipping")
                    continue
                if not original_transcription:
                    print(f"  Warning: No {language} transcription found in {txt_file}, skipping")
                    continue
                
                # Store file path and translation for later upload
                fewshot_files.append((wav_path, original_transcription, english_translation))
                print(f"  ↳ Loaded few-shot example: {wav_file}")
        else:
            print(f"    Warning: Few-shot directory not found: {fewshot_dir}")
            print(f"  ↳ Proceeding without few-shot examples (zero-shot)")

        out_path = os.path.join(results_dir, f"{language}.json")
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

                # Build the prompt with few-shot examples using inline data
                # Create multi-turn conversation for few-shot learning
                prompt_parts = [f"You are a translation expert. Here are {len(fewshot_files)} examples of {language} audio transcribed, then translated into English. Following these examples, transcribe the last given audio, and use the transcription to provide its exact English translation. Return only the English translation without any additional text."]
                    
                for i, (file_ref, transcription, translation) in enumerate(fewshot_uploaded, 1):
                    prompt_parts.append(f"Example {i} audio:")
                    prompt_parts.append(file_ref)
                    prompt_parts.append(f"Transcription: {transcription}\n\n")
                    prompt_parts.append(f"\nTranslation: {translation}\n\n")
                    
                prompt_parts.append("Now transcribe this audio and exactly translate the transcription to English. Provide only the English translation, without any additional text or formatting:")

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
                                    "file_name": f"{user}_{file_path.stem}",
                                    "prediction": resp.text.strip(),
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
