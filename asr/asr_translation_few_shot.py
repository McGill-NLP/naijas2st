"""Few-shot cascaded ASR -> translation pipeline using Gemini.

Reads ASR transcriptions for low-resource languages, augments the prompt
with N text-only few-shot examples drawn from local ``few_shot_data``
files, and asks Gemini to translate each sample into English.
"""

from google import genai
from datasets import load_dataset
from huggingface_hub import login
import os
import json

def get_fleurs_english_ref_for_one_sample(sample_id):
    """Fetch the English reference transcription for a FLEURS sample.

    Args:
        sample_id (int): FLEURS sample ID to look up.

    Returns:
        str | None: The English transcription string, or ``None`` if no
        sample with that ID is found in the streamed split.
    """
    ds_en = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
    for sample in ds_en:
        if sample["id"] == sample_id:
            return sample.get("transcription")


def main():
    """Translate ASR transcriptions into English with few-shot Gemini prompts.

    Workflow:
        1. Authenticate with HuggingFace via ``HF_TOKEN`` (optional) and
           instantiate a Gemini client from ``GOOGLE_API_KEY``.
        2. For each ``(language_name, FLEURS_code)`` in ``language_codes``,
           read the first ``number_of_few_shot_examples`` parallel
           ``.txt`` files under ``few_shot_data/<code>/``; each file is
           expected to contain ``<language> transcription:`` and
           ``English transcription:`` lines that are parsed into a
           ``(source, target)`` pair.
        3. Load the cascaded ASR transcriptions JSON
           (``<transcriptions_dataset>/<code>_transcriptions.json``).
        4. For every test utterance, build a prompt that lists the 5
           few-shot examples followed by the test source, and ask
           ``gemini-2.5-flash`` for the English translation (with
           up-to-5 retries on transient errors).
        5. Append ``{"id", "prediction"}`` to the per-language results
           list and write it to ``<results_dir>/<code>.json``.

    Inputs:
        ``HF_TOKEN``, ``GOOGLE_API_KEY`` env vars; per-language ASR
        JSONs and ``few_shot_data/<code>/*.txt`` examples.

    Outputs:
        ``<results_dir>/<code>.json`` for every configured language.

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
        # "Yoruba":    "yo_ng",
        # "Hausa":     "ha_ng",
        # "Igbo":      "ig_ng",
        "Luganda":   "lg_ug",
    }


    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    transcriptions_dataset = "./RESULTS/asr_CTC-7B"
    number_of_few_shot_examples = 5
    max_examples = None
    results_dir = "./RESULTS/asr_translation_CTC_7B_few_shot"
    fewshot_base_dir = "few_shot_data"
    os.makedirs(results_dir, exist_ok=True)

    for language_name, code in language_codes.items():
        print(f"\n→ Processing {language_name} ({code})…")

         # Load local few-shot examples
        fewshot_dir = os.path.join(fewshot_base_dir, code)
        fewshot_files = []  # Store file paths instead of uploaded refs
        
        if os.path.exists(fewshot_dir):
            # Get wav files and sort them
            txt_files = sorted([f for f in os.listdir(fewshot_dir) if f.endswith('.txt')])[:number_of_few_shot_examples]

            for txt_path in txt_files:
                print(f"  ↳ Processing few-shot file: {txt_path}")
                txt_file = f"./{os.path.join(fewshot_dir, txt_path)}"
                print(txt_file)
                if not os.path.exists(txt_file):
                    print(f"  Warning: Missing {txt_file}, skipping {txt_file}")
                    continue
                
                with open(txt_file, 'r', encoding='utf-8') as f:
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
                    print(f"  Warning: No English transcription found in {txt_path}, skipping")
                    continue
                if not original_transcription:
                    print(f"  Warning: No {language_name} transcription found in {txt_path}, skipping")
                    continue
                
                # Store file path and translation for later upload
                fewshot_files.append((original_transcription, english_translation))
        else:
            print(f"    Warning: Few-shot directory not found: {fewshot_dir}")
            print(f"  ↳ Proceeding without few-shot examples (zero-shot)")

        out_path = os.path.join(results_dir, f"{code}.json")
        results = []
        language_transcriptions_dataset = f"{transcriptions_dataset}/{code}_transcriptions.json"

        with open(language_transcriptions_dataset) as fp:
            transcriptions = json.load(fp)

        # Iterate test split as a stream
        for sentence in transcriptions:
            print(f"  ↳ {sentence['ID']}")
            id = sentence["ID"]
            transcription = sentence["transcription"]

            prompt_parts = [f"You are a translation expert. Here are {number_of_few_shot_examples} examples of {language_name} sentences translated into English. Following those examples, translate the following {language_name} sentence into English, ensuring that the meaning is preserved as accurately as possible. Provide the translation without any additional commentary or formatting.\n"]
            
            prompt_parts.append(f"{language_name} Example 1: {fewshot_files[0][0]}\n English translation: {fewshot_files[0][1]}\n\n")
            prompt_parts.append(f"{language_name} Example 2: {fewshot_files[1][0]}\n English translation: {fewshot_files[1][1]}\n\n")
            prompt_parts.append(f"{language_name} Example 3: {fewshot_files[2][0]}\n English translation: {fewshot_files[2][1]}\n\n")
            prompt_parts.append(f"{language_name} Example 4: {fewshot_files[3][0]}\n English translation: {fewshot_files[3][1]}\n\n")
            prompt_parts.append(f"{language_name} Example 5: {fewshot_files[4][0]}\n English translation: {fewshot_files[4][1]}\n\n") 
            
            prompt_parts.append(f"Source sentence in {language_name}: {transcription}\n")
            prompt_parts.append(f"English translation:\n")

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
                                contents=prompt_parts
                            )

                    results.append({
                                "id": id,
                                "prediction": resp.text.strip(),
                            })
                    success = True
                except Exception as e:
                    success = False
                    print(f"caught error, retrying: {e}")

        # Save results per language
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)

        print(f"  ↳ saved {len(results)} translations to {out_path}")


if __name__ == "__main__":
    main()
