"""Cascaded ASR -> translation pipeline using Gemini.

Reads pre-computed ASR transcriptions for several low-resource languages
and uses Gemini to translate them into English, saving the predictions
as JSON files per language.
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
    """Translate pre-computed ASR transcriptions into English with Gemini.

    Workflow:
        1. Authenticate with HuggingFace via ``HF_TOKEN`` if available
           (only used to keep API quota healthy; no datasets are loaded here).
        2. Instantiate a Gemini client from ``GOOGLE_API_KEY``.
        3. For each ``(language_name, FLEURS_code)`` in ``language_codes``,
           load ``./RESULTS/asr/<code>_transcriptions.json``.
        4. For every utterance, prompt ``gemini-2.5-flash`` with the LRL
           transcription and ask for the English translation, retrying up
           to 5 times on transient API errors.
        5. Append ``{"id", "prediction"}`` to a per-language results list
           and write it to ``./RESULTS/asr_translation/<code>.json``.

    Inputs (env vars / files):
        ``HF_TOKEN``, ``GOOGLE_API_KEY`` env vars; per-language
        ``./RESULTS/asr/<code>_transcriptions.json`` files.

    Outputs:
        ``./RESULTS/asr_translation/<code>.json`` for every configured
        language, each a list of prediction records.

    Returns:
        None.
    """
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        print("Warning: No HF_TOKEN found. You may hit rate limits.")


    language_codes = {
        "Irish":    "ga_ie",
        "Welsh":     "cy_gb",
        "Swahili":   "sw_ke",
        "Yoruba":    "yo_ng",
        "Hausa":     "ha_ng",
        "Igbo":      "ig_ng",
    }


    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    transcriptions_dataset = "./RESULTS/asr"
    results_dir = "./RESULTS/asr_translation"
    os.makedirs(results_dir, exist_ok=True)

    for language_name, code in language_codes.items():
        print(f"\n→ Processing {language_name} ({code})…")

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

            prompt_parts = [f"You are a translation expert. Please translate the following {language_name} sentence into English, ensuring that the meaning is preserved as accurately as possible. Provide the translation without any additional commentary of formatting.\n"]
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
