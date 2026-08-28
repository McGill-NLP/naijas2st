"""Step 2: ask Gemini to grade each step-1 translation on a 0-100 scale.

Reads per-language ``stage1`` translation JSONs and asks Gemini 2.5 Flash
to score each English machine translation against the LRL source on a
0-100 quality scale, producing per-language ``stage2`` JSONs with the
score reasoning attached.
"""

from google import genai
from datasets import load_dataset
from huggingface_hub import login
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



def main():
    """Score each stage-1 translation on 0-100 quality with Gemini 2.5 Flash.

    Workflow (per FLEURS language in ``language_codes``):
        1. Authenticate with HuggingFace via ``HF_TOKEN`` and
           instantiate a Gemini client.
        2. Open ``<translations_dataset>/<code>.json`` (the stage-1
           output containing ``prediction``, ``reference`` and
           ``source`` per item).
        3. For every sentence, build a quality-scoring prompt that
           shows the LRL source and the model translation and asks the
           model to rate it from ``[0]`` (nonsense) to ``[100]``
           (perfect) along with a short rationale, then provide the
           score.
        4. Call ``gemini-2.5-flash`` with up to 5 retries.
        5. Append ``{"id", "prediction": <scoring text>, "reference",
           "source"}`` to the per-language results list. The model's
           textual ``[score]`` is parsed out by a downstream script
           (``score_extraction.py``).
        6. Write ``<results_dir>/<code>.json`` per language.

    Outputs:
        ``./RESULTS/stage2/<code>.json`` for every language.

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
        "Welsh":     "cy_gb",
        "Swahili":   "sw_ke",
        "Yoruba":    "yo_ng",
        "Hausa":     "ha_ng",
        "Igbo":      "ig_ng",
        "Luganda":   "lg_ug",
    }


    # Make placeholder later
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    translations_dataset = "./RESULTS/stage1"
    results_dir = "./RESULTS/stage2"
    os.makedirs(results_dir, exist_ok=True)

    for language_name, code in language_codes.items():
        print(f"\n→ Processing {language_name} ({code})…")

        out_path = os.path.join(results_dir, f"{code}.json")
        results = []
        test_dataset = f"{translations_dataset}/{code}.json"

        with open(test_dataset) as fp:
            test_data = json.load(fp)

        # Iterate test split as a stream
        for sentence in test_data:
            print(f"  ↳ {sentence['id']}")
            id = sentence["id"]
            prediction = sentence["prediction"]
            reference = sentence["reference"]
            source = sentence["source"]

            # Build the prompt with few-shot examples using inline data
            # Create multi-turn conversation for few-shot learning
            prompt_parts = [f"You are a translation expert. Please evaluate the given English machine translation based on the source sentence in {language_name} on a scale from [0] to [100], following these quality levels:\n[0] Nonsens/No meaning preserved\n[100] Perfect/All meaning preserved. Provide the rational of the score rating, followed by the score.\n"]
            prompt_parts.append(f"Source sentence in {language_name}: {source}\n")
            prompt_parts.append(f"Machine translation in English: {prediction}\n")   

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
                                # "file_name": file_name,
                                "id": id,
                                "prediction": resp.text.strip(),
                                "reference": reference,
                                "source": source
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
