"""Step 3 (zero-shot): re-translate using the stage-2 score as conditioning.

Reads stage-2 scored translations and asks Gemini to produce a refined
translation; the score from stage 2 is folded into the prompt so the
model can take its prior judgement into account. Outputs per-language
stage-3 JSONs.
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



def main():
    """Re-translate FLEURS using the stage-2 score as conditioning context.

    Workflow:
        1. Authenticate with HuggingFace, instantiate a Gemini client,
           and create the output directory.
        2. For each FLEURS language:
            - Stream the FLEURS test split for the LRL source.
            - Load the stage-2 results JSON for that language so the
              model's prior English translation and its self-assigned
              score are available per ID.
            - For each FLEURS sample, write a temp WAV and upload it.
            - Build a prompt that shows the prior translation and its
              score and asks Gemini for a refined English translation
              (zero-shot, no demonstrations).
            - Call Gemini with retries; append ``{"id", "prediction",
              "reference", "source"}`` and rewrite the per-language
              stage-3 JSON after every utterance.

    Inputs:
        Stage-2 ``<results_dir>/<code>.json`` files; FLEURS test
        audio.

    Outputs:
        ``./RESULTS/stage3/<code>.json`` per language with refined
        predictions.

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

    max_examples = None
    results_dir = "./RESULTS/stage3"
    os.makedirs(results_dir, exist_ok=True)

    for language_name, code in language_codes.items():
        print(f"\n→ Processing {language_name} ({code})…")
        scored_translation_dir = f"./RESULTS/stage2/{code}_with_scores_merged.json"

        # Load streaming test dataset
        ds_test = load_dataset("google/fleurs", code, split="test", streaming=True)
        ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16_000))

        tmp_dir = tempfile.mkdtemp(prefix=f"fleurs_{code}_")

        out_path = os.path.join(results_dir, f"{code}.json")
        results = []

        # Iterate test split as a stream
        test_iter = ds_test if max_examples is None else islice(ds_test, max_examples)
        for sample in test_iter:
            print(f"  ↳ {sample['id']}")
            
            arr, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
            file_name = f"{sample['id']}.wav"
            tmp_path = os.path.join(tmp_dir, f"test_{file_name}")
            sf.write(tmp_path, arr, sr)

            with open(scored_translation_dir) as fp:
                scored_translations = json.load(fp)
                for sentence in scored_translations:
                    if sentence["id"] == sample["id"]:
                        scored_prediction = sentence["prediction_original"]
                        score = sentence["score"]
                        reference = sentence["reference"]
                        source = sentence["source"]
                        print(f"    ↳ Found scored translation with score {score} for ID {sample['id']}")
                        break

            prompt_parts = [f"You are a translation expert. The following audio in {language_name} was previously machine translated to English and this machine translation was scored on its accuracy from [0] to [100]. Given the audio, the machine translation and its given score, transcribe the audio from {language_name} and use the transcription to translate it to English accurately, avoiding the mistakes made in the previous machine translation according to the score. Provide only the new translation, not the transcription or any additional text or formatting. \n"]

            test_audio = client.files.upload(file=tmp_path)
            prompt_parts.append(test_audio)
            prompt_parts.append(f"\nPrevious machine translation: {scored_prediction}\n")
            prompt_parts.append(f"Score of previous machine translation: {score}\n")
            prompt_parts.append(f"Please provide the new accurate English translation based on the audio, its transcription and the previously scored machine translation. Provide only the new translation, without any additional text or formatting.\n")


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
                                "id": sample["id"],
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
