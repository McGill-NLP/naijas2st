"""Zero-shot English -> LRL text translation with the Gemma 3n instruct model.

Reads English ASR transcriptions from the cascaded test set and asks
Gemma 3n to translate each utterance into the target LRL, without
any in-context examples. Writes per-language prediction JSONs.
"""

from pathlib import Path
import os
import json
from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_ID = "google/gemma-3n-E2B-it"

language_list = ["pidgin"]
accent_dir = {
    # 'yoruba': 'english_south_accent',
    # 'igbo': 'english_south_accent',
    # 'hausa': 'english_north_accent', 
    'pidgin': 'english_south_accent'
}
test_base_dir = Path("./RESULTS/naijas2st/asr_LLM_1B_test_set_for_cascaded")


def main():
    """Zero-shot Gemma 3n English -> LRL text translation over cascaded ASR.

    Workflow:
        1. Load the ``google/gemma-3n-E2B-it`` processor and model with
           ``device_map="auto"``.
        2. For each language in ``language_list``:
            - Resolve the English ASR JSON for the language's accent
              (``<accent>_transcriptions.json`` under ``test_base_dir``).
            - For every ASR utterance, build a minimal chat prompt
              (system instruction asking for a clean ``<language>``
              translation, user content with the English transcription),
              tokenise with ``apply_chat_template``, and generate up to
              40 new tokens with the Gemma 3 decoding path
              (``processor.decode`` after slicing off the prompt
              tokens).
            - Append ``{"ID", "transcription", "prediction"}`` to results.
        3. Write the predictions to
           ``./RESULTS/naijas2st/gemma3/eng_to_lrl_zero_shot/<language>.json``.

    Outputs:
        One predictions JSON per language under the zero-shot folder.

    Returns:
        None.
    """
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto"
    )

    for language in language_list:
        print(f"\n→ Processing {language}...")
        language_format = f"{accent_dir[language]}_transcriptions.json"
        test_set = test_base_dir / language_format
        results = []
        with open(test_set, 'r') as test_set_json:
            for utterance in json.load(test_set_json):
                print(utterance["transcription"][0])
                print(utterance["ID"])
                input = utterance["transcription"][0]
                # Prompt
                messages = [
                    {"role": "system", "content": f"Your are an expert English to {language} translator. Translate the following English text to {language}. Only output the {language} translation, without any additional text or formatting."},
                    {"role": "user", "content": f"{input}"},
                ]

                # Process input
                text = processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True, 
                    enable_thinking=False
                )
                inputs = processor(text=text, return_tensors="pt").to(model.device)
                input_len = inputs["input_ids"].shape[-1]

                # # Generate output for Gemma 4
                # outputs = model.generate(**inputs, max_new_tokens=1024)
                # response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

                # # Parse output
                # output = processor.parse_response(response)

                # translation = output['content']

                # Generate outputs for Gemma 3
                outputs = model.generate(**inputs, max_new_tokens=40)
                translation = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])

                results.append({
                    "ID": utterance["ID"],
                    "transcription": utterance["transcription"][0],
                    "prediction": translation
                })
        # Save results to JSON
        output_json = Path(f"./RESULTS/naijas2st/gemma3/eng_to_lrl_zero_shot/{language}.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Saved results to {output_json}")


if __name__ == "__main__":
    main()
