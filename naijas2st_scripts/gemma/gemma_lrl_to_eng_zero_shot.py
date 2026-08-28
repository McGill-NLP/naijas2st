"""Zero-shot LRL -> English text translation with Gemma 4 (or 3n).

Reads LRL ASR transcriptions from the cascaded test set and asks Gemma
to translate each utterance to English without any in-context
examples. Writes per-language prediction JSONs to ``RESULTS/.../gemma4/
lrl_to_eng_zero_shot/<language>.json``.
"""

from pathlib import Path
import json
from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_ID = "google/gemma-4-E4B-it"
# MODEL_ID = "google/gemma-3n-E2B-it"

language_list = [
                # 'yoruba',
                # 'igbo',
                # 'hausa', 
                'pidgin'
                ]
test_base_dir = Path("./RESULTS/naijas2st/asr_LLM_1B_test_set_for_cascaded")


def main():
    """Zero-shot Gemma 4 LRL -> English text translation over cascaded ASR.

    Workflow:
        1. Load the ``google/gemma-4-E4B-it`` processor and model with
           ``device_map="auto"``.
        2. For each language in ``language_list``:
            - Load the cascaded LRL ASR JSON
              (``<language>_transcriptions.json``).
            - For every utterance, build a Gemma 4 chat prompt
              (system: "expert <language> to English translator", user:
              the LRL transcription), tokenise it, and generate up to
              1024 new tokens. Use ``processor.parse_response`` to pull
              the assistant content out of the raw decoded string
              (Gemma 4 uses tagged outputs).
            - Append ``{"ID", "transcription", "prediction"}`` to results.
        3. Write per-language predictions to
           ``./RESULTS/naijas2st/gemma4/lrl_to_eng_zero_shot/<language>.json``.

    Outputs:
        One predictions JSON per language.

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
        language_format = f"{language}_transcriptions.json"
        test_set = test_base_dir / language_format
        results = []
        with open(test_set, 'r') as test_set_json:
            for utterance in json.load(test_set_json):
                print(utterance["transcription"][0])
                print(utterance["ID"])
                input = utterance["transcription"][0]
                # Prompt
                messages = [
                    {"role": "system", "content": f"Your are an expert {language} to English translator. Translate the following {language} text to English. Only output the English translation, without any additional text or formatting."},
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

                # Generate output Gemma 4
                outputs = model.generate(**inputs, max_new_tokens=1024)
                response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
                # Parse output Gemma 4
                output = processor.parse_response(response)
                translation = output['content']

                results.append({
                    "ID": utterance["ID"],
                    "transcription": utterance["transcription"][0],
                    "prediction": translation
                })
        # Save results to JSON
        output_json = Path(f"./RESULTS/naijas2st/gemma4/lrl_to_eng_zero_shot/{language}.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Saved results to {output_json}")


if __name__ == "__main__":
    main()
