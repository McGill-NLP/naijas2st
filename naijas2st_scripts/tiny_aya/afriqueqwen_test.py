"""Few-shot cascaded LRL->English translation with McGill-NLP AfriqueGemma-12B.

For each configured LRL, reads the cascaded ASR transcription JSON,
loads five ``(LRL, English)`` parallel few-shot examples from
``few_shot/<language>``, builds a single text prompt, and uses
AfriqueGemma to generate the English translation. Results are written
to per-language prediction JSONs.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import os 
from pathlib import Path
import json

model_name = "McGill-NLP/AfriqueGemma-12B"


def main():
    """Few-shot AfriqueGemma-12B LRL -> English translation over cascaded ASR.

    Workflow:
        1. Load ``McGill-NLP/AfriqueGemma-12B`` tokenizer and model
           with ``device_map="auto"`` and ``torch_dtype="auto"``.
        2. For each language in ``language_list``:
            - Load the cascaded ASR JSON for the language.
            - Walk ``few_shot/<language>/`` ``.txt`` files (encoded
              ``latin-1`` to tolerate stray bytes), parse out the
              ``<language>``/``english`` transcription lines, and
              collect ``(source, target)`` tuples in
              ``few_shot_examples``.
            - For every test utterance, build a single text prompt
              listing exactly five demos followed by the test
              transcription, ask the model for up to
              ``max_length=50`` new tokens, decode only the new
              tokens and strip surrounding whitespace.
            - Append ``{"file_name": <ID>, "prediction": <text>}`` to
              ``results``.
        3. Write the predictions to
           ``./RESULTS/naijas2st/cascaded_afrique_gemma/<language>.json``.

    Outputs:
        One predictions JSON per language.

    Returns:
        None.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )


    language_list = ["yoruba",
                    #  "hausa",
                    #  "igbo"
                     ]

    number_of_few_shot_examples = 5
    results_dir = "./RESULTS/naijas2st/cascaded_afrique_gemma"
    os.makedirs(results_dir, exist_ok=True)
    fewshot_base_dir = Path("few_shot/")
    test_base_dir = Path("RESULTS/naijas2st/asr_LLM_1B_test_set_for_cascaded/")
    max_length = 50


    for language in language_list:
        print(f"\n→ Processing {language}...")
        language_format = f"{language}_transcriptions.json"
        test_set = test_base_dir / language_format
        few_shot_dir = fewshot_base_dir / language
        english_dir = fewshot_base_dir / f"{language}_english"
        few_shot_examples = []

        for txt_file in os.listdir(few_shot_dir):
            with open(os.path.join(few_shot_dir, txt_file), 'r', encoding='latin-1') as f:
                lines = f.read().strip().split('\n')
            original_transcription = ""
            english_translation = ""
            for line in lines:
                if line.lower().startswith(f'{language.lower()} transcription:'):
                    original_transcription = line.split(':', 1)[1].strip()

                if line.lower().startswith('english transcription:'):
                    english_translation = line.split(':', 1)[1].strip()

            if not english_translation:
                continue
            if not original_transcription:
                continue

            # Store file path and translation for later upload
            few_shot_examples.append((original_transcription, english_translation))

        results = []
        with open(test_set, 'r') as test_set_json:
            for utterance in json.load(test_set_json):
                print(utterance["transcription"][0])
                id = utterance["ID"]
                input = utterance["transcription"][0]

                prompt = f"You are a professional translator. Here are 5 examples of {language} transcriptions and their English translations: \
                    {few_shot_examples[0]} \n {few_shot_examples[1]} \n {few_shot_examples[2]} \n {few_shot_examples[3]} \n {few_shot_examples[4]} \n \
                    Following these examples, translate the following {language} transcription to English.\
                    When formatting the answer, only output the English translation without any additional text or formatting. \n {input}\n "
                
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                # Generate text
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                )

                new_tokens = generated_ids[0][inputs.input_ids.shape[-1]:]
                output = tokenizer.decode(new_tokens, skip_special_tokens=True)

                results.append({
                    "file_name": id,
                    "prediction": output.strip(),
                    })
                
        # Save results to JSON
        output_json = Path(f"{results_dir}/{language}.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Saved results to {output_json}")


if __name__ == "__main__":
    main()
