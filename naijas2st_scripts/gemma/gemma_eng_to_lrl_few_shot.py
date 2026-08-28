"""Few-shot English -> LRL text translation with the Gemma 3n instruct model.

Reads English ASR transcriptions of the cascaded test set, loads N
parallel few-shot ``(LRL, English)`` examples from
``few_shot/<accent>``, builds a system + user chat prompt, and uses
Gemma 3n to translate each English utterance into the target LRL.
Results are written to a per-language JSON.
"""

from pathlib import Path
import os
import json
from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_ID = "google/gemma-3n-E2B-it"

language_list = ["pidgin"]
test_base_dir = Path("./RESULTS/naijas2st/asr_LLM_1B_test_set_for_cascaded")

number_of_few_shot_examples = 5
fewshot_base_dir = Path("few_shot/")

accent_dir = {
    # 'yoruba': 'english_south_accent',
    # 'igbo': 'english_south_accent',
    # 'hausa': 'english_north_accent', 
    'pidgin': 'english_south_accent'
}

# Module-level model handles populated in main(); helpers below reference them.
processor = None
model = None


def load_few_shot_examples(language: str) -> list:
    """Read ``(LRL, English)`` parallel pairs from the accent few-shot directory.

    Args:
        language (str): Target LRL name (used to find the LRL
            transcription line in each ``.txt`` file).

    Returns:
        list[tuple[str, str]]: List of ``(lrl_source, english_target)``
        string tuples; files missing either transcription are skipped.
    """
    language_format = f"{accent_dir[language]}"
    few_shot_dir = fewshot_base_dir / language_format
    examples = []
    for txt_file in os.listdir(few_shot_dir):
        with open(os.path.join(few_shot_dir, txt_file), 'r', encoding='latin-1') as f:
            lines = f.read().strip().split('\n')
        src, tgt = "", ""
        for line in lines:
            if line.lower().startswith(f'{language.lower()} transcription:'):
                src = line.split(':', 1)[1].strip()
            if line.lower().startswith('english transcription:'):
                tgt = line.split(':', 1)[1].strip()
        if src and tgt:
            examples.append((src, tgt))
    return examples


def build_prompt(language: str, few_shot_examples: list, input_text: str) -> str:
    """Format a Gemma chat prompt with few-shot examples and the input text.

    Args:
        language (str): Target LRL name shown in the prompt.
        few_shot_examples (list[tuple[str, str]]): ``(src, tgt)`` pairs
            to inline as demonstrations.
        input_text (str): English text to be translated.

    Returns:
        str: Stringified chat prompt with the generation header
        appended (ready to feed into ``processor(text=...)``).
    """
    examples_str = "\n".join(
        f"  {language}: {src}\n  English: {tgt}"
        for src, tgt in few_shot_examples[:number_of_few_shot_examples]
    )
    user_content = (
        f"You are a professional translator. "
        f"Here are {number_of_few_shot_examples} examples of English transcriptions "
        f"and their {language} translations:\n{examples_str}\n\n"
        f"Following these examples, translate the following English transcription to {language}. "
        f"Only output the {language} translation without any additional text or formatting.\n"
    )
    messages = [{"role": "system", "content": user_content}, {"role": "user", "content": f"{input_text}"},]

    # apply_chat_template with tokenize=False returns a string
    return processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True, 
                enable_thinking=False
            )


def main():
    """Few-shot Gemma 3n English -> LRL text translation over cascaded ASR output.

    Workflow:
        1. Load the ``google/gemma-3n-E2B-it`` processor and model with
           ``device_map="auto"`` (assigned to module-level ``processor``
           and ``model`` so the helpers can reuse them).
        2. For each language in ``language_list``:
            - Resolve the English ASR JSON for that LRL's accent
              (``<accent>_transcriptions.json`` under ``test_base_dir``).
            - Load the few-shot ``(LRL, English)`` pairs via
              :func:`load_few_shot_examples`; warn if fewer than
              ``number_of_few_shot_examples`` are available.
            - For every utterance, build a chat prompt with
              :func:`build_prompt` (system content listing the demos,
              user content with the English ASR transcription),
              tokenise it, and call ``model.generate`` with
              ``max_new_tokens=40``.
            - Decode only the new tokens and append
              ``{"ID", "transcription", "prediction"}`` to ``results``.
        3. Write the predictions to
           ``./RESULTS/naijas2st/gemma3/eng_to_lrl_few_shot/<language>.json``
           as indented UTF-8 JSON.

    Inputs:
        Cascaded ASR JSONs and ``few_shot/`` directories.

    Outputs:
        One predictions JSON per language with the Gemma translations.

    Returns:
        None.
    """
    global processor, model

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
        few_shot_examples = load_few_shot_examples(language)
        if len(few_shot_examples) < number_of_few_shot_examples:
            print(f"  Warning: Only found {len(few_shot_examples)} few-shot examples for {language}.")

        with open(test_set, 'r') as test_set_json:
            for utterance in json.load(test_set_json):
                print(utterance["transcription"][0])
                print(utterance["ID"])
                input_text = utterance["transcription"][0]
                #
                # Process input
                text = build_prompt(language, few_shot_examples, input_text)

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
        output_json = Path(f"./RESULTS/naijas2st/gemma3/eng_to_lrl_few_shot/{language}.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Saved results to {output_json}")


if __name__ == "__main__":
    main()
