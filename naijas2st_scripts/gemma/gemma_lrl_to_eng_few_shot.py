"""Few-shot LRL -> English text translation with the Gemma 3n instruct model.

Reads ASR transcriptions of the cascaded test set, loads N parallel
``(LRL, English)`` few-shot examples from ``few_shot/<language>``,
constructs a system + user chat prompt, and uses Gemma to translate
each LRL utterance into English. Writes per-language prediction JSONs.
"""

from pathlib import Path
import os
import json
from transformers import AutoProcessor, AutoModelForCausalLM


# MODEL_ID = "google/gemma-4-E4B-it"
MODEL_ID = "google/gemma-3n-E2B-it"

language_list = ["pidgin"]
test_base_dir = Path("./RESULTS/naijas2st/asr_LLM_1B_test_set_for_cascaded")

number_of_few_shot_examples = 5
fewshot_base_dir = Path("few_shot/")

# Module-level model handles populated in main(); helpers below reference them.
processor = None
model = None


def load_few_shot_examples(language: str) -> list:
    """Read ``(LRL, English)`` parallel pairs from ``few_shot/<language>``.

    Args:
        language (str): LRL name; used both to find the directory and
            pick the LRL transcription line in each ``.txt``.

    Returns:
        list[tuple[str, str]]: List of ``(lrl_source, english_target)``
        tuples; files missing either side are skipped.
    """
    few_shot_dir = fewshot_base_dir / language
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
    """Format a Gemma chat prompt with few-shot examples and the LRL input.

    Args:
        language (str): LRL name shown in the prompt.
        few_shot_examples (list[tuple[str, str]]): ``(src, tgt)``
            LRL/English pairs.
        input_text (str): LRL text to translate to English.

    Returns:
        str: Stringified chat prompt with the generation header
        appended.
    """
    examples_str = "\n".join(
        f"  {language}: {src}\n  English: {tgt}"
        for src, tgt in few_shot_examples[:number_of_few_shot_examples]
    )
    user_content = (
        f"You are a professional translator. "
        f"Here are {number_of_few_shot_examples} examples of {language} transcriptions "
        f"and their English translations:\n{examples_str}\n\n"
        f"Following these examples, translate the following {language} transcription to English. "
        f"Only output the English translation without any additional text or formatting.\n"
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
    """Few-shot Gemma 3n LRL -> English text translation over cascaded ASR.

    Workflow:
        1. Load the ``google/gemma-3n-E2B-it`` processor and model with
           ``device_map="auto"`` and store on module-level
           ``processor`` / ``model`` (so prompt-building helpers can
           share them).
        2. For each language in ``language_list``:
            - Load the cascaded LRL ASR JSON
              (``<language>_transcriptions.json``).
            - Load ``few_shot/<language>/`` parallel ``(LRL, English)``
              pairs via :func:`load_few_shot_examples`; warn if fewer
              than ``number_of_few_shot_examples`` are found.
            - For every utterance, build a chat prompt via
              :func:`build_prompt` and call ``model.generate`` with
              ``max_new_tokens=40``.
            - Decode the new tokens and append
              ``{"ID", "transcription", "prediction"}``.
        3. Write per-language predictions to
           ``./RESULTS/naijas2st/gemma3/lrl_to_eng_few_shot/<language>.json``.

    Outputs:
        One predictions JSON per language.

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
        language_format = f"{language}_transcriptions.json"
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

                # # Generate output Germma 4
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
        output_json = Path(f"./RESULTS/naijas2st/gemma3/lrl_to_eng_few_shot/{language}.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Saved results to {output_json}")


if __name__ == "__main__":
    main()
