"""Batched few-shot cascaded English->LRL translation with CohereLabs Tiny-Aya.

Counterpart of ``tiny_aya_lrl_to_eng.py`` for the English -> LRL direction.
Builds chat prompts with N parallel demonstrations and runs Tiny-Aya in
batches to translate each English ASR transcription into the target LRL.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import torch
from pathlib import Path
from huggingface_hub import login


def build_prompt(language: str, 
                few_shot_examples: list, 
                input_text: str,
                tokenizer,
                number_of_few_shot_examples: int) -> str:
    """Format a single Tiny-Aya chat prompt for English -> LRL with examples.

    Args:
        language (str): Target LRL name shown in the prompt.
        few_shot_examples (list[tuple[str, str]]): ``(src, tgt)``
            pairs from the few-shot directory.
        input_text (str): English text to translate.
        tokenizer (transformers.PreTrainedTokenizer): Tiny-Aya
            tokenizer for chat-template formatting.
        number_of_few_shot_examples (int): How many examples to inline.

    Returns:
        str: Stringified chat prompt ready for the tokenizer.
    """
    examples_str = "\n".join(
        f"  English: {tgt}\n  {language}: {tgt}"
        for src, tgt in few_shot_examples[:number_of_few_shot_examples]
    )
    content = (
        f"You are a professional translator. "
        f"Here are {number_of_few_shot_examples} examples of English transcriptions "
        f"and their {language} translations:\n{examples_str}\n\n"
        f"Following these examples, translate the following English transcription to {language}. "
        f"Only output the {language} translation without any additional text or formatting.\n"
        f"{input_text}"
    )
    messages = [{"role": "user", "content": content}]
    # apply_chat_template with tokenize=False returns a string
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_translation(gen_text: str) -> str:
    """Strip Tiny-Aya control tokens from a generated string.

    Args:
        gen_text (str): Raw decoded model output.

    Returns:
        str: The text between ``<|START_RESPONSE|>`` and
        ``<|END_RESPONSE|>``, or the suffix after ``<|CHATBOT_TOKEN|>``
        as a fallback.
    """
    if "<|START_RESPONSE|>" in gen_text and "<|END_RESPONSE|>" in gen_text:
        return gen_text.split("<|START_RESPONSE|>")[1].split("<|END_RESPONSE|>")[0].strip()
    print("  Warning: Response tags not found, using fallback extraction.")
    return gen_text.split("<|CHATBOT_TOKEN|>")[-1].strip()


def load_few_shot_examples(language: str, fewshot_base_dir: Path) -> list:
    """Read ``(LRL, English)`` parallel pairs from ``few_shot/<language>``.

    Args:
        language (str): LRL name; selects the directory and the LRL
            transcription line.
        fewshot_base_dir (pathlib.Path): Root containing per-language
            few-shot folders.

    Returns:
        list[tuple[str, str]]: ``(lrl_source, english_target)`` pairs;
        files missing either side are skipped.
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


def main():
    """Batched few-shot Tiny-Aya English -> LRL translation over cascaded ASR.

    Workflow:
        1. Choose device, authenticate with HuggingFace if ``HF_TOKEN``
           is set, and load ``CohereLabs/tiny-aya-global`` with
           ``padding_side="left"`` (required for batched decoder-only
           generation).
        2. For each LRL in ``language_list``:
            - Map the LRL to its English accent via
              ``accent_directory`` and load
              ``<accent>_transcriptions.json``.
            - Load ``(LRL, English)`` few-shot pairs from
              ``few_shot/<language>/`` (note: the build_prompt body
              here intentionally reformats pairs as
              ``English -> language`` so demos line up with the
              English-input/LRL-output task).
            - For each ``BATCH_SIZE`` chunk of utterances, build chat
              prompts with :func:`build_prompt`, tokenise the batch
              with padding, and call ``model.generate`` with
              ``max_new_tokens=128``, ``temperature=0.1``,
              ``top_p=0.95``.
            - Slice off the prompt tokens and batch-decode the rest;
              append ``{"file_name": uid, "prediction": translation}``.
        3. Write the per-language predictions to
           ``<results_dir>/<language>.json``.

    Outputs:
        One predictions JSON per language under
        ``./RESULTS/naijas2st/cascaded_tiny_aya_eng_to_lrl``.

    Returns:
        None.
    """
    language_list = ["pidgin"]
    accent_directory = {
        # "yoruba": "english_south_accent",
        # "hausa": "english_north_accent",
        # "igbo": "english_south_accent", 
        "pidgin": "english_south_accent"
    }
    number_of_few_shot_examples = 5
    results_dir = "./RESULTS/naijas2st/cascaded_tiny_aya_eng_to_lrl"
    fewshot_base_dir = Path("few_shot/")
    test_base_dir = Path("./RESULTS/naijas2st/asr_LLM_1B_test_set_for_cascaded/")
    BATCH_SIZE = 8       # Tune this down if you get OOM errors
    MAX_NEW_TOKENS = 128 # Translations rarely exceed this

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Device setup ────────────────────────────────────────────────────────────
    print(f"Using device: {device}")

    # ── Auth & model loading ─────────────────────────────────────────────────────
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        print("Warning: No HF_TOKEN found. You may hit rate limits.")

    model_id = "CohereLabs/tiny-aya-global"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"  # Required for decoder-only batch generation

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",  # Automatically handles multi-GPU or CPU fallback
    )
    model.eval()

    os.makedirs(results_dir, exist_ok=True)

    # ── Main loop ────────────────────────────────────────────────────────────────
    for language in language_list:
        print(f"\n→ Processing {language}...")

        few_shot_examples = load_few_shot_examples(accent_directory[language], fewshot_base_dir)
        if len(few_shot_examples) < number_of_few_shot_examples:
            print(f"  Warning: Only found {len(few_shot_examples)} few-shot examples for {accent_directory[language]}.")

        test_set_path = test_base_dir / f"{accent_directory[language]}_transcriptions.json"
        with open(test_set_path, 'r') as f:
            utterances = json.load(f)

        results = []

        # Process in batches
        for batch_start in range(0, len(utterances), BATCH_SIZE):
            batch = utterances[batch_start: batch_start + BATCH_SIZE]

            ids     = [u["ID"] for u in batch]
            inputs  = [u["transcription"][0] for u in batch]
            prompts = [build_prompt(language, few_shot_examples, inp, tokenizer, number_of_few_shot_examples)\
                        for inp in inputs]

            print(f"  Batch {batch_start // BATCH_SIZE + 1} "
                  f"({batch_start + 1}–{batch_start + len(batch)} / {len(utterances)})")

            # Tokenize whole batch at once
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.inference_mode():
                gen_tokens = model.generate(
                    **encoded,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,  # Suppress padding warning
                )

            # Decode only the newly generated tokens (skip the prompt)
            prompt_len = encoded["input_ids"].shape[1]
            new_tokens = gen_tokens[:, prompt_len:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for uid, translation in zip(ids, decoded):
                translation = translation.strip()
                print(f"    [{uid}] {translation[:80]}{'...' if len(translation) > 80 else ''}")
                results.append({"file_name": uid, "prediction": translation})

        # Save after each language is fully done
        output_json = Path(f"{results_dir}/{language}.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"  ✓ Saved {len(results)} results to {output_json}")

    print("\n✓ All languages processed.")


if __name__ == "__main__":
    main()
