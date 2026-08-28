"""Batched few-shot cascaded LRL->English translation with CohereLabs Tiny-Aya.

For each LRL, reads the cascaded ASR transcription JSON, builds a
single chat prompt per utterance with N few-shot examples loaded from
``few_shot/<language>``, then runs Tiny-Aya in batches for efficiency
and saves the English predictions to a per-language JSON.
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
    """Format a single Tiny-Aya chat prompt with few-shot demonstrations.

    Args:
        language (str): LRL name shown in the prompt.
        few_shot_examples (list[tuple[str, str]]): ``(src, tgt)``
            LRL/English pairs.
        input_text (str): LRL utterance to translate.
        tokenizer (transformers.PreTrainedTokenizer): Tiny-Aya
            tokenizer (used for ``apply_chat_template``).
        number_of_few_shot_examples (int): How many examples to inline.

    Returns:
        str: Stringified chat prompt ready for
        ``tokenizer.__call__``.
    """
    examples_str = "\n".join(
        f"  {language}: {src}\n  English: {tgt}"
        for src, tgt in few_shot_examples[:number_of_few_shot_examples]
    )
    content = (
        f"You are a professional translator. "
        f"Here are {number_of_few_shot_examples} examples of {language} transcriptions "
        f"and their English translations:\n{examples_str}\n\n"
        f"Following these examples, translate the following {language} transcription to English. "
        f"Only output the English translation without any additional text or formatting.\n"
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
        language (str): LRL name (used both for directory and to pick
            the LRL transcription line).
        fewshot_base_dir (pathlib.Path): Root containing per-language
            few-shot folders.

    Returns:
        list[tuple[str, str]]: ``(lrl_source, english_target)``
        string tuples; files missing either side are skipped.
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
    """Batched few-shot Tiny-Aya LRL -> English translation over cascaded ASR.

    Workflow:
        1. Choose device (``cuda`` if available, otherwise ``cpu``),
           authenticate with HuggingFace via ``HF_TOKEN`` if set, and
           load the ``CohereLabs/tiny-aya-global`` tokenizer + model.
           The tokenizer's ``padding_side`` is set to ``left`` so
           batched decoder-only generation works correctly.
        2. For each language in ``language_list``:
            - Load ``few_shot/<language>/`` parallel pairs via
              :func:`load_few_shot_examples`.
            - Load the cascaded ASR JSON
              ``<language>_transcriptions.json``.
            - Iterate the utterances in ``BATCH_SIZE`` chunks. Build a
              chat prompt per utterance with :func:`build_prompt`
              (system+user content with the demos and the test
              transcription), then tokenise the whole batch with
              ``padding=True``.
            - Call ``model.generate`` with
              ``max_new_tokens=MAX_NEW_TOKENS=128``, ``do_sample=True``,
              ``temperature=0.1`` and ``top_p=0.95`` (low-temperature
              sampling).
            - Slice off the prompt tokens and batch-decode only the
              new tokens; append ``{"file_name": uid, "prediction":
              translation}`` per item.
        3. Write the per-language predictions to
           ``<results_dir>/<language>.json``.
        4. Print a final completion message.

    Outputs:
        One predictions JSON per language.

    Returns:
        None.
    """
    language_list = ["pidgin"]
    number_of_few_shot_examples = 5
    results_dir = "./RESULTS/naijas2st/cascaded_tiny_aya_global_yor_last"
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

        few_shot_examples = load_few_shot_examples(language, fewshot_base_dir)
        if len(few_shot_examples) < number_of_few_shot_examples:
            print(f"  Warning: Only found {len(few_shot_examples)} few-shot examples for {language}.")

        test_set_path = test_base_dir / f"{language}_transcriptions.json"
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
