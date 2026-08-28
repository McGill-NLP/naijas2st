"""Test for the Gemma 4 instruct model.

Loads the model once, sends a tiny "tell me a joke" chat prompt and
prints the parsed response. Useful for verifying environment + GPU
setup before running the full translation pipelines.
"""

from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-E4B-it"


def main():
    """Smoke-test Gemma 4 by generating a joke from a fixed chat prompt.

    Workflow:
        1. Load the ``google/gemma-4-E4B-it`` processor and model with
           ``device_map="auto"`` and auto dtype.
        2. Build a tiny two-turn chat (``system``: "You are a helpful
           assistant.", ``user``: "Tell me a great joke.") and format it
           via ``processor.apply_chat_template`` with the generation
           prompt appended.
        3. Tokenise the chat string, run ``model.generate`` with
           ``max_new_tokens=1024``, decode the new tokens
           (``skip_special_tokens=False`` so the response tags survive),
           and pull the assistant content out via
           ``processor.parse_response``.
        4. Print the resulting joke to stdout.

    Outputs:
        A single joke printed on stdout. Useful for verifying that the
        Gemma 4 model loads, generates and tag-parses correctly.

    Returns:
        None.
    """
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto"
    )

    # Prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a great joke."},
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

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=1024)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    # Parse output
    output = processor.parse_response(response)

    print(output['content'])


if __name__ == "__main__":
    main()
