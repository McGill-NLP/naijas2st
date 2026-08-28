"""Synthesise LRL audio from English->LRL text via Azure ``gpt-audio`` TTS.

Reads English->LRL text predictions JSONs and asks Azure ``gpt-audio``
to read each LRL prediction aloud (voice=``alloy``, WAV output), saving
one wav per item under ``RESULTS/naijas2st/sts_openai_..._/<lang>/``.
"""

import os
import json
import time
import base64
from openai import AzureOpenAI
from pathlib import Path

# Set up Azure OpenAI client

endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
model_name = "gpt-audio"
deployment = "gpt-audio"

subscription_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# =========================
# HELPERS
# =========================

def encode_audio_to_base64(file_path):
    """Return a binary audio file as a base64-encoded string.

    Args:
        file_path (str | os.PathLike): Path to the audio file.

    Returns:
        str: Base64-encoded ASCII string.
    """
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_azure(messages):
    """Send a text-only chat-completion request to the Azure deployment.

    Args:
        messages (list[dict]): OpenAI-style messages list.

    Returns:
        str: The first response choice's message content.
    """
    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=500,
        temperature=0
    )
    return resp.choices[0].message.content


def save_wav_from_response(resp, output_path):
    """Decode the base64 audio in an OpenAI response and write it as a WAV.

    Args:
        resp (openai.types.chat.ChatCompletion): Azure chat-completion
            response with an attached ``audio`` field on the first
            choice.
        output_path (str | os.PathLike): Destination ``.wav`` path.

    Returns:
        None.

    Raises:
        ValueError: If the response message has no audio payload.
    """
    message = resp.choices[0].message

    if not hasattr(message, "audio") or not message.audio:
        raise ValueError("No audio returned from model")

    audio_bytes = base64.b64decode(message.audio.data)

    with open(output_path, "wb") as f:
        f.write(audio_bytes)


def main():
    """Synthesise LRL audio from English->LRL text via Azure ``gpt-audio`` TTS.

    Workflow (per language in ``language_list``):
        1. Resolve ``<input_dir>/<language>.json`` (the cascaded
           English->LRL predictions) and create
           ``<output_dir>/<language>/``.
        2. For each item in the predictions JSON:
            - Recover a clean wav stem from the original audio path.
            - Build a chat ``messages`` payload (system: "you are a
              ``<language>`` speech expert"; user: "Say the following
              sentence in ``<language>``: ``<translation>``").
            - Call ``client.chat.completions.create`` with
              ``modalities=["text", "audio"]``, ``audio.format=wav``
              and the ``alloy`` voice, retrying up to 5 times on error
              with a 2 s back-off.
            - Decode the base64 audio in the response via
              :func:`save_wav_from_response` and write
              ``<results_dir>/<stem>.wav``.

    Inputs:
        ``AZURE_OPENAI_ENDPOINT`` / ``AZURE_OPENAI_API_KEY``;
        English->LRL prediction JSONs.

    Outputs:
        ``<output_dir>/<language>/<stem>.wav`` per translated item.

    Returns:
        None.
    """
    # =========================
    # CONFIG
    # =========================

    language_list = ["yoruba", "hausa", "igbo"]
    accent_dictionary = {
        "yoruba": "english_south_accent",
        "hausa": "english_north_accent",
        "igbo": "english_south_accent"
    }

    number_of_few_shot_examples = 5
    max_examples = None

    input_dir = Path("./RESULTS/naijas2st/openai_stt_eng_to_lrl_few_shot")
    output_dir = "./RESULTS/naijas2st/sts_openai_eng_to_lrl_few_shot"
    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # MAIN LOOP
    # =========================

    for language in language_list:
        print(f"\n→ Processing {language}...")

        test_json = input_dir / f"{language}.json"


        print('test json path:', test_json)

        results_dir = os.path.join(output_dir, f"{language}/")
        os.makedirs(results_dir, exist_ok=True)

        counter = 0

        with open(test_json, "r", encoding="utf-8") as fp:
            test_set = json.load(fp)

            for test_item in test_set:
                file_name = test_item['file_name']

                stem_name = file_name.split("/")[3].split(".")[0]
                print(f"→ Processing {stem_name}...")

                translation = test_item['prediction']
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": f"You are a {language} speech expert."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Say the following sentence in {language}: {translation}"
                            }
                        ]
                    }
                ]

                wav_file_path = os.path.join(results_dir, f"{stem_name}.wav")
                success = False
                retries = 0

                while not success:
                    print(f"    ↳ Attempt {retries + 1}")
                    retries += 1

                    if retries > 5:
                        print("  ↳ failed to process, moving on...")
                        break

                    try:
                        resp = client.chat.completions.create(
                            model=deployment,
                            modalities=["text", "audio"],
                            audio={
                                "voice": "alloy",
                                "format": "wav"
                            },
                            messages=messages,
                            temperature=0
                        )

                        save_wav_from_response(resp, wav_file_path)
                        counter += 1
                        success = True

                    except Exception as e:
                        print(f"caught error, retrying: {e}")
                        time.sleep(2)

        print(f"  ↳ saved {counter} for {language} translations to {results_dir}")


if __name__ == "__main__":
    main()
