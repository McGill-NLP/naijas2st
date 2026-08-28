"""Synthesise English audio from LRL->English text via Azure ``gpt-audio`` TTS.

Reads per-language LRL->English predictions JSONs and asks Azure
``gpt-audio`` to read each English translation aloud in a Northern or
Southern Nigerian English accent. Saves one WAV per item under
``RESULTS/.../sts_openai_lrl_to_eng_few_shot/<language>``.
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
    """Send a chat-completion request to Azure and return the message text.

    Args:
        messages (list[dict]): OpenAI-style chat messages.

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
    """Decode the base64 audio payload of an Azure response and write a WAV.

    Args:
        resp (openai.types.chat.ChatCompletion): Azure chat-completion
            response with an attached ``audio`` field on the first
            choice.
        output_path (str | os.PathLike): Destination ``.wav`` path.

    Returns:
        None.

    Raises:
        ValueError: If the response has no audio payload.
    """
    message = resp.choices[0].message

    if not hasattr(message, "audio") or not message.audio:
        raise ValueError("No audio returned from model")

    audio_bytes = base64.b64decode(message.audio.data)

    with open(output_path, "wb") as f:
        f.write(audio_bytes)


def main():
    """Synthesise English audio in a Nigerian accent from LRL->English text.

    Workflow (per language in ``language_list``):
        1. Resolve ``<input_dir>/<language>.json`` (the cascaded
           LRL->English predictions) and create
           ``<output_dir>/<language>/``.
        2. For every item:
            - Recover a clean wav stem from the audio path.
            - Build a chat ``messages`` payload (system: "you are a
              ``<language>`` speech expert"; user: ``"Say the
              following sentence in a Northern/Southern Nigerian
              English Accent: <english>"``, using
              ``accent_dictionary[language]`` for the accent).
            - Call ``client.chat.completions.create`` with
              ``modalities=["text", "audio"]``, ``audio.format=wav``
              and the ``alloy`` voice (up to 5 retries, 2 s back-off).
            - Decode the base64 audio payload via
              :func:`save_wav_from_response` and write
              ``<results_dir>/<stem>.wav``.

    Inputs:
        ``AZURE_OPENAI_ENDPOINT`` / ``AZURE_OPENAI_API_KEY``;
        LRL->English prediction JSONs.

    Outputs:
        ``<output_dir>/<language>/<stem>.wav`` per item.

    Returns:
        None.
    """
    # =========================
    # CONFIG
    # =========================

    language_list = ["yoruba", "hausa", "igbo"]
    accent_dictionary = {
        "yoruba": "Southern Nigerian English Accent",
        "hausa": "Northern Nigerian English Accent",
        "igbo": "Southern Nigerian English Accent"
    }

    number_of_few_shot_examples = 5
    max_examples = None

    input_dir = Path("./RESULTS/naijas2st/openai_stt_lrl_to_eng_few_shot")
    output_dir = "./RESULTS/naijas2st/sts_openai_lrl_to_eng_few_shot"
    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # MAIN LOOP
    # =========================

    for language in language_list:
        print(f"\n→ Processing {language}...")

        test_json = input_dir / f"{language}.json"

        accent = accent_dictionary[language]
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
                                "text": f"You are a speech expert."
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Say the following sentence in English: {translation}"
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
