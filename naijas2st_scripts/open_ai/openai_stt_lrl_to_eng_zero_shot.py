"""Zero-shot LRL audio -> English text translation via Azure ``gpt-audio``.

For each LRL test wav, base64-encodes the audio and prompts Azure
``gpt-audio`` to transcribe and translate it directly into English.
No in-context demonstrations are used. Predictions are written
incrementally to ``RESULTS/.../openai_..._lrl_to_eng_zero_shot/<lang>.json``.
"""

import os
import json
import time
import base64
import tempfile
import soundfile as sf
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


def main():
    """Zero-shot LRL audio -> English text translation via Azure ``gpt-audio``.

    Workflow (per LRL in ``language_list``):
        1. Resolve ``test/<language>/`` (Pidgin uses
           ``<user>/*.wav``; others use ``<user>/recordings/*.wav``)
           and a per-language temp dir.
        2. For each test wav:
            - Normalise the audio through ``soundfile`` into the temp dir.
            - Build a minimal chat ``messages`` payload: a system
              instruction asking for an English translation, then a
              user turn containing the base64-encoded audio.
            - Call :func:`call_azure` with up to 5 retries (2 s
              back-off).
            - Append ``{"file_name", "prediction"}`` to ``results``;
              in ``finally``, rewrite the per-language JSON after each
              call so the run is resumable.

    Outputs:
        Per-language ``<language>.json`` with model translations.

    Returns:
        None.
    """
    # =========================
    # CONFIG
    # =========================

    language_list = ["pidgin"]

    results_dir = "./RESULTS/naijas2st/openai_gpt_audio_stt_lrl_to_eng_zero_shot"
    os.makedirs(results_dir, exist_ok=True)

    test_base_dir = Path("./naijas2st_scripts/test/")

    # =========================
    # MAIN LOOP
    # =========================

    for language in language_list:
        print(f"\n→ Processing {language}...")

        tmp_dir = tempfile.mkdtemp(prefix=f"local_{language}_")
        test_set = test_base_dir / language

        # =========================
        # OUTPUT FILE
        # =========================

        out_path = os.path.join(results_dir, f"{language}.json")
        results = []

        # =========================
        # PROCESS TEST SET
        # =========================

        for user in test_set.iterdir():
            # user_dir = user / "recordings"
            user_dir = user if language == "pidgin" else user / "recordings"
            for file_path in user_dir.iterdir():
                print(f"  ↳ {file_path.stem}")

                # Normalize audio
                arr, sr = sf.read(file_path)
                tmp_path = os.path.join(tmp_dir, f"test_{file_path.name}")
                sf.write(tmp_path, arr, sr)

                # =========================
                # BUILD MESSAGE PAYLOAD
                # =========================

                messages = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"You are a translation expert. Transcribe the given {language} audio, and use the transcription to provide its exact English translation.\
                                        Return only the English translation without any additional text."
                                )
                            }
                        ]
                    }
                ]

                
                # -------------------------
                # TEST AUDIO
                # -------------------------

                test_audio_b64 = encode_audio_to_base64(tmp_path)

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": test_audio_b64,
                                "format": "wav"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"Translate this {language} audio into English."
                        }
                    ]
                })

                # =========================
                # RETRY LOOP
                # =========================

                success = False
                retries = 0

                while not success:
                    retries += 1
                    print(f"    ↳ Attempt {retries}")

                    if retries > 5:
                        print("    ↳ Failed after retries, skipping")
                        break

                    try:
                        output = call_azure(messages)
                        print(f"response: {output}")
                        results.append({
                            "file_name": f"{user}_{file_path.stem}",
                            "prediction": output.strip(),
                        })

                        success = True

                    except Exception as e:
                        print(f"    ↳ Error: {e}")
                        time.sleep(2)

                    finally:
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"  ↳ saved {len(results)} translations to {out_path}")


if __name__ == "__main__":
    main()
