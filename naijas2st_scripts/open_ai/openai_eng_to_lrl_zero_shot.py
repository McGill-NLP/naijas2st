"""Zero-shot English audio -> LRL text translation via Azure ``gpt-audio``.

For each language, picks the appropriate English accent test set and
sends each base64-encoded wav to the Azure ``gpt-audio`` deployment
with a system prompt asking for a direct LRL translation. Predictions
are written incrementally to ``RESULTS/.../openai_..._zero_shot/<lang>.json``.
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
    """Read a binary audio file and return its base64-encoded string.

    Args:
        file_path (str | os.PathLike): Path to the audio file.

    Returns:
        str: Base64-encoded ASCII string suitable for the Azure
        ``input_audio`` chat payload.
    """
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_azure(messages):
    """Send a chat-completion request to the Azure deployment.

    Args:
        messages (list[dict]): OpenAI chat messages list (may include
            ``input_audio`` parts).

    Returns:
        str: The assistant message text from the first response
        choice.
    """
    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=500,
        temperature=0
    )
    return resp.choices[0].message.content


def main():
    """Zero-shot English audio -> LRL text translation via Azure ``gpt-audio``.

    Workflow (per target LRL in ``language_list``):
        1. Pick the English accent test directory matching the LRL
           via ``accent_dictionary``; create a temp dir for normalised
           wavs.
        2. Open ``<results_dir>/<language>.json`` as the running output
           buffer.
        3. Walk
           ``english_<accent>/<user>/recordings/*.wav`` and, for each
           wav:
            - Normalise the audio through ``soundfile`` so the WAV
              header is clean.
            - Build a chat ``messages`` payload with a system prompt
              instructing the model to translate the English audio into
              the target LRL plus a user turn carrying the base64-
              encoded WAV.
            - Call :func:`call_azure` with up to 5 retries (sleeping
              2 s between attempts on error).
            - Append ``{"file_name", "prediction"}`` to ``results``;
              in ``finally``, rewrite ``<results_dir>/<language>.json``
              so the run is resumable on crash.

    Inputs:
        ``AZURE_OPENAI_ENDPOINT`` / ``AZURE_OPENAI_API_KEY`` env vars;
        local English test wavs.

    Outputs:
        One ``<language>.json`` per LRL under ``results_dir``.

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

    results_dir = "./RESULTS/naijas2st/openai_gpt_audio_stt_eng_to_lrl_zero_shot"
    os.makedirs(results_dir, exist_ok=True)

    test_base_dir = Path("./naijas2st_scripts/test/")

    # =========================
    # MAIN LOOP
    # =========================

    for language in language_list:
        print(f"\n→ Processing {language}...")

        tmp_dir = tempfile.mkdtemp(prefix=f"local_{language}_eng_")
        accent = accent_dictionary[language]
        test_set = test_base_dir / accent
        # =========================
        # OUTPUT FILE
        # =========================

        out_path = os.path.join(results_dir, f"{language}.json")
        results = []

        # =========================
        # PROCESS TEST SET
        # =========================

        for user in test_set.iterdir():
            user_dir = user / "recordings"

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
                                    f"You are a {language} translation expert. Transcribe the given English audio, and use the transcription to provide its exact {language} translation.\
                                        Return only the {language} translation without any additional text."
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
