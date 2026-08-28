"""Few-shot LRL audio -> English text translation via Azure ``gpt-audio``.

Loads few-shot ``(LRL audio, LRL transcription, English translation)``
examples from ``few_shot/<language>``, interleaves them as user/audio
and assistant/translation turns, and asks Azure ``gpt-audio`` to
translate each test LRL audio directly into English. Results are
streamed to the per-language output JSON.
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
    """Few-shot LRL audio -> English text translation via Azure ``gpt-audio``.

    Workflow (per LRL in ``language_list``):
        1. Resolve ``few_shot/<language>`` (parallel wav+txt pairs) and
           ``test/<language>`` plus a per-language temp dir.
        2. Load up to ``number_of_few_shot_examples`` examples into
           ``fewshot_examples`` as ``(wav_path, lrl_text, english_text)``
           triples (skips entries missing either side).
        3. For every wav under ``test/<language>/<user>/`` (or
           ``test/<language>/<user>`` for Pidgin):
            - Normalise the audio via ``soundfile``.
            - Build a chat ``messages`` list with a system prompt
              describing the few-shot setup, then for each demo append
              a ``user`` turn with the demo audio + ``"Example i"`` and
              an ``assistant`` turn with the English translation.
            - Append a final user turn with the base64-encoded test
              audio plus an explicit "translate this ``<language>``
              audio into English" instruction.
            - Call :func:`call_azure` with up to 5 retries (2 s
              back-off); append ``{"file_name", "prediction"}`` and
              rewrite ``<results_dir>/<language>.json`` after each call.

    Outputs:
        Per-language ``<language>.json`` with model translations.

    Returns:
        None.
    """
    # =========================
    # CONFIG
    # =========================

    language_list = ["pidgin"]

    number_of_few_shot_examples = 5
    max_examples = None

    results_dir = "./RESULTS/naijas2st/openai_gpt_audio_stt_lrl_to_eng_few_shot"
    os.makedirs(results_dir, exist_ok=True)

    fewshot_base_dir = Path("./naijas2st_scripts/few_shot/")
    test_base_dir = Path("./naijas2st_scripts/test/")

    # =========================
    # MAIN LOOP
    # =========================

    for language in language_list:
        print(f"\n→ Processing {language}...")

        tmp_dir = tempfile.mkdtemp(prefix=f"local_{language}_")
        fewshot_dir = fewshot_base_dir / language
        test_set = test_base_dir / language

        fewshot_examples = []

        # =========================
        # LOAD FEW-SHOT EXAMPLES
        # =========================

        if fewshot_dir.exists():
            wav_files = sorted([f for f in os.listdir(fewshot_dir) if f.endswith(".wav")])[:number_of_few_shot_examples]

            for wav_file in wav_files:
                wav_path = fewshot_dir / wav_file
                txt_path = fewshot_dir / wav_file.replace(".wav", ".txt")

                if not txt_path.exists():
                    print(f"  Warning: Missing {txt_path}, skipping")
                    continue

                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.read().strip().split("\n")

                original_transcription = ""
                english_translation = ""

                for line in lines:
                    if line.lower().startswith(f"{language} transcription:"):
                        original_transcription = line.split(":", 1)[1].strip()

                    if line.lower().startswith("english transcription:"):
                        english_translation = line.split(":", 1)[1].strip()

                if not original_transcription:
                    print(f"  Warning: Missing source transcription, skipping")
                    continue

                if not english_translation:
                    print(f"  Warning: Missing English translation, skipping")
                    continue

                fewshot_examples.append(
                    (wav_path, original_transcription, english_translation)
                )

            print(f"  ↳ Loaded {len(fewshot_examples)} few-shot examples")

        else:
            print(f"  Warning: Few-shot directory not found → zero-shot mode")

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
            user_dir = user
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
                                    f"You are a translation expert. Here are {len(fewshot_examples)} examples of {language} audio transcribed, then translated into English.\
                                        Following these examples, transcribe the last given audio, and use the transcription to provide its exact English translation.\
                                        Return only the English translation without any additional text."
                                )
                            }
                        ]
                    }
                ]

                # -------------------------
                # FEW-SHOT EXAMPLES
                # -------------------------

                for i, (wav_path, transcription, translation) in enumerate(fewshot_examples, 1):
                    audio_b64 = encode_audio_to_base64(wav_path)

                    # user: audio
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_b64,
                                    "format": "wav"
                                }
                            },
                            {
                                "type": "text",
                                "text": f"Example {i}"
                            }
                        ]
                    })

                    # assistant: translation only
                    messages.append({
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": translation
                            }
                        ]
                    })

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
