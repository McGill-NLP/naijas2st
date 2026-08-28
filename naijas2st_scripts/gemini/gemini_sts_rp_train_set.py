"""Generate a synthetic British-accented English TTS training set.

Builds 250-utterance "speaker blocks" from the metadata sheet
(``ETR``/``EMD``/``ENX`` text IDs, ``ETR``/``EMD`` duplicated so each
appears twice), assigns a different prebuilt Gemini voice per block,
and synthesises each text in a standard RP British accent. Writes one
WAV per ``<speaker>_<text_id>`` and a metadata Excel listing every
generated utterance.
"""

from google import genai
from google.genai import types
import os
import wave
import pandas as pd
from collections import deque


def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Write raw PCM bytes to disk as a WAV file.

    Args:
        filename (str | os.PathLike): Destination ``.wav`` path.
        pcm (bytes): Raw PCM bytes returned by the TTS API.
        channels (int): Number of audio channels.
        rate (int): Sample rate in Hz.
        sample_width (int): Sample width in bytes.

    Returns:
        None.
    """
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def main():
    """Generate a synthetic RP British TTS training corpus from metadata.

    Workflow:
        1. Instantiate the Gemini client and ensure
           ``./british_train_set/`` exists.
        2. Read ``recordings_metadata.xlsx`` keeping only
           ``user_id``/``language``/``text_id``/``text`` and filter to
           text IDs starting with ``ETR``, ``EMD`` or ``ENX``. Drop
           duplicate ``text_id`` rows (some are recorded multiple times).
        3. Split rows into three deques (seeded shuffle, ``random_state=42``):
            - ``special_q``: ``ETR``/``EMD`` rows (each will be spoken twice).
            - ``enx_q``: ``ENX`` rows (spoken once).
            - ``normal_q``: anything else (fallback).
           A ``duplicate_later`` deque schedules the second occurrence
           of each ``ETR``/``EMD`` entry within the same speaker block.
        4. Iteratively build ``BLOCK_SIZE=250``-entry "speaker blocks"
           by drawing from the queues in priority order
           (duplicate -> special -> enx -> normal) until everything is
           consumed. Each speaker gets a unique synthetic ID
           ``EB00{block_idx:02d}`` and a different prebuilt voice
           cycled from ``voices``.
        5. For every entry in every block, prompt
           ``gemini-2.5-flash-preview-tts`` with
           ``"Say the following in a standard RP British accent: <text>"``
           (up to 5 retries) and write
           ``<base_results_dir>/<speaker>_<text_id>.wav`` plus a row
           in ``metadata_rows``.
        6. Write ``metadata_rows`` as ``british_train_set/metadata.xlsx``
           with columns ``user_id``, ``language`` (``en-GB``),
           ``text_id``, ``text``.

    Inputs:
        ``GOOGLE_API_KEY``; ``recordings_metadata.xlsx``.

    Outputs:
        Synthetic WAV corpus under ``./british_train_set/`` and a
        matching metadata Excel.

    Returns:
        None.
    """
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    client = genai.Client(api_key=GOOGLE_API_KEY)

    base_results_dir = "./british_train_set/"
    new_metadata_file = "./british_train_set/metadata.xlsx"
    os.makedirs(base_results_dir, exist_ok=True)

    LANGUAGE = "en-GB"

    # --- PATHS ---
    metadata_file = "./naijas2st_scripts/recordings_metadata.xlsx"

    # --- LOAD DATA ---
    df = pd.read_excel(metadata_file, usecols=['user_id', 'language', 'text_id', 'text'])

    # Keep only relevant prefixes
    df = df[df['text_id'].str.startswith(('ETR', 'EMD', 'ENX'))]

    # Drop duplicate text_ids (multiple recordings → one text)
    df = df.drop_duplicates(subset=['text_id']).reset_index(drop=True)

    print(f"Total unique samples: {len(df)}")

    # --- SPLIT DATA ---
    is_special = df['text_id'].str.startswith(('ETR', 'EMD'))
    is_enx = df['text_id'].str.startswith('ENX')

    special_df = df[is_special]          # will be duplicated
    enx_df = df[is_enx]                  # single only
    normal_df = df[~is_special & ~is_enx]

    # --- SHUFFLE ---
    special_q = deque(special_df.sample(frac=1, random_state=42).to_dict('records'))
    enx_q = deque(enx_df.sample(frac=1, random_state=42).to_dict('records'))
    normal_q = deque(normal_df.sample(frac=1, random_state=42).to_dict('records'))

    duplicate_later = deque()  # only for ETR / EMD

    # --- BUILD BLOCKS ---
    BLOCK_SIZE = 250
    blocks = []
    metadata_rows = []
    while special_q or enx_q or normal_q or duplicate_later:

        block = []

        while len(block) < BLOCK_SIZE:

            # 1. second occurrences (ETR/EMD)
            if duplicate_later:
                block.append(duplicate_later.popleft())
                continue

            # 2. first occurrences of ETR/EMD
            if special_q:
                item = special_q.popleft()
                block.append(item)
                duplicate_later.append(item)  # schedule second copy later
                continue

            # 3. ENX (no duplication)
            if enx_q:
                block.append(enx_q.popleft())
                continue

            # 4. fallback normal
            if normal_q:
                block.append(normal_q.popleft())
                continue

            break

        if block:
            blocks.append(block)

    print(f"Built {len(blocks)} speaker blocks")

    # --- VOICES ---
    voices = [
        'Kore', 'Orus', 'Alnilam', 'Vindemiatrix', 'Iapetus',
        'Charon', 'Rasalgethi', 'Achernar', 'Zephyr',
        'Erinome', 'Sadaltager'
    ]

    # --- TTS GENERATION ---
    for block_idx, block in enumerate(blocks):

        voice = voices[block_idx % len(voices)]
        speaker_name = f"EB00{block_idx:02d}"

        print(f"\n--- Block {block_idx} | Speaker {speaker_name} | Voice {voice} ---")

        for row in block:
            text_id = row['text_id']
            text = row['text']

            prompt_parts = [f"Say the following in a standard RP British accent: {text}"]

            success = False
            retries = 0

            while not success:
                retries += 1
                print(f"Processing {text_id} (attempt {retries})")

                if retries > 5:
                    print("  ↳ failed, skipping")
                    break

                try:
                   resp = client.models.generate_content(
                                    model="gemini-2.5-flash-preview-tts",
                                    contents=prompt_parts,
                                    config=types.GenerateContentConfig(
                                        response_modalities=["AUDIO"],
                                        speech_config=types.SpeechConfig(
                                            voice_config=types.VoiceConfig(
                                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                                voice_name=voice,
                                                )
                                            )
                                        ),
                                    )
                                    )

                   data = resp.candidates[0].content.parts[0].inline_data.data

                   wav_file = f"{base_results_dir}{speaker_name}_{text_id}.wav"
                   wave_file(wav_file, data)

                   # --- ADD METADATA ROW ---
                   metadata_rows.append({
                      "user_id": speaker_name,
                      "language": LANGUAGE,
                      "text_id": text_id,
                      "text": text
                   })

                   success = True

                except Exception as e:
                    print(f"  ↳ error: {e}")


    metadata_df = pd.DataFrame(metadata_rows)
    # Optional: enforce column order
    metadata_df = metadata_df[["user_id", "language", "text_id", "text"]]

    metadata_df.to_excel(new_metadata_file, index=False)

    print(f"Saved metadata to {new_metadata_file}")


if __name__ == "__main__":
    main()
