"""Download a small set of FLEURS English few-shot examples to disk.

Streams the FLEURS English train split, locates a fixed list of sample
IDs, and saves each one as a 16 kHz wav plus a text file with the
English transcription under ``few_shot_data/en_us/``.
"""

import os
import soundfile as sf
from datasets import load_dataset, Audio


def main():
    """Save a fixed list of FLEURS English samples as wav + transcription files.

    Workflow:
        1. Stream ``google/fleurs`` ``en_us`` train split and cast its
           ``audio`` column to 16 kHz.
        2. Walk the stream and build ``id2ex`` containing only the
           examples whose ``id`` is in ``list_of_ids``. Break early
           once all requested IDs have been collected.
        3. Iterate ``list_of_ids`` in fixed order; for each, write
           the audio array to ``<output_dir>/<id>_en_us.wav`` and the
           transcription to ``<output_dir>/<id>.txt`` (single line
           ``English transcription: <text>``).
        4. Print the number of saved few-shot files.

    Note:
        There is an earlier loop in this script that does roughly the
        same job, written defensively in case ``id2ex`` is incomplete;
        the second loop using ``id2ex`` is the canonical path.

    Outputs:
        ``few_shot_data/en_us/<id>_en_us.wav`` and
        ``few_shot_data/en_us/<id>.txt`` for every requested sample ID.

    Returns:
        None.
    """
    ds_en = load_dataset("google/fleurs", "en_us", split="train", streaming=True, trust_remote_code=True)
    ds_en = ds_en.cast_column("audio", Audio(sampling_rate=16_000))


    list_of_ids = [46, 279, 581, 722, 903]

    output_dir = "./few_shot_data/en_us"
    os.makedirs(output_dir, exist_ok=True)
    counter = 0
    # Access the first example

    id2ex = {}
    for ex in ds_en:
        if ex["id"] in list_of_ids:
            id2ex[ex["id"]] = ex
        if len(id2ex) == len(list_of_ids):
            break


    for ex in ds_en:
        for i in list_of_ids:
            print(i)
            if ex['id'] == i:
                arr, sr = ex['audio']['array'], ex['audio']['sampling_rate']
                wav_path = os.path.join(output_dir, f"{ex['id']}_en_us.wav")
                sf.write(wav_path, arr, sr)
                print(f"Processing file {ex['id']}")

                # save transcription
                txt_path = os.path.join(output_dir, f"{ex['id']}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"English transcription: {ex['transcription']}\n")
                counter += 1


    for ex_id in list_of_ids:
        ex = id2ex[ex_id]
        arr, sr = ex['audio']['array'], ex['audio']['sampling_rate']
        wav_path = os.path.join(output_dir, f"{ex_id}_en_us.wav")
        sf.write(wav_path, arr, sr)
        print(f"Processing file {ex_id}")
        # save transcription
        txt_path = os.path.join(output_dir, f"{ex_id}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"English transcription: {ex['transcription']}\n")
        counter += 1

    # Display the content of the example
    print(f"Saved {counter} few-shot examples")
    # Output includes: audio array, path, transcription, lang_id, etc.


if __name__ == "__main__":
    main()
