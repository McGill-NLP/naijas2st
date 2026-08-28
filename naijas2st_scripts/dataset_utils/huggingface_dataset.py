"""Report total English audio duration in the African/Celtic HF dataset.

Streams ``McGill-NLP/NaijaS2ST`` and sums the ``duration``
field for samples whose language is ``english``.
"""

from datasets import load_dataset


def main():
    """Stream the African/Celtic HF dataset and print total English duration.

    Workflow:
        1. Load ``McGill-NLP/NaijaS2ST`` in streaming mode
           so the whole archive never has to fit on disk.
        2. Iterate over the ``train`` split, accumulating ``duration``
           seconds for every sample where ``language == "english"``.
           Print a live counter (``\\r``) so progress is visible on a
           long stream.
        3. After iteration, print the total in seconds and in hours.

    Inputs:
        Network access to HuggingFace plus a valid auth token if the
        dataset is gated.

    Outputs:
        Two lines on stdout with the English duration in seconds and
        in hours.

    Returns:
        None.
    """
    ds = load_dataset("McGill-NLP/NaijaS2ST", streaming=True)

    total_english = 0
    for i,sample in enumerate(ds["train"]):
        if sample["language"] == "english":
            duration = sample["duration"]
            total_english += duration
        print(i, end="\r")

    print(f"Total duration of English samples: {total_english} seconds")
    print(f"Total duration of English samples: {total_english / 3600:.2f} hours")


if __name__ == "__main__":
    main()
