"""Copy LRL audio counterparts to a fixed set of English ground-truth wavs.

For each English wav under ``ENGLISH_DIR``, extract its utterance ID
(``..._ETE_<id>``) and copy the matching LRL wav (e.g. ``ITE_<id>``)
from ``MULTILINGUAL_ROOT/<language>`` into ``OUTPUT_ROOT/<language>``.
Used to produce LRL audio bundles aligned with the human-evaluation
English signal set.
"""

import shutil
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

# English wav directory (the files you already sampled/generated)
ENGLISH_DIR = Path("RESULTS_DIR")

# Root directory containing ALL language folders
# Example:
# multilingual_root/
#   english/
#   igbo/
#   yoruba/
#   hausa/
MULTILINGUAL_ROOT = Path("RESULTS_DIR")

# Output directory
OUTPUT_ROOT = Path("RESULTS_DIR")

# Language configuration
LANGUAGES = {
    "igbo": {
        "prefix": "ITE",
        "speaker_prefix": "I"
    },
    # "yoruba": {
    #     "prefix": "YTE",
    #     "speaker_prefix": "Y"
    # },
    # "hausa": {
    #     "prefix": "HTE",
    #     "speaker_prefix": "H"
    # }
}


def main():
    """Copy LRL wavs matching the fixed English signal set into ``OUTPUT_ROOT``.

    Workflow:
        1. Make sure ``OUTPUT_ROOT`` exists and create a subdirectory
           per language defined in ``LANGUAGES``.
        2. For each language, walk ``MULTILINGUAL_ROOT/<lang>/**/*.wav``
           and build ``index = {utterance_id: wav_path}`` by splitting
           the wav stem on ``_`` (expected format
           ``<speaker>_<lang_prefix>_<utterance_id>``); print the size of
           each language index.
        3. List every wav under ``ENGLISH_DIR`` (the chosen English
           ground-truth set) and, for each one:
            - Parse out the utterance ID from the wav stem.
            - For every configured LRL, look up that utterance ID in
              the language's index and ``shutil.copy2`` the matching
              wav into ``OUTPUT_ROOT/<lang>/<source.name>``.
            - Record any (language, utterance) missing from the index
              for the final report.
        4. After the copy loop, print a summary of how many matches
           were missing and the first 20 (language, utterance) pairs.

    Inputs:
        - English wavs under ``ENGLISH_DIR``.
        - Multilingual prediction wavs under ``MULTILINGUAL_ROOT/<lang>``.

    Outputs:
        Copied wavs under ``OUTPUT_ROOT/<lang>`` and a stdout report
        of missing matches.

    Returns:
        None.
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for lang in LANGUAGES:
        (OUTPUT_ROOT / lang).mkdir(exist_ok=True)

    # =========================================================
    # INDEX ALL LANGUAGE FILES
    # =========================================================

    print("Indexing language files...")

    language_indices = {}

    for lang, config in LANGUAGES.items():

        lang_dir = MULTILINGUAL_ROOT / lang

        index = {}

        for wav_path in lang_dir.glob("**/*.wav"):

            stem = wav_path.stem

            # Expected:
            # Y0005_YTE_0251
            parts = stem.split("_")

            if len(parts) < 3:
                continue

            speaker_id = parts[0]
            lang_prefix = parts[1]
            utterance_id = parts[2]

            index[utterance_id] = wav_path

        language_indices[lang] = index

        print(f"{lang}: indexed {len(index)} files")

    # =========================================================
    # EXTRACT MATCHING FILES
    # =========================================================

    english_files = list(ENGLISH_DIR.glob("*.wav"))

    print(f"\nFound {len(english_files)} English files")

    missing = []

    for eng_file in english_files:

        # Example:
        # EB0003_ETE_0251.wav
        stem = eng_file.stem
        parts = stem.split("_")

        if len(parts) < 3:
            print(f"Skipping malformed filename: {eng_file.name}")
            continue

        utterance_id = parts[2]

        # -----------------------------------------------------
        # Find corresponding files in each language
        # -----------------------------------------------------

        for lang, config in LANGUAGES.items():

            lang_index = language_indices[lang]

            if utterance_id not in lang_index:
                missing.append((lang, utterance_id))
                print(f"Missing {lang} file for utterance {utterance_id}")
                continue

            source_file = lang_index[utterance_id]

            dest_file = (
                OUTPUT_ROOT
                / lang
                / source_file.name
            )
            print(dest_file)

            shutil.copy2(source_file, dest_file)

    print("\nDone.")

    # =========================================================
    # REPORT MISSING FILES
    # =========================================================

    if missing:

        print(f"\nMissing matches: {len(missing)}")

        for lang, utt in missing[:20]:
            print(f"{lang}: {utt}")


if __name__ == "__main__":
    main()
