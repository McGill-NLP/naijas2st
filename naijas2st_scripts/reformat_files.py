"""Attach source and reference texts to LRL->English prediction JSONs.

Reads per-language prediction JSON files (where each item already has an
``ID`` pointing to an audio filename), recovers the text ID from the
filename, looks up the original LRL text and the matching English
reference in the metadata Excel sheet, and writes
``<language>_reformatted.json`` files with ``source`` and ``reference``
fields populated.
"""

import json
import pandas as pd
from pathlib import Path
import re


def extract_text_id_from_filename(filename):
    """Extract the ``<L>TE_<digits>`` text ID from a recording filename.

    Args:
        filename (str | os.PathLike): Recording filename, e.g.
            ``Y071c_YTE_0823.wav``.

    Returns:
        str | None: The trailing text ID (e.g. ``"YTE_0823"``), or
        ``None`` if no match is found.
    """
    filename = Path(filename).stem

    match = re.search(r'([A-Z]TE_\d+)$', filename)
    if match:
        return match.group(1)

    return None


def get_corresponding_reference_id(source_text_id, language_code):
    """Map an LRL text ID to its English counterpart.

    Args:
        source_text_id (str | None): An LRL text ID like ``"YTE_0823"``
            / ``"HTE_xxxx"`` / ``"ITE_xxxx"``.
        language_code (str): Single-letter language code (currently
            unused, kept for symmetry with the eng->lrl direction).

    Returns:
        str | None: The English text ID (``"ETE_<digits>"``), or
        ``None`` if the input does not look like a supported LRL text
        ID.
    """
    if source_text_id and source_text_id.startswith(('YTE_', 'ITE_', 'HTE_')):
        return 'ETE_' + source_text_id.split('_')[1]
    return None


def main():
    """Attach source/reference text to every prediction JSON for SSA-COMET.

    Workflow:
        1. Read the metadata Excel sheet
           (``recordings_metadata.xlsx``) into a DataFrame and build a
           ``text_id_lookup`` dict mapping text ID to
           ``{"text", "language"}``.
        2. Glob ``output_dir/*.json`` and process every file whose stem
           (lower-cased) appears in ``language_code_map``; others are
           skipped with a printed message.
        3. For each prediction file:
            - Load the JSON list of items with ``ID`` audio paths and
              model-produced ``prediction`` fields.
            - For every item, extract the LRL text ID from the audio
              filename via :func:`extract_text_id_from_filename` and
              attach ``source`` (LRL text) and ``source_text_id`` if a
              metadata match is found.
            - Map the LRL text ID to its English counterpart with
              :func:`get_corresponding_reference_id` and attach
              ``reference`` / ``reference_text_id`` when available.
            - Track how many items were processed for reporting.
        4. Write the enriched list to
           ``<output_dir>/<language>_reformatted.json``, print a sample
           item, and continue with the next file.

    Inputs:
        - ``./RESULTS/naijas2st/.../*.json`` prediction files
          (each item must have an ``ID`` field).
        - ``./naijas2st_scripts/test/recordings_metadata.xlsx``.

    Outputs:
        ``<language>_reformatted.json`` next to every input JSON, plus
        per-language sample previews on stdout.

    Returns:
        None.
    """
    # =========================
    # Configuration
    # =========================

    # Directory containing all language JSON files
    output_dir = Path("./RESULTS/naijas2st/gemma4/lrl_to_eng_zero_shot")
    # Metadata Excel file
    metadata_input = Path("./naijas2st_scripts/test/recordings_metadata.xlsx")


    language_code_map = {
        "yoruba": "I",
        "hausa": "H",
        "igbo": "I",
        "pidgin": "P",
    }
    df = pd.read_excel(metadata_input)

    text_id_lookup = {}
    for _, row in df.iterrows():
        text_id_lookup[row["text_id"]] = {
            "text": row["text"],
            "language": row["language"],}

    print(f"Loaded {len(text_id_lookup)} metadata entries")

    json_files = list(output_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        language_name = json_file.stem.lower()
        if language_name not in language_code_map:
            print(f"\nSkipping {json_file.name} (no language code mapping)")
            continue

        language_code = language_code_map[language_name]

        print(f"\n=========================")
        print(f"Processing: {json_file.name}")
        print(f"Language code: {language_code}")
        print(f"=========================")

        # Load predictions
        predictions = json.loads(json_file.read_text(encoding="utf-8"))
        processed_count = 0

        # Enrich predictions
        for item in predictions:
            if not isinstance(item, dict) or "ID" not in item:
                continue

            source_text_id = extract_text_id_from_filename(item["ID"])
            print(f"\nProcessing file: {item['ID']}")
            print(f"Source text_id: {source_text_id}")

            if not source_text_id:
                continue
            if source_text_id in text_id_lookup:
                item["source"] = text_id_lookup[source_text_id]["text"]
                item["source_text_id"] = source_text_id
                print(f"Found source text")

            reference_text_id = get_corresponding_reference_id(source_text_id, language_code)

            print(f"Reference text_id: {reference_text_id}")

            if (reference_text_id and reference_text_id in text_id_lookup):
                item["reference"] = text_id_lookup[reference_text_id]["text"]
                item["reference_text_id"] = reference_text_id

                print(f"Found reference text")

            processed_count += 1

        output_json = output_dir / f"{language_name}_reformatted.json"

        output_json.write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"\nWrote: {output_json}")
        print(f"Processed items: {processed_count}")

        if predictions:
            print("\nSample item:")
            sample = predictions[0]
            for key in [
                "ID",
                "source_text_id",
                "source",
                "reference_text_id",
                "reference",
                "prediction",
            ]:
                if key in sample:
                    value = sample[key]
                    if isinstance(value, str) and len(value) > 80:
                        print(f"  {key}: {value[:80]}...")
                    else:
                        print(f"  {key}: {value}")

    print("\nDone processing all language files.")


if __name__ == "__main__":
    main()
