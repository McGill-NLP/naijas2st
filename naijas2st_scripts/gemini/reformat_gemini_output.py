"""Enrich Gemini LRL->English predictions with source and reference text.

Loads the gemini predictions JSON (with only ``file_name`` and
``prediction`` per item), extracts the original LRL text ID from the
filename, looks up the LRL source and English reference text in the
metadata Excel sheet, and writes a ``*_reformatted.json`` ready for
SSA-COMET evaluation.
"""

import json
import pandas as pd
from pathlib import Path
import re

def extract_text_id_from_filename(filename):
    """Return the trailing ``<L>TE_<digits>`` text ID from a recording path.

    Args:
        filename (str | os.PathLike): A path or filename whose stem
            ends with ``XTE_<digits>``.

    Returns:
        str | None: The matched text ID (e.g. ``"YTE_0823"``) or
        ``None`` when the regex does not match.
    """
    match = re.search(r'([A-Z]TE_\d+)$', filename)
    if match:
        return match.group(1)
    return None

def get_corresponding_reference_id(source_text_id, language_code='Y'):
    """Map an LRL text ID to its English (``ETE_*``) counterpart.

    Args:
        source_text_id (str | None): LRL text ID like ``"YTE_0823"`` /
            ``"HTE_xxxx"`` / ``"ITE_xxxx"``.
        language_code (str): Single-letter language code (kept for
            symmetry with the eng->lrl direction; unused here).

    Returns:
        str | None: The English text ID (``"ETE_<digits>"``), or
        ``None`` if input is not a supported LRL ID.
    """
    if source_text_id and source_text_id.startswith(('YTE_', 'ITE_', 'HTE_')):
        return 'ETE_' + source_text_id.split('_')[1]
    return None


def main():
    """Enrich Gemini Pidgin predictions with source/reference text for SSA-COMET.

    Workflow:
        1. Load ``gemini_output_dir/pidgin.json`` (one item per audio
           with ``file_name`` and ``prediction``).
        2. Read the recordings metadata Excel into a
           ``text_id_lookup`` dict mapping text ID to ``{"text",
           "language"}``.
        3. For every prediction item:
            - Recover the LRL text ID from ``file_name`` via
              :func:`extract_text_id_from_filename`.
            - Attach ``source`` (LRL text) and ``source_text_id`` if a
              metadata match exists.
            - Map the LRL text ID to its English ID with
              :func:`get_corresponding_reference_id` and attach
              ``reference`` / ``reference_text_id`` when found.
        4. Write the enriched list to
           ``pidgin_reformatted.json`` and print a sample item for
           sanity checking.

    Inputs:
        Gemini Pidgin predictions JSON; recordings metadata Excel.

    Outputs:
        ``pidgin_reformatted.json`` next to the input file.

    Returns:
        None.
    """
    # Configuration
    gemini_output_dir = Path("./RESULTS/naijas2st/few_shot_S2T_lrl_to_eng_gemini25/")
    metadata_input = Path("./naijas2st_scripts/recordings_metadata.xlsx")
    output_json = Path("./RESULTS/naijas2st/few_shot_S2T_lrl_to_eng_gemini25/pidgin_reformatted.json")

    # Load predictions and metadata
    predictions = json.loads(gemini_output_dir.joinpath("pidgin.json").read_text(encoding="utf-8"))
    df = pd.read_excel(metadata_input)

    # Create lookup dictionaries by text_id
    text_id_lookup = {}
    for _, row in df.iterrows():
        text_id = row['text_id']
        text_id_lookup[text_id] = {
            'text': row['text'],
            'language': row['language']
        }

    # Enrich predictions with reference and source
    for item in predictions:
        if not isinstance(item, dict) or 'file_name' not in item:
            continue
        
        # Extract source text_id from filename
        source_text_id = extract_text_id_from_filename(item['file_name'])
        print(f"Processing file: {item['file_name']} -> source_text_id: {source_text_id}")
        
        if source_text_id:
            # Add source text
            if source_text_id in text_id_lookup:
                item['source'] = text_id_lookup[source_text_id]['text']
                print(f"  Found source text for {source_text_id}")
                item['source_text_id'] = source_text_id
                print(f"  Source text: {item['source']}")
            
            # Get and add reference text (English)
            reference_text_id = get_corresponding_reference_id(source_text_id, 'P')
            print(f"  Corresponding reference_text_id: {reference_text_id}")
            if reference_text_id and reference_text_id in text_id_lookup:
                item['reference'] = text_id_lookup[reference_text_id]['text']
                print(f"  Found reference text for {reference_text_id}")
                item['reference_text_id'] = reference_text_id
                print(f"  Reference text: {item['reference']}")

    # Write enriched output
    output_json.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote enriched predictions file: {output_json}")
    print(f"Total items processed: {len(predictions)}")

    # Print sample to verify
    if predictions:
        print("\nSample item:")
        sample = predictions[0]
        for key in ['file_name', 'source_text_id', 'source', 'reference_text_id', 'reference', 'prediction']:
            if key in sample:
                value = sample[key]
                if isinstance(value, str) and len(value) > 80:
                    print(f"  {key}: {value[:80]}...")
                else:
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
