"""Attach ``source`` and ``reference`` text to a Gemma/NLLB output JSON.

Reads a per-language predictions JSON (only ``ID``/``prediction``),
recovers the source text ID from the audio filename, looks up the
source text and the corresponding target-language reference text in the
metadata Excel sheet, and writes a ``*_reformatted.json`` for SSA-COMET.
"""

import json
import pandas as pd
from pathlib import Path
import re

def extract_text_id_from_filename(filename):
    """Return the trailing ``<L>TE_<digits>`` text ID from a filename.

    Args:
        filename (str | os.PathLike): Recording path or stem ending
            with ``XTE_<digits>``.

    Returns:
        str | None: The matched text ID (e.g. ``"YTE_0823"``), or
        ``None`` when the regex does not match.
    """
    filename = Path(filename).stem
    print(filename)
    match = re.search(r'([A-Z]TE_\d+)$', filename)
    if match:
        print(match.group(1))
        return match.group(1)
    return None

def get_corresponding_reference_id(source_text_id, language_code):
    """Map an English ``ETE_*`` text ID to its target-language counterpart.

    Args:
        source_text_id (str | None): An English text ID like
            ``"ETE_0823"``.
        language_code (str): Single-letter target language code
            (e.g. ``"I"`` for Igbo).

    Returns:
        str | None: ``"<language_code>TE_<digits>"`` for the target
        language, or ``None`` if input is not an ``ETE_*`` ID.
    """
    if source_text_id and source_text_id.startswith('ETE_'):
        return f'{language_code}TE_' + source_text_id.split('_')[1]
    return None


def main():
    """Attach source/reference text to a Gemma or NLLB output JSON.

    Workflow:
        1. Load the predictions JSON
           (``RESULTS/naijas2st/gemma3/lrl_to_eng_few_shot/yoruba.json``
           in the current configuration) where each item has ``ID``
           and ``prediction`` fields.
        2. Read the recordings metadata Excel sheet into
           ``text_id_lookup`` mapping text ID to
           ``{"text", "language"}``.
        3. For each item:
            - Recover the source text ID from ``ID`` via
              :func:`extract_text_id_from_filename`.
            - Attach ``source`` (text) and ``source_text_id`` from the
              metadata if a match is found.
            - Look up the target-language text ID via
              :func:`get_corresponding_reference_id` (note: the
              currently enabled mapping rewrites ``ETE_*`` to
              ``<lang>TE_*`` so this script is configured for
              English -> LRL evaluation; the ``language_code`` argument
              is the single-letter target code, e.g. ``"I"`` for Igbo).
            - Attach ``reference`` (text) and ``reference_text_id`` when
              a metadata row is found.
        4. Write the enriched list to ``output_json`` and print a
           single sample item for sanity.

    Inputs:
        Predictions JSON; recordings metadata Excel.

    Outputs:
        Reformatted JSON ready for SSA-COMET / BLEU evaluation.

    Returns:
        None.
    """
    # Configuration
    gemini_output_dir = Path("./RESULTS/naijas2st/gemma3/lrl_to_eng_few_shot")
    metadata_input = Path("./naijas2st_scripts/test/recordings_metadata.xlsx")
    output_json = Path("./RESULTS/naijas2st/gemma3/lrl_to_eng_zero_shot/yoruba_reformatted.json")

    # Load predictions and metadata
    predictions = json.loads(gemini_output_dir.joinpath("yoruba.json").read_text(encoding="utf-8"))
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
        if not isinstance(item, dict) or 'ID' not in item:
            continue
        
        # Extract source text_id from filename
        source_text_id = extract_text_id_from_filename(item['ID'])
        print(f"Processing file: {item['ID']} -> source_text_id: {source_text_id}")
        
        if source_text_id:
            # Add source text
            if source_text_id in text_id_lookup:
                item['source'] = text_id_lookup[source_text_id]['text']
                print(f"  Found source text for {source_text_id}")
                item['source_text_id'] = source_text_id
                print(f"  Source text: {item['source']}")
            
            # Get and add reference text (English)
            reference_text_id = get_corresponding_reference_id(source_text_id, 'I')
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
        for key in ['ID', 'source_text_id', 'source', 'reference_text_id', 'reference', 'prediction']:
            if key in sample:
                value = sample[key]
                if isinstance(value, str) and len(value) > 80:
                    print(f"  {key}: {value[:80]}...")
                else:
                    print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
