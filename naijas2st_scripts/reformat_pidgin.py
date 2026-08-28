"""Attach source and reference Pidgin/English texts to a predictions JSON.

Specialised version of ``reformat_files.py`` for the Pidgin subset:
loads a two-sheet metadata Excel of Pidgin/English pairs, recovers each
prediction's text ID from the audio filename, and writes a single
``pidgin_reformatted.json`` enriched with ``source``/``reference``
according to the chosen direction.
"""

import json
import pandas as pd
from pathlib import Path
import re

def extract_text_id_from_filename(filename):
    """Extract the ``<L>TE_<digits>`` text ID from a Pidgin wav filename.

    Args:
        filename (str): Recording path or filename, e.g.
            ``.../yoruba/Y071c_YTE_0823.wav``.

    Returns:
        str | None: The trailing text ID (e.g. ``"YTE_0823"``), or
        ``None`` if the regex does not match.
    """
    print(f"Extracting text_id from filename: {filename}")
    match = re.search(r'([A-Z]TE_\d+).wav$', filename)
    print(f"Regex match result: {match}")
    if match:
        print(f"Extracted text_id: {match.group(1)} from filename: {filename}")
        return match.group(1)
    return None



def main():
    """Enrich the Pidgin predictions JSON with Pidgin/English source+reference.

    Workflow:
        1. Choose translation ``direction`` (``lrl_to_eng`` or
           ``eng_to_lrl``) and the input/output paths.
        2. Load ``pidgin.json`` (the model-produced predictions) and
           the metadata Excel containing the ``Joel`` and ``Johnson``
           sheets; concatenate both sheets and build
           ``text_id_lookup`` mapping ``source_filename`` to
           ``{"pidgin_text", "english_text"}``.
        3. For every prediction item:
            - Skip items without a ``file_name`` (with a logged message).
            - Recover the LRL text ID from the audio filename via
              :func:`extract_text_id_from_filename`.
            - For ``eng_to_lrl`` mode, rewrite the text ID prefix to
              ``PTE_...`` so it lines up with the Pidgin side of the
              metadata.
            - Attach ``source``/``reference`` from the metadata row
              according to the direction (``english_text`` is the
              source for ``eng_to_lrl``; Pidgin text is the source for
              ``lrl_to_eng``) and record ``source_text_id``.
        4. Write the enriched list to
           ``pidgin_reformatted.json`` and print a sample item.

    Inputs:
        ``pidgin.json`` predictions, ``audio_dataset/metadata_pidgin.xlsx``.

    Outputs:
        ``pidgin_reformatted.json`` ready for SSA-COMET / BLEU evaluation.

    Returns:
        None.
    """
    # Configuration
    direction = "lrl_to_eng"
    gemini_output_dir = Path("./RESULTS/naijas2st/cascaded_tiny_aya_global_yor_last/")
    metadata_input = Path("./audio_dataset/metadata_pidgin.xlsx")
    output_json = Path("./RESULTS/naijas2st/cascaded_tiny_aya_global_yor_last/pidgin_reformatted.json")

    # Load predictions and metadata
    predictions = json.loads(gemini_output_dir.joinpath("pidgin.json").read_text(encoding="utf-8"))
    df = pd.read_excel(metadata_input,  sheet_name=['Joel', 'Johnson'])
    df = pd.concat(df.values(), ignore_index=True)

    # Create lookup dictionaries by text_id
    text_id_lookup = {}
    for _, row in df.iterrows():
        text_id = row['source_filename']
        text_id_lookup[text_id] = {
            'pidgin_text': row['Pidgin'],
            'english_text': row['English']
        }

    # Enrich predictions with reference and source
    for item in predictions:
        if not isinstance(item, dict) or 'file_name' not in item:
            print(f"Skipping item without 'file_name': {item}")
            continue
        
        # Extract source text_id from filename
        source_text_id = extract_text_id_from_filename(item['file_name'])
        print(f"Processing file: {item['file_name']} -> source_text_id: {source_text_id}")
        
        if source_text_id:
            if direction == "eng_to_lrl":
                    source_text_id = f'PTE_' + source_text_id.split('_')[1]
            # Add source text
            if source_text_id in text_id_lookup:
                if direction == "eng_to_lrl":
                    item['source'] = text_id_lookup[source_text_id]['english_text']
                    item['reference'] = text_id_lookup[source_text_id]['pidgin_text']
                if direction == "lrl_to_eng":
                    item['source'] = text_id_lookup[source_text_id]['pidgin_text']
                    item['reference'] = text_id_lookup[source_text_id]['english_text']
                print(f"  Found source text for {source_text_id}")
                item['source_text_id'] = source_text_id
                print(f"  Source text: {item['source']}")
                print(f"  Found reference text for {source_text_id}")
                print(f"  Reference text: {item['reference']}")
            else:
                print(f"  No metadata found for source_text_id: {source_text_id}")

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
