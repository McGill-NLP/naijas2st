"""SSA-COMET evaluation driver for African/Celtic predictions.

Loads ``McGill-NLP/ssa-comet-mtl-final`` and scores a reformatted
predictions JSON (with ``source``, ``reference`` and ``prediction``
fields, or ``file_name`` IDs that can be parsed). Reports per-sample
scores and a corpus average.
"""

from comet import download_model, load_from_checkpoint
import json
import re
from pathlib import Path


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
    match = re.search(r'([A-Z]TE_\d+)$', filename)
    if match:
        return match.group(1)
    return None


model_path = download_model("McGill-NLP/ssa-comet-mtl-final")
model = load_from_checkpoint(model_path)

def format_data(json_file, language_name):
    """Build the SSA-COMET ``{src, mt, ref}`` payload from a predictions JSON.

    Workflow:
        1. Load the per-language ``*_reformatted.json`` predictions file.
        2. Build a list of valid sample IDs by reading ``id`` first,
           then falling back to extracting from ``file_name`` or
           ``ID`` via :func:`extract_text_id_from_filename`. Skip
           items where no ID can be recovered.
        3. Iterate every item again and, for any with non-empty
           ``prediction`` (or ``translation``), ``reference`` and
           ``source`` strings, append
           ``{"mt": <prediction>, "ref": <reference>, "src": <source>}``
           to the output list.
        4. The result is fed directly to ``model.predict`` (a
           Comet ``MultiTaskLM``-style model loaded at module import).

    Args:
        json_file (str | os.PathLike): Path to a per-language
            ``*_reformatted.json`` with ``source``/``reference``/
            ``prediction`` fields.
        language_name (str): Human-friendly name used in log output.

    Returns:
        list[dict[str, str]]: List of ``{"src", "mt", "ref"}`` dicts
        ready to feed to the COMET model's ``predict`` method.
    """
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data_file = json.load(f)
    
    # Get all sample IDs (handle both 'id' field and extracting from 'file_name')
    sample_ids = []
    for sample in data_file:
        if 'id' in sample:
            sample_ids.append(sample['id'])
        elif 'file_name' in sample:
            extracted_id = extract_text_id_from_filename(sample['file_name'])
            if extracted_id is not None:
                sample_ids.append(extracted_id)
        elif 'ID' in sample:
            extracted_id = extract_text_id_from_filename(sample['ID'])
            if extracted_id is not None:
                sample_ids.append(extracted_id)

    if not sample_ids:
        print(f"Error: No valid sample IDs found for {language_name}")
        return 1.0

    data = []
    for sample in data_file:
        # Get prediction (could be 'prediction' or 'translation')
        mt = sample.get('prediction') or sample.get('translation')
        src = sample.get('source')
        ref = sample.get('reference')
        # Get ID
        sample_id = sample.get('id')
        if sample_id is None and 'file_name' in sample:
            sample_id = extract_text_id_from_filename(sample['file_name'])
        elif sample_id is None and 'ID' in sample:
            sample_id = extract_text_id_from_filename(sample['ID'])
        if sample_id is None:
            continue

        if mt and mt.strip() and ref and ref.strip() and src and src.strip():
            new_data = {'mt': f'{mt.strip()}', 'ref': f'{ref.strip()}', 'src': f'{src.strip()}'}
            data.append(new_data)

    return data



def average_ssa_comet_score(ssa_comet_scores):
    """Compute the mean SSA-COMET score from the model's nested return.

    Args:
        ssa_comet_scores (list): COMET ``predict`` return wrapped in
            an outer list (i.e.
            ``[(per_sentence_scores, system_score)]``).

    Returns:
        float: Mean per-sentence score, ignoring ``None`` entries.
    """
    scores = [score for score in ssa_comet_scores[0][0] if score is not None]
    print(scores)
    average_score = sum(scores) / len(scores)
    return average_score

if __name__ == "__main__":
    language_name = "Naija Pidgin (Nigeria)"
    data = format_data("./RESULTS/naijas2st/cascaded_tiny_aya_global_yor_last/pidgin_reformatted.json", language_name)
    print(data)
    ssa_comet_score = model.predict(data, batch_size=8, gpus=1)
    average_ssa_comet_score = average_ssa_comet_score([ssa_comet_score])
    print(f"{language_name}: SSA-COMET = {ssa_comet_score}")

    print(f"Average SSA-COMET Score: {average_ssa_comet_score}")








