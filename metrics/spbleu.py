"""Compute chrF, sacreBLEU, spBLEU and chrF++ on a reformatted predictions JSON.

Loads a per-language ``*_reformatted.json`` (with ``reference`` /
``prediction`` fields), normalises sample IDs, and prints chrF,
sacreBLEU, spBLEU, spBLEU-1K plus the local chrF++ implementation
from ``chrf_plus.computeChrF``.
"""

import evaluate
import json
import os
from chrf_plus import computeChrF
import sys
import argparse


metric = evaluate.load("sacrebleu")


def extract_id_from_filename(file_name):
    """Extract the trailing integer sample ID from a filename.

    Args:
        file_name (str): Filename whose stem ends with ``_<int>``
            (e.g. ``"sample_42.wav"``).

    Returns:
        int | None: The extracted ID, or ``None`` if the trailing
        token cannot be parsed as ``int``.
    """
    try:
        # Extract the ID from the file name
        stem_name = os.path.splitext(os.path.basename(file_name))[0]
        return int(stem_name.split('_')[-1])

    except (IndexError, ValueError):
        return None


def format_data(json_file, language_name):
    """Load a predictions JSON and return aligned ``(predictions, references)``.

    Builds the inputs needed by the various BLEU/chrF metrics by
    pairing each prediction string with its ``reference`` string from
    the same record; entries missing either side are skipped.

    Args:
        json_file (str | os.PathLike): Path to the predictions JSON.
        language_name (str): Name of the language being evaluated.

    Returns:
        tuple[list[str], list[str]]: ``(predictions, references)``
        aligned by position.
    """
    with open(json_file, "r") as f:
        data_file = json.load(f)
        sample_ids = []
        for sample in data_file:
            if 'id' in sample:
                sample_ids.append(sample['id'])
            elif 'ID' in sample:
                extracted_id = extract_id_from_filename(sample['ID'])
                sample_ids.append(extracted_id)
            elif 'file_name' in sample:
                extracted_id = extract_id_from_filename(sample['file_name'])
                if extracted_id is not None:
                    sample_ids.append(extracted_id)
    if not sample_ids:
        print(f"Error: No valid sample IDs found for {language_name}")

    
    predictions = []
    references = []
    number_of_samples = 0
    for sample in data_file:
        # Get prediction (could be 'prediction' or 'translation')
        mt = sample.get('prediction') or sample.get('translation') or sample.get('transcription')
        ref = sample.get('reference')
        # Get ID
        sample_id = sample.get('id')
        if sample_id is None and 'file_name' in sample:
            sample_id = extract_id_from_filename(sample['file_name'])
        if sample_id is None and 'ID' in sample:
            sample_id = extract_id_from_filename(sample['ID'])
        if sample_id is None:
            continue
        
        if mt and mt.strip() and ref and ref.strip():
            predictions.append(mt.strip())
            references.append(ref.strip())
            number_of_samples += 1
    
    print(f"Number of valid samples for {language_name}: {number_of_samples}")
    
    return predictions, references



if __name__ == "__main__":
    language_name = "Naija Pidgin (Nigeria)"
    file = "./RESULTS/naijas2st/cascaded_tiny_aya_eng_to_lrl/pidgin_reformatted.json"
    predictions, references = format_data(json_file=file, language_name=language_name)

    chrf = evaluate.load("chrf")
    results_chrf = chrf.compute(predictions=predictions, references=references)
    print(f"Language: {language_name}")
    print(f"Number of samples: {len(predictions)}")
    #Score using the default sacreBLEU tokenizer (mosestokenizer).
    print("chrf score = ", round(results_chrf["score"], 1))
    results = metric.compute(predictions=predictions, references=references)
    print("sacreBLEU score = ", round(results["score"], 1))

    #Score using the default SentencePiece tokenizer (spBLEU).
    results = metric.compute(tokenize="spm", predictions=predictions, references=references)
    print("spBLEU score = ", round(results["score"], 1))

    #Score using the spBLEU-1K SentencePiece tokenizer.
    results = metric.compute(tokenize="spBLEU-1K", predictions=predictions, references=references)
    print("spBLEU-1K score = ", round(results["score"], 1))

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-nc", "--ncorder", help="character n-gram order (default=6)", type=int, default=6)
    argParser.add_argument("-nw", "--nworder", help="word n-gram order (default=2)", type=int, default=2)
    argParser.add_argument("-b", "--beta", help="beta parameter (default=2)", type=float, default=2.0)
    argParser.add_argument("-s", "--sent", help="show sentence level scores", action="store_true")
    args = argParser.parse_args()

    sentence_level_scores = None
    totalF, averageTotalF, totalPrec, totalRec = computeChrF(references, predictions, args.nworder, args.ncorder, args.beta, sentence_level_scores)

    sys.stdout.write("c%i+w%i-F%i\t%.4f\n"  % (args.ncorder, args.nworder, args.beta, 100*totalF))
    sys.stdout.write("c%i+w%i-avgF%i\t%.4f\n"  % (args.ncorder, args.nworder, args.beta, 100*averageTotalF))
