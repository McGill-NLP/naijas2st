"""BLEU / chrF / WER evaluation against FLEURS English references.

For each language results JSON, looks up the corresponding FLEURS
``en_us`` reference per sample ID and computes corpus BLEU, chrF and
WER with sacrebleu / evaluate. Prints per-language and average tables
and saves the aggregate as a JSON.
"""

import json
import sacrebleu
import evaluate
from pathlib import Path
from datasets import load_dataset

def load_english_references(sample_ids):
    """Load English FLEURS references for a list of sample IDs.

    Args:
        sample_ids (list[int]): Sample IDs to look up.

    Returns:
        dict[int, str]: Mapping from each sample ID to its English
        transcription string.
    """
    print("Loading English references from FLEURS...")
    ds_en = load_dataset("google/fleurs", "en_us", split="test", streaming=True)
    
    english_refs = {}
    for sample in ds_en:
        if sample['id'] in sample_ids:
            english_refs[sample['id']] = sample['transcription']
        if len(english_refs) == len(sample_ids):
            break
    
    print(f"  Loaded {len(english_refs)} English references")
    return english_refs


def extract_id_from_filename(file_name):
    """Extract the numeric sample ID from a filename.

    Handles formats like ``"1758.wav"`` or
    ``"10045899353945045473.wav"``.

    Args:
        file_name (str): Filename whose stem is the integer ID.

    Returns:
        int | None: The extracted ID, or ``None`` if conversion to
        ``int`` fails.
    """
    try:
        # Remove .wav extension and convert to int
        return int(file_name.replace('.wav', ''))
    except (ValueError, AttributeError):
        return None


def compute_bleu_for_gemini(json_file, language_name):
    """Compute corpus BLEU for Gemini translations vs. FLEURS English refs.

    Args:
        json_file (str | os.PathLike): Path to the JSON file with
            ``prediction`` (or ``translation``) and either ``id`` or
            ``file_name`` per item.
        language_name (str): Name of the language for log output.

    Returns:
        float: Corpus BLEU score in the ``0-100`` range from
        ``sacrebleu.corpus_bleu``. Returns ``0.0`` if no aligned pairs
        were found.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get all sample IDs (handle both 'id' field and extracting from 'file_name')
    sample_ids = []
    for sample in data:
        if 'id' in sample:
            sample_ids.append(sample['id'])
        elif 'file_name' in sample:
            extracted_id = extract_id_from_filename(sample['file_name'])
            if extracted_id is not None:
                sample_ids.append(extracted_id)
            else:
                print(f"Warning: Could not extract ID from filename: {sample.get('file_name')}")
        else:
            print(f"Warning: Sample missing both 'id' and 'file_name' fields")
    
    if not sample_ids:
        print(f"Error: No valid sample IDs found for {language_name}")
        return 0.0
    
    # Load English references from FLEURS
    english_refs = load_english_references(sample_ids)
    
    predictions = []
    references = []
    
    for sample in data:
        # Get prediction (could be 'prediction' or 'translation')
        pred = sample.get('prediction') or sample.get('translation')
        
        # Get ID
        sample_id = sample.get('id')
        if sample_id is None and 'file_name' in sample:
            sample_id = extract_id_from_filename(sample['file_name'])
        
        if sample_id is None:
            continue
        
        if pred is None or not pred.strip():
            print(f"Warning: Missing prediction for {sample.get('file_name', sample_id)}")
            continue
        
        # Get English reference from FLEURS
        ref = english_refs.get(sample_id)
        
        if ref is None or not ref.strip():
            print(f"Warning: Missing English reference for ID {sample_id}")
            continue
        
        predictions.append(pred.strip())
        references.append(ref.strip())
    
    if not predictions or not references:
        print(f"Warning: No valid predictions/references for {language_name}")
        return 0.0
    
    # sacreBLEU expects references as list of lists
    bleu_result = sacrebleu.corpus_bleu(predictions, [references])
    print(f"{language_name}: BLEU = {bleu_result.score:.2f} ({len(predictions)} samples)")
    
    return bleu_result.score


def compute_chrf_for_gemini(json_file, language_name):
    """Compute corpus chrF for Gemini translations vs. FLEURS English refs.

    chrF is often more robust than BLEU on morphologically rich
    languages.

    Args:
        json_file (str | os.PathLike): Path to the predictions JSON.
        language_name (str): Name of the language for log output.

    Returns:
        float: chrF score from ``sacrebleu.corpus_chrf``. Returns
        ``0.0`` when no aligned pairs are found.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get all sample IDs (handle both 'id' field and extracting from 'file_name')
    sample_ids = []
    for sample in data:
        if 'id' in sample:
            sample_ids.append(sample['id'])
        elif 'file_name' in sample:
            extracted_id = extract_id_from_filename(sample['file_name'])
            if extracted_id is not None:
                sample_ids.append(extracted_id)
    
    if not sample_ids:
        print(f"Error: No valid sample IDs found for {language_name}")
        return 0.0
    
    # Load English references from FLEURS
    english_refs = load_english_references(sample_ids)
    
    predictions = []
    references = []
    
    for sample in data:
        # Get prediction (could be 'prediction' or 'translation')
        pred = sample.get('prediction') or sample.get('translation')
        
        # Get ID
        sample_id = sample.get('id')
        if sample_id is None and 'file_name' in sample:
            sample_id = extract_id_from_filename(sample['file_name'])
        
        if sample_id is None:
            continue
            
        ref = english_refs.get(sample_id)
        
        if pred and pred.strip() and ref and ref.strip():
            predictions.append(pred.strip())
            references.append(ref.strip())
    
    if not predictions:
        print(f"Warning: No valid data for {language_name}")
        return 0.0
    
    chrf_result = sacrebleu.corpus_chrf(predictions, [references])
    print(f"{language_name}: chrF = {chrf_result.score:.2f}")
    
    return chrf_result.score


def compute_wer_for_gemini(json_file, language_name):
    """Compute Word Error Rate for Gemini translations.

    Lower is better; ``0.0`` is perfect.

    Args:
        json_file (str | os.PathLike): Path to the predictions JSON.
        language_name (str): Name of the language for log output.

    Returns:
        float: WER score from the ``evaluate`` ``wer`` metric in
        the ``0.0-1.0`` range. Returns ``1.0`` when no aligned pairs
        are found.
    """
    wer_metric = evaluate.load("wer")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get all sample IDs (handle both 'id' field and extracting from 'file_name')
    sample_ids = []
    for sample in data:
        if 'id' in sample:
            sample_ids.append(sample['id'])
        elif 'file_name' in sample:
            extracted_id = extract_id_from_filename(sample['file_name'])
            if extracted_id is not None:
                sample_ids.append(extracted_id)
    
    if not sample_ids:
        print(f"Error: No valid sample IDs found for {language_name}")
        return 1.0
    
    # Load English references from FLEURS
    english_refs = load_english_references(sample_ids)
    
    predictions = []
    references = []
    
    for sample in data:
        # Get prediction (could be 'prediction' or 'translation')
        pred = sample.get('prediction') or sample.get('translation')
        
        # Get ID
        sample_id = sample.get('id')
        if sample_id is None and 'file_name' in sample:
            sample_id = extract_id_from_filename(sample['file_name'])
        
        if sample_id is None:
            continue
            
        ref = english_refs.get(sample_id)
        
        if pred and pred.strip() and ref and ref.strip():
            predictions.append(pred.strip())
            references.append(ref.strip())
    
    if not predictions:
        print(f"Warning: No valid data for {language_name}")
        return 1.0  # worst possible WER
    
    wer_score = wer_metric.compute(predictions=predictions, references=references)
    print(f"{language_name}: WER = {wer_score:.4f}")
    
    return wer_score


def evaluate_all_languages(results_dir="./RESULTS/asr_translation_LLM_7B_few_shot"):
    """Compute BLEU, chrF and WER for every language results JSON in a directory.

    Workflow:
        1. Iterate over a fixed ``(language_name, FLEURS_code)``
           dictionary covering Irish, Welsh, Swahili, Yoruba, Hausa,
           Igbo and Luganda.
        2. For each language, derive
           ``<results_dir>/<code>.json``; skip the language with a
           printed message if the file is missing.
        3. Call :func:`compute_bleu_for_gemini`,
           :func:`compute_chrf_for_gemini` and
           :func:`compute_wer_for_gemini` in turn. Each helper loads
           the corresponding FLEURS English test split, aligns
           predictions to references by ``id`` (or
           ``extract_id_from_filename(file_name)``), normalises text,
           and computes the metric with ``sacrebleu`` (BLEU/chrF) or
           ``evaluate`` (WER).
        4. Aggregate the three scores into
           ``results[code] = {"language", "bleu", "chrf", "wer"}``.

    Args:
        results_dir (str | os.PathLike): Directory containing
            per-language ``<code>.json`` prediction files.

    Returns:
        dict[str, dict]: ``{FLEURS_code: {"language": str, "bleu":
        float, "chrf": float, "wer": float}}`` ready for
        :func:`print_summary` / ``json.dump``.
    """
    language_codes = {
        "Irish":    "ga_ie",
        "Welsh":     "cy_gb",
        "Swahili":   "sw_ke",
        "Yoruba":    "yo_ng",
        "Hausa":     "ha_ng",
        "Igbo":      "ig_ng",
        "Luganda":   "lg_ug",
    }
    
    results = {}
    
    for lang_name, code in language_codes.items():
        json_file = Path(results_dir) / f"{code}.json"
        print(json_file)
        if not json_file.exists():
            print(f"\nSkipping {lang_name}: file not found")
            continue
        
        print(f"\n{'='*50}")
        print(f"Evaluating {lang_name} ({code})")
        print(f"{'='*50}")
        
        bleu = compute_bleu_for_gemini(str(json_file), lang_name)
        chrf = compute_chrf_for_gemini(str(json_file), lang_name)
        wer = compute_wer_for_gemini(str(json_file), lang_name)
        
        results[code] = {
            "language": lang_name,
            "bleu": bleu,
            "chrf": chrf,
            "wer": wer
        }
    
    return results


def print_summary(results):
    """Print a per-language and averaged BLEU/chrF/WER table.

    Workflow:
        1. Print a header row
           ``Language | Code | BLEU | chrF | WER``.
        2. Print one formatted row per language with two-decimal BLEU
           and chrF and four-decimal WER.
        3. Compute the arithmetic mean of BLEU, chrF and WER across
           languages and print an ``AVERAGE`` summary row.

    Args:
        results (dict[str, dict]): ``{code: {"language", "bleu",
            "chrf", "wer"}}`` dict built by
            :func:`evaluate_all_languages`.

    Returns:
        None. Output is written to stdout.
    """
    print(f"\n{'='*70}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*70}")
    print(f"{'Language':<15} {'Code':<10} {'BLEU':<10} {'chrF':<10} {'WER':<10}")
    print(f"{'-'*70}")
    
    for code, metrics in results.items():
        print(f"{metrics['language']:<15} {code:<10} {metrics['bleu']:<10.2f} "
              f"{metrics['chrf']:<10.2f} {metrics['wer']:<10.4f}")
    
    # Calculate averages
    avg_bleu = sum(m['bleu'] for m in results.values()) / len(results)
    avg_chrf = sum(m['chrf'] for m in results.values()) / len(results)
    avg_wer = sum(m['wer'] for m in results.values()) / len(results)
    
    print(f"{'-'*70}")
    print(f"{'AVERAGE':<15} {'':<10} {avg_bleu:<10.2f} {avg_chrf:<10.2f} {avg_wer:<10.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Evaluate all languages
    results = evaluate_all_languages()
    
    # Print summary table
    print_summary(results)
    
    # Optionally save to JSON
    with open("RESULTS/metrics_BLEU_ASR_LLM-7B.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Results saved to /metrics_BLEU_ASR_LLM-7B.json")