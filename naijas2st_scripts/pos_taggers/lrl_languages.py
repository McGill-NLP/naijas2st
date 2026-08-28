"""Add POS tags to LRL reference/prediction strings using Masakhane taggers.

For each language, loads the matching ``masakhane/<lang>-pos-tagger-afroxlmr``
model and annotates the ``reference`` and ``prediction`` fields in the
SeamlessM4T English->LRL ``*_reformatted.json`` with POS tags.
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import json

def pos_tag(language):
    """POS-tag reference and prediction strings for one LRL using Masakhane.

    Workflow:
        1. Load the SeamlessM4T English -> LRL reformatted JSON
           (``seamless_eng_to_lrl_finetuned_mono_<language>/
           eng_to_<language>_translations_reformatted.json``).
        2. Load ``masakhane/<language>-pos-tagger-afroxlmr`` (tokenizer
           + ``AutoModelForTokenClassification``) and wrap them in a
           ``TokenClassificationPipeline``.
        3. For each item:
            - Skip items missing the ``reference`` field with a printed
              warning.
            - Tag both the ``reference`` and ``prediction`` strings,
              storing the result as ``[[word, entity], ...]`` lists in
              ``pos_tags_reference`` / ``pos_tags_prediction`` (raw
              softmax scores are dropped to keep the JSON compact).
        4. Write the enriched list to
           ``RESULTS/.../pos_tags/end_to_end/<language>_pos_tagged.json``
           with no indentation to minimise size.

    Args:
        language (str): LRL name; selects both the input JSON and the
            Masakhane tagger checkpoint.

    Returns:
        None. Writes ``<language>_pos_tagged.json`` alongside the rest
        of the end-to-end POS outputs.
    """
    input_json = f"./RESULTS/naijas2st/seamless_stt_all/eng_to_lrl/seamless_eng_to_lrl_finetuned_mono_{language}/eng_to_{language}_translations_reformatted.json"
    output_json = f"./RESULTS/naijas2st/pos_tags/end_to_end/{language}_pos_tagged.json"
    with open(input_json, 'r') as infile:
        data = json.load(infile)
    
    model_name = f"masakhane/{language}-pos-tagger-afroxlmr"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)

    for i, item in enumerate(data):
        if 'reference' not in item:
            print(f"Missing prediction at index {i}: {item}")
            continue

        item['pos_tags_reference'] = [
            [t['word'], t['entity']] for t in pipeline(item['reference'])
        ]
        item['pos_tags_prediction'] = [
            [t['word'], t['entity']] for t in pipeline(item['prediction'])
        ]

    with open(output_json, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False, separators=(',', ':'))


if __name__ == "__main__":
    languages = ['yoruba', 'igbo', 'hausa']
    for language in languages:
        print(f"Processing {language}...")
        pos_tag(language)