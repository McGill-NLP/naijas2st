"""POS-tag English reference and prediction sentences with spaCy.

spaCy variant of ``english.py``: uses ``en_core_web_sm`` to tag both
the reference and prediction strings in each language's
``*_reformatted.json`` and writes the tags back into the JSON.
"""

import json
import spacy

nlp = spacy.load("en_core_web_sm")

def pos_tagging(lang):
    """POS-tag reference and prediction English strings with spaCy.

    Workflow:
        1. Open the audio-LLM predictions JSON for ``lang``
           (``RESULTS/naijas2st/few_shot_S2T_lrl_to_eng_gemini31/
           <lang>_reformatted.json``).
        2. For each item, run the loaded ``en_core_web_sm`` pipeline on
           both ``reference`` and ``prediction``; attach the resulting
           ``[[token.text, token.pos_], ...]`` lists as
           ``pos_tags_reference`` / ``pos_tags_prediction``.
           ``transcription_tags`` / ``prediction_tags`` are reset per
           item to avoid leaking tags across rows.
        3. Write the enriched list to
           ``RESULTS/.../pos_tags/audiollm/lrl_to_eng_<lang>_pos_tagged_spacy.json``
           with compact ``separators`` (no spaces) since spaCy
           output JSON tends to be large.

    Args:
        lang (str): LRL name (used to locate the per-language
            ``*_reformatted.json``).

    Returns:
        None. Writes a ``_spacy.json`` POS-tagged file next to the input.
    """
    input_json = f"./RESULTS/naijas2st/few_shot_S2T_lrl_to_eng_gemini31/{lang}_reformatted.json"
    output_json = f"./RESULTS/naijas2st/pos_tags/audiollm/lrl_to_eng_{lang}_pos_tagged_spacy.json"

    with open(input_json, 'r') as infile:
        data = json.load(infile)

    for item in data:
        transcription_tags = []
        prediction_tags = []

        for token in nlp(item['reference']):
            transcription_tags.append([token.text, token.pos_])
        item['pos_tags_reference'] = transcription_tags

        for token in nlp(item['prediction']):
            prediction_tags.append([token.text, token.pos_])
        item['pos_tags_prediction'] = prediction_tags

    with open(output_json, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False, separators=(',', ':'))


if __name__ == "__main__":
    for lang in ['yoruba', 'igbo', 'hausa']:
        print(f"Processing {lang}...")
        pos_tagging(lang)