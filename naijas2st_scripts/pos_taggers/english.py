"""POS-tag English reference and prediction sentences with NLTK.

For each language's ``*_reformatted.json`` prediction file, tokenises
the ``reference`` and ``prediction`` strings, runs the NLTK averaged
perceptron tagger, and writes the tagged tokens back into the JSON.
"""

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import json
import nltk

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def pos_tagging(lang):
    """POS-tag reference and prediction strings for one LRL's predictions JSON.

    Workflow:
        1. Open the audio-LLM predictions JSON for ``lang``
           (``RESULTS/naijas2st/few_shot_S2T_lrl_to_eng_gemini31/
           <lang>_reformatted.json``) which has ``reference`` (English
           gold) and ``prediction`` (model English) fields per item.
        2. For each item, run NLTK ``pos_tag(word_tokenize(text))`` on
           both ``reference`` and ``prediction`` and attach the result
           lists as ``pos_tags_reference`` / ``pos_tags_prediction``.
           Each tag is a ``(word, POS)`` tuple using the Penn Treebank
           tagset.
        3. Write the enriched list to
           ``RESULTS/.../pos_tags/audiollm/lrl_to_eng_<lang>_pos_tagged.json``.

    Args:
        lang (str): LRL name (used to locate the per-language
            ``*_reformatted.json``).

    Returns:
        None. The tagged JSON is written next to the input.
    """
    input_json = f"./RESULTS/naijas2st/few_shot_S2T_lrl_to_eng_gemini31/{lang}_reformatted.json"
    output_json = f"./RESULTS/naijas2st/pos_tags/audiollm/lrl_to_eng_{lang}_pos_tagged.json"
    with open(input_json, 'r') as infile:
        data = json.load(infile)
    
    for item in data:
        reference = item['reference']
        pos_tags_transcription = pos_tag(word_tokenize(reference))
        item['pos_tags_reference'] = pos_tags_transcription

        prediction = item['prediction']
        pos_tags_prediction = pos_tag(word_tokenize(prediction))
        item['pos_tags_prediction'] = pos_tags_prediction
    
    with open(output_json, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    langs = ['yoruba', 'igbo', 'hausa']
    for lang in langs:
        print(f"Processing {lang}...")
        pos_tagging(lang)