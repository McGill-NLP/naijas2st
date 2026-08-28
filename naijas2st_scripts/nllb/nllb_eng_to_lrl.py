"""Cascaded English -> LRL text translation using NLLB-200-3.3B.

Reads English ASR transcription JSONs (matched to the appropriate
accent for each target LRL) and translates each utterance into the
target language with NLLB-200. Writes one results JSON per language.
"""

import json
from pathlib import Path
from transformers import pipeline


def main():
    """Translate English ASR transcriptions into each target LRL with NLLB-200.

    Workflow:
        1. For each ``(language, language_code)`` pair, pick the
           English ASR JSON whose accent matches the target language
           (``accent_dir[language]``) under ``test_base_dir``.
        2. For every utterance in that JSON:
            - Instantiate a fresh
              ``transformers.pipeline("translation",
              model="facebook/nllb-200-3.3B", src_lang="eng_Latn",
              tgt_lang=language_code)`` (per-utterance, same as the
              LRL->Eng variant).
            - Run inference and pull ``translation_text`` from the
              first hypothesis.
            - Append ``{"ID", "transcription", "translation"}``.
        3. Write the per-language predictions to
           ``./RESULTS/naijas2st/cascaded/eng_to_lrl_LLM_1B_nllb-200-3.3B_<language>.json``.

    Outputs:
        One per-language English -> LRL translated JSON.

    Returns:
        None.
    """
    test_base_dir = Path("./RESULTS/naijas2st/asr_LLM_1B_test_set_for_cascaded")
    # test_base_dir = Path("./RESULTS/naijas2st/asr_CTC_1B_test_set/")
    language_dir = {
        # "yoruba": "yor_Latn",
        # "hausa": "hau_Latn",
        # "igbo": "ibo_Latn", 
        "pidgin": "pcm_Latn"
    }
    accent_dir = {
        # "yoruba": "english_south_accent",
        # "igbo": "english_south_accent", 
        # "hausa": "english_north_accent", 
        "pidgin": "english_south_accent"
    }

    for language, language_code in language_dir.items():
        print(f"\n→ Processing English to {language}...")
        language_format = f"{language}_transcriptions.json"
        accent_format = f"{accent_dir[language]}_transcriptions.json"
        test_set = test_base_dir / accent_format
        # english_file = "english_transcriptions.json"
        # test_set = test_base_dir / english_file

        results = []
        with open(test_set, 'r') as test_set_json:
            for utterance in json.load(test_set_json):
                print(utterance["transcription"][0])
                print(utterance["ID"])
                inputs = utterance["transcription"][0]

                translator = pipeline(
                    task="translation", 
                    model="facebook/nllb-200-3.3B", 
                    src_lang="eng_Latn", 
                    tgt_lang=language_code,  
                    max_length=1024,  
                    min_length=1,
                    truncation=True 
                )
                
                print('pipeline prepared')
                translation = (translator(utterance["transcription"][0]))[0]['translation_text']
                print(f"Translation: {translation}")

                results.append({
                    "ID": utterance["ID"],
                    "transcription": utterance["transcription"][0],
                    "translation": translation
                })

        # Save results to JSON
        output_json = Path(f"./RESULTS/naijas2st/cascaded/eng_to_lrl_LLM_1B_nllb-200-3.3B_{language}.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Saved results to {output_json}")


if __name__ == "__main__":
    main()
