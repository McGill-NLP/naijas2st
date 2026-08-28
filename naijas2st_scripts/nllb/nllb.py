"""Cascaded LRL -> English text translation using NLLB-200-3.3B.

Reads per-language ASR transcription JSONs produced by the cascaded
pipeline and translates each into English with Hugging Face's
``facebook/nllb-200-3.3B``. Writes one results JSON per language.
"""

import json
from pathlib import Path
from transformers import pipeline


def main():
    """Translate cascaded LRL ASR transcriptions to English with NLLB-200-3.3B.

    Workflow:
        1. Resolve ``test_base_dir`` (cascaded ASR JSONs) and the
           ``language_dir`` mapping from folder name to FLORES code.
        2. For each ``(language, language_code)`` pair:
            - Load ``<language>_transcriptions.json`` whose
              ``transcription[0]`` field holds the LRL ASR string.
            - For every utterance, instantiate a fresh
              ``transformers.pipeline("translation",
              model="facebook/nllb-200-3.3B", src_lang=language_code,
              tgt_lang="eng_Latn")``. (Re-creating the pipeline per
              utterance is wasteful but matches the original behaviour.)
            - Run inference, take ``translation_text`` from the first
              hypothesis, and append
              ``{"ID", "transcription", "translation"}`` to ``results``.
        3. Write ``results`` to
           ``./RESULTS/naijas2st/cascaded/LLM_1B_nllb-200-3.3B_<language>.json``.

    Outputs:
        One per-language translated JSON.

    Returns:
        None.
    """
    test_base_dir = Path("./RESULTS/naijas2st/asr_LLM_1B_test_set_for_cascaded")
    language_dir = {
        # "yoruba": "yor_Latn",
        # "hausa": "hau_Latn",
        # "igbo": "ibo_Latn", 
        "pidgin": "pcm_Latn"
    }

    for language, language_code in language_dir.items():
        print(f"\n→ Processing {language}...")
        language_format = f"{language}_transcriptions.json"
        test_set = test_base_dir / language_format

        results = []
        with open(test_set, 'r') as test_set_json:
            for utterance in json.load(test_set_json):
                print(utterance["transcription"][0])
                print(utterance["ID"])
                inputs = utterance["transcription"][0]

                translator = pipeline(
                    task="translation", 
                    model="facebook/nllb-200-3.3B", 
                    src_lang=language_code, 
                    tgt_lang="eng_Latn",  
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
        output_json = Path(f"./RESULTS/naijas2st/cascaded/LLM_1B_nllb-200-3.3B_{language}.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Saved results to {output_json}")


if __name__ == "__main__":
    main()
