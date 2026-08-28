"""Quick test for the Masakhane Yoruba POS tagger.

Loads ``masakhane/yoruba-pos-tagger-afroxlmr`` and prints the token
classification output for a hard-coded example sentence. The
input/output JSON paths below are placeholders for a future batch
pipeline.
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline


def main():
    """Sanity-check the Masakhane Yoruba POS tagger.

    Workflow:
        1. Load the ``masakhane/yoruba-pos-tagger-afroxlmr`` tokenizer
           and ``AutoModelForTokenClassification``.
        2. Wrap them in a ``TokenClassificationPipeline``.
        3. Run the pipeline on a hard-coded example sentence and print
           the per-token classifications.
        4. Define (but does not use) input/output JSON paths for a
           future batch pipeline (kept as placeholders for the
           Yoruba ``cascaded LLM_1B_nllb-200-3.3B`` predictions).

    Outputs:
        Token-classification output on stdout. No files written.

    Returns:
        None.
    """
    model_name = "masakhane/yoruba-pos-tagger-afroxlmr"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
    outputs = pipeline("Sowore gba ìtúsílẹ̀ lẹ́yìn ọjọ́ méjì ní àtìmọ́lé ọlọ́pàá")
    print(outputs)

    input_json = "./RESULTS/naijas2st/cascaded/LLM_1B_nllb-200-3.3B_yoruba_reformatted.json"
    output_json = "./RESULTS/naijas2st/pos_tags/cascadedLLM_1B_nllb-200-3.3B_yoruba_pos_tagged.json"


if __name__ == "__main__":
    main()
