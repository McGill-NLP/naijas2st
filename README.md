# LRL LLM Speech Translation

Research code for benchmarking and improving speech translation for
low-resource languages (LRLs), comparing three families of systems on the
FLEURS benchmark and on a new NaijaS2ST test corpus
(`McGill-NLP/NaijaS2ST`):

1. **Cascaded** — Omnilingual ASR (Meta `omniASR_LLM_1B` / `LLM_7B` and
   Wav2Vec-CTC `1B`) followed by an MT model (NLLB‑200‑3.3B, AfriqueGemma,
   AfriqueQwen / Tiny‑Aya, or Gemini text MT).
2. **End-to-end** — Fine-tuned SeamlessM4T v2 (S2TT and S2ST) trained
   monolingually, multilingually, and in mixed pipelines.
3. **Audio LLMs** — Direct speech translation with Gemini 2.5 Flash,
   Gemini 3.1 Pro, Gemma 4 / 3n multimodal, and Azure `gpt-audio`
   (zero-shot, few-shot, many-shot, and score-conditioned refinement).

Covered languages: **Hausa**, **Igbo**, **Yoruba**, **Naija Pidgin**, **English**

This project is in collaboration with Mila, McGill University and Google Deepmind.

A preprint of the paper based on this project can be found here: https://arxiv.org/html/2604.16287v1


## Repository layout

```
.
├── FLEURS_scripts/             # FLEURS speech-translation pipeline (LRL <-> Eng)
├── naijas2st_scripts/          # All models/utilities for the NaijaS2ST dataset
├── metrics/                    # BLEU, chrF/chrF++, spBLEU, SSA-COMET, WER eval
├── plots/                      # Paper-ready bar/line/heatmap/violin plots + IAA
├── asr/                        # ASR processing and pipeline scripts
└── requirements_general_env.txt # requirements for env for all models except SeamlessM4T and Gemma
```

### `FLEURS_scripts/` — staged Gemini self-refinement pipeline

The four FLEURS scripts implement a three-stage *translate → score →
re-translate* loop with Gemini, plus zero/few-shot baselines in both
directions.

| Script | Role |
|---|---|
| `step1_few_shot_translation.py` | Stage 1: FLEURS LRL audio → English with Gemini 3.1 Pro, 5 audio few-shot demos per language. |
| `step1_few_shot_translation_eng_to_lrl.py` | Mirror of stage 1 in the English-audio → LRL-text direction. |
| `zero_shot_eng_to_lrl.py` | Zero-shot English audio → LRL text baseline. |
| `step2_scoring.py` | Stage 2: Gemini-as-judge rates each stage-1 translation 0–100. |
| `step3_scored_translation.py` | Stage 3: zero-shot re-translation conditioned on the stage-2 score. |
| `step3_few_shot_scored_translation.py` | Stage 3 variant that adds in-context (audio, prior translation, score, fixed translation) demos. |
| `get_one_example.py` | Helper to materialise a fixed list of FLEURS English wavs as local few-shot examples. |

### `naijas2st_scripts/` — models, datasets, and utilities

```
naijas2st_scripts/
├── gemini/         # Gemini 2.5 / 3.1 zero+few-shot S2T (LRL↔Eng) and TTS rendering of the translation
├── gemma/          # Gemma 4 / 3n text-MT (cascaded) and multimodal S2T (zero/few-shot)
├── nllb/           # Cascaded NLLB-200-3.3B text translation, both directions
├── open_ai/        # Azure gpt-audio LRL↔Eng S2T (zero/few-shot) and TTS variants
├── tiny_aya/       # CohereLabs Tiny-Aya + McGill AfriqueGemma/Qwen text-MT
├── seamless_m4T/   # SeamlessM4T v2 end-to-end S2T / S2ST
│   ├── dataset/    # HF→manifest builders (parallel LRL/Eng pairs, accent-aware)
│   ├── training/   # HF Trainer + custom STSTrainer drivers (mono/multi + pidgin)
│   ├── inference/  # Batch S2T and S2ST inference (custom-trainer + zero-shot)
│   ├── sts/        # Discrete unit extraction for SeamlessM4T S2ST targets
│   └── test_checkpoints.py
├── pos_taggers/    # NLTK English + Masakhane afro-xlmr POS taggers and evaluation
├── dataset_utils/  # HF dataset stats (hours), parallel-sampling, renaming
├── test/           # Local recordings (yoruba/igbo/hausa/pidgin) for inference
├── reformat_*.py   # Attach `source` & `reference` text via the metadata Excel
├── parquet_conversion.py    # Convert wavs into a partitioned Parquet dataset
├── create_predictions_json.py
└── check_mismatches.py      # Audit prediction lines vs. wav counts per user
```

The `seamless_m4T/dataset/NaijaS2ST_builder*.py` builders
stream `McGill-NLP/naijas2st`, pair each LRL utterance with
its English counterpart by shared base text ID (using a preferred
Nigerian English accent per language), and write a JSONL manifest for
SeamlessM4T fine-tuning. The `training/` drivers wrap HuggingFace
`Seq2SeqTrainer` (with a custom `STSTrainer` for S2ST) and an on-demand
paired dataset; the `inference/` drivers use either the
`seamless_communication.Translator` API or HF `SeamlessM4Tv2*` models.

### `metrics/` — evaluation

| Script | What it computes |
|---|---|
| `metrics.py` | BLEU + chrF (sacrebleu) + WER (evaluate) against FLEURS English references, aligned by sample ID. |
| `metrics_zero_shot.py` | Same as above, but identifies samples by `audio.path` for prediction JSONs that only store `file_name`. |
| `chrf_plus.py` | Maja Popović's reference chrF/chrF++ implementation (importable from `spbleu.py`). |
| `spbleu.py` | chrF, sacreBLEU, spBLEU, spBLEU-1K and chrF++ on `*_reformatted.json` predictions. |
| `ssa_comet.py` | SSA-COMET (`McGill-NLP/ssa-comet-mtl-final`) on the `(src, mt, ref)` triple. |

### `plots/` — paper-ready figures + inter-annotator analysis

- `paper_ready_plots.py`, `stt_method_line_plots.py`, `stt_method_bar_plot.py`, `heatmap.py` — spBLEU comparisons across (method, direction, language) on the NaijaS2ST test set.
- `sentence_length.py` — Violin plot of prediction-vs-reference length ratios per system.
- `human_eval_correlation.py` — Spearman/Pearson correlations and ICC2 across human annotators in `human_eval/`, plus method-significance tests and disagreement boxplots.

### `asr/` - asr scripts

| Script | Purpose |
|---|---|
| `asr_translation.py` | Cascaded LRL ASR JSONs → Gemini 2.5 Flash English MT. |
| `asr_translation_few_shot.py` | Same cascade with 5 text-only few-shot examples. |
| `fb_asr.py` | Omnilingual ASR LLM-7B over FLEURS test for a fixed language set. |
| `fb_asr_NaijaS2ST.py` | Omnilingual ASR LLM-1B over the local NaijaS2ST test set. |
| `full_fleurs_fb_asr.py` | Resume-able Omnilingual ASR sweep across all FLEURS languages (driven by `omnlingual_to_fleurs.csv`). |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_general_env.txt   # Env for all models except SeamlessM4T and Gemma

export HF_TOKEN=...               # for HuggingFace + FLEURS streaming
export GOOGLE_API_KEY=...         # for any Gemini script
export AZURE_OPENAI_ENDPOINT=...  # only for naijas2st_scripts/open_ai/
export AZURE_OPENAI_API_KEY=...
```

### Typical FLEURS pipeline

```bash
# Audio → English with 5-shot Gemini 3.1 Pro
python FLEURS_scripts/step1_few_shot_translation.py
# Gemini-as-judge scores stage-1 translations 0–100
python FLEURS_scripts/step2_scoring.py
# Score-conditioned re-translation (zero-shot)
python FLEURS_scripts/step3_scored_translation.py
# Evaluate any of the above
python metrics/metrics.py
```

### Typical NaijaS2ST pipeline

```bash
# 1. Cascaded baseline
python asr/fb_asr_NaijaS2ST.py                        # ASR
python naijas2st_scripts/nllb/nllb.py                         # NLLB-200 MT
python naijas2st_scripts/reformat_files.py                    # add source/reference
python metrics/ssa_comet.py                                        # COMET eval

# 2. End-to-end SeamlessM4T
python naijas2st_scripts/seamless_m4T/dataset/NaijaS2ST_builder.py
python naijas2st_scripts/seamless_m4T/training/train_explicit_language_pairs.py \
    --task stt --model-dir ./models/seamless-stt
python naijas2st_scripts/seamless_m4T/inference/batch_inference_custom_trainer.py \
    --task stt --model-path ./models/seamless-stt/final \
    --data-dir naijas2st_scripts/test/ \
    --output-dir RESULTS/naijas2st/seamless_stt/

# 3. Audio-LLM (Gemini 3.1 Pro few-shot)
python naijas2st_scripts/gemini/few_shot_naijas2st.py
```

## Conventions

- **Prediction schema**: every per-language results JSON is a list of
  records with at least `id`/`ID`/`file_name`, `prediction`, and
  (after `reformat_*.py`) `source` and `reference`. The COMET-ready
  schema produced by `reformat_file.py` is
  `{ID, source, reference, prediction}`.
- **Few-shot data**: each `few_shot_data/<code>/` (FLEURS) or
  `few_shot/<language>/` (NaijaS2ST) folder holds parallel
  `<id>_<code>.wav` + `<id>.txt` pairs; the `.txt` has two lines
  (`<Language> transcription:` and `English transcription:`).
- **Resumability**: most long-running scripts rewrite their per-language
  results JSON after every utterance so they can be killed and restarted
  safely; some additionally skip already-processed IDs.
- **Caching**: HuggingFace cache is routed through `$SCRATCH_CACHE`
  (default `./hf_cache`)


## Citation

If you use this code or reuse our models in your research, please cite our paper:

```bibtex
@misc{maltais2026naijas2stmultiaccentbenchmarkspeechtospeech,
      title={NaijaS2ST: A Multi-Accent Benchmark for Speech-to-Speech Translation in Low-Resource Nigerian Languages}, 
      author={Marie Maltais and Yejin Jeon and Min Ma and Shamsuddeen Hassan Muhammad and Idris Abdulmumin and Maryam Ibrahim Mukhtar and Daud Abolade and Joel Okepefi and Johnson Sewedo and David Ifeoluwa Adelani},
      year={2026},
      eprint={2604.16287},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2604.16287}, 
}
```
