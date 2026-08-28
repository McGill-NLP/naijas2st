"""Fine-tune SeamlessM4T v2 for Nigerian Pidgin -> English STT on a local dataset.

Audio is laid out under ``<data_root>/<split>/<user_id>/<text_id>.wav``
and the labels come from a single Excel sheet (``Joel`` and ``Johnson``
unioned) with ``source_filename``, ``english`` and three speaker
columns. Train text IDs are non-``TE`` (``PMD_*``/``PTR_*``/``PNX_*``);
dev text IDs are ``PTE_*``.
"""

import argparse
import os
import sys
import torch
import librosa
import soundfile as sf


def main():
    """Parse CLI args and run Pidgin -> English SeamlessM4T STT training.

    Workflow:
        1. Configure HuggingFace cache directories under
           ``SCRATCH_CACHE`` so model and dataset caches live on
           scratch.
        2. Parse CLI args (model dir, epochs, batch sizes, grad-accum,
           learning rate, warmup/eval/save/log steps, max audio length,
           resume, num workers, gradient checkpointing, force
           reprocess, base model, LoRA flags, tensorboard, data root).
        3. Read pidgin metadata from an Excel sheet via
           :func:`load_xlsx_metadata` (concatenates the ``Joel`` and
           ``Johnson`` sheets, drops rows missing ``source_filename``
           or ``english``).
        4. Build (or load cached) per-split pair metadata via
           :func:`create_pair_metadata`. Train/dev are split by
           the ``TE`` marker in the text ID; each row expands into up
           to three (audio, English-text) pairs, one per non-null
           ``user 1/2/3`` column, with audio resolved as
           ``<data_root>/<split>/<user_id>_updated/<text_id>.wav``.
        5. Load ``SeamlessM4Tv2ForSpeechToText`` + processor.
        6. Construct :class:`LocalPidginDataset` for train and dev
           and run :func:`prefilter_dataset` to drop entries whose
           wav is too short or too long (using ``sf.info`` so the full
           array doesn't need decoding).
        7. Build ``Seq2SeqTrainingArguments``, the padding collator,
           and a ``Seq2SeqTrainer`` with the NaN-loss callback. Train,
           save the final model + processor to ``<output_dir>/final``.

    Inputs:
        ``SCRATCH_CACHE``; CLI args; local pidgin recordings and
        metadata Excel.

    Outputs:
        Trained Pidgin -> English STT checkpoint under
        ``--model-dir`` and a final saved model directory.

    Returns:
        None.
    """
    print('Parsing arguments...')
    sys.stdout.flush()

    # =========================================================================
    # ENV / CACHE
    # =========================================================================
    scratch_cache = os.environ.get("SCRATCH_CACHE", "./hf_cache")
    os.makedirs(scratch_cache, exist_ok=True)

    os.environ['HF_HOME']                = scratch_cache
    os.environ['HF_DATASETS_CACHE']      = f"{scratch_cache}/datasets"
    os.environ['HUGGINGFACE_HUB_CACHE']  = f"{scratch_cache}/hub"
    os.environ['HF_DATASETS_OFFLINE']    = '0'
    os.environ['XET_CACHE_DIR']          = f"{scratch_cache}/xet"
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # =========================================================================
    # ARGS
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='SeamlessM4T Fine-Tuning — Local Pidgin→english STT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start_train_local.py \\
      --data-root ./data \\
      --metadata ./metadata.xlsx \\
      --model-dir ./my-model \\
      --epochs 5
        """
    )

    # --- data ---
    parser.add_argument('--data-root',    type=str, required=True,
                        help='Root dir containing train/ and dev/ subdirs with <user_id>/<text_id>.wav')
    parser.add_argument('--metadata',     type=str, required=True,
                        help='Path to xlsx metadata file (sheets: Joel, Johnson)')
    parser.add_argument('--metadata-dir', type=str, default='./pair_metadata',
                        help='Directory to cache processed metadata JSON')
    parser.add_argument('--force-reprocess', action='store_true',
                        help='Ignore cached metadata and rebuild from scratch')

    # --- model ---
    parser.add_argument('--model-dir',    type=str, required=True)
    parser.add_argument('--base-model',   type=str,
                        default='facebook/seamless-m4t-v2-large')
    parser.add_argument('--lora',         action='store_true')
    parser.add_argument('--lora-r',       type=int, default=16)

    # --- training ---
    parser.add_argument('--epochs',           type=int,   default=3)
    parser.add_argument('--batch-size',       type=int,   default=2)
    parser.add_argument('--grad-accum',       type=int,   default=8)
    parser.add_argument('--lr',               type=float, default=3e-5)
    parser.add_argument('--warmup-steps',     type=int,   default=500)
    parser.add_argument('--max-audio-length', type=int,   default=30)
    parser.add_argument('--save-steps',       type=int,   default=1000)
    parser.add_argument('--eval-steps',       type=int,   default=500)
    parser.add_argument('--logging-steps',    type=int,   default=50)
    parser.add_argument('--resume',           type=str,   default=None)
    parser.add_argument('--num-workers',      type=int,   default=0)
    parser.add_argument('--no-fp16',          action='store_true')
    parser.add_argument('--no-gradient-checkpointing', action='store_true')
    parser.add_argument('--no-tensorboard',   action='store_true')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SEAMLESSM4T FINE-TUNING — PIDGIN → english STT")
    print("=" * 70)
    print(f"  Base Model  : {args.base_model}")
    print(f"  Data Root   : {args.data_root}")
    print(f"  Metadata    : {args.metadata}")
    print(f"  Output Dir  : {args.model_dir}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch Size  : {args.batch_size} x{args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print(f"  LR          : {args.lr}")
    print(f"  Num Workers : {args.num_workers}")

    if torch.cuda.is_available():
        print(f"\n  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  Memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n  WARNING: No GPU detected!")

    print("\n" + "=" * 70)
    print("Ready to start. Press Ctrl+C within 5 seconds to cancel...")

    import time
    try:
        for i in range(5, 0, -1):
            print(f"  Starting in {i}...", end='\r')
            time.sleep(1)
        print(" " * 30, end='\r')
    except KeyboardInterrupt:
        print("\n\nTraining cancelled.")
        return

    # =========================================================================
    # IMPORTS
    # =========================================================================
    import pandas as pd
    import json
    import gc
    import numpy as np
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Any, Dict, List
    from transformers import (
        SeamlessM4Tv2ForSpeechToText,
        SeamlessM4TProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        TrainerCallback,
    )

    # =========================================================================
    # CONFIG
    # =========================================================================
    class Config:
        MODEL_NAME                  = args.base_model
        TASK                        = 'stt'
        OUTPUT_DIR                  = args.model_dir
        DATA_ROOT                   = Path(args.data_root)
        METADATA_XLSX               = args.metadata
        METADATA_DIR                = args.metadata_dir
        NUM_TRAIN_EPOCHS            = args.epochs
        PER_DEVICE_TRAIN_BATCH_SIZE = args.batch_size
        GRADIENT_ACCUMULATION_STEPS = args.grad_accum
        LEARNING_RATE               = args.lr
        WARMUP_STEPS                = args.warmup_steps
        EVAL_STEPS                  = args.eval_steps
        SAVE_STEPS                  = args.save_steps
        LOGGING_STEPS               = args.logging_steps
        MAX_AUDIO_LENGTH            = args.max_audio_length
        BF16                        = True
        FP16                        = False
        GRADIENT_CHECKPOINTING      = not args.no_gradient_checkpointing
        DATALOADER_NUM_WORKERS      = args.num_workers
        SRC_LANG                    = 'pcm'   # Nigerian Pidgin
        TGT_LANG                    = 'eng'

    # =========================================================================
    # METADATA LOADING
    # =========================================================================

    def load_xlsx_metadata() -> pd.DataFrame:
        """Read and concatenate every sheet in the metadata Excel file.

        Returns:
            pandas.DataFrame: DataFrame with at least
            ``source_filename``, ``english``, and the three speaker
            columns. Rows missing ``source_filename`` or ``english``
            are dropped.

        Raises:
            ValueError: If the Excel is missing one of the required
                columns.
        """
        print(f"\nReading metadata from {Config.METADATA_XLSX}...")
        xl = pd.ExcelFile(Config.METADATA_XLSX)
        print(f"  Sheets found: {xl.sheet_names}")

        frames = []
        for sheet in xl.sheet_names:
            df = xl.parse(sheet, dtype=str)
            df['_sheet'] = sheet
            frames.append(df)
            print(f"  Sheet '{sheet}': {len(df)} rows, columns: {list(df.columns)}")

        combined = pd.concat(frames, ignore_index=True)

        # Normalise column names (strip whitespace, lowercase)
        combined.columns = [c.strip().lower() for c in combined.columns]

        required = {'source_filename', 'english', 'user 1', 'user 2', 'user 3'}
        missing = required - set(combined.columns)
        if missing:
            raise ValueError(
                f"xlsx is missing required columns: {missing}. "
                f"Found columns: {list(combined.columns)}"
            )

        before = len(combined)
        combined = combined.dropna(subset=['source_filename', 'english'])
        combined = combined[combined['source_filename'].str.strip() != '']
        combined = combined[combined['english'].str.strip() != '']
        print(f"  Total rows after dropping empties: {len(combined)} (dropped {before - len(combined)})")

        for col in ['source_filename', 'english', 'user 1', 'user 2', 'user 3']:
            combined[col] = combined[col].str.strip()

        return combined

    # =========================================================================
    # PAIR METADATA CREATION
    # =========================================================================

    USER_COLS = ['user 1', 'user 2', 'user 3']

    def is_dev_id(text_id: str) -> bool:
        """Dev-set Pidgin text IDs contain ``"TE"`` (e.g. ``PTE_0010``).

        Args:
            text_id (str): Source-side text ID.

        Returns:
            bool: ``True`` when the (upper-cased) ID contains
            ``"TE"``.
        """
        return 'TE' in text_id.upper()

    def create_pair_metadata(split_name: str, df_all: pd.DataFrame) -> dict:
        """Build per-split pair metadata, expanding each row by user column.

        Each Excel row can yield up to three (audio, English) pairs (one
        per non-empty ``user 1/2/3``). Audio is resolved as
        ``<data_root>/<split>/<user_id>_updated/<source_filename>.wav``.

        Args:
            split_name (str): ``"train"`` or ``"dev"``.
            df_all (pandas.DataFrame): Combined metadata frame from
                :func:`load_xlsx_metadata`.

        Returns:
            dict: ``{"split": str, "num_pairs": int, "pairs":
            list[dict]}`` metadata dict written to disk.
        """
        print(f"\n{'='*70}")
        print(f"PAIR METADATA: {split_name.upper()}")
        print(f"{'='*70}")

        metadata_file = Path(Config.METADATA_DIR) / f"{split_name}_metadata.json"

        if metadata_file.exists() and not args.force_reprocess:
            print(f"✓ Loading cached metadata from {metadata_file}")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"✓ Loaded {metadata['num_pairs']} pairs")
            return metadata

        # Filter to the right split by text_id convention
        if split_name == 'dev':
            df_split = df_all[df_all['source_filename'].apply(is_dev_id)].copy()
        else:
            df_split = df_all[~df_all['source_filename'].apply(is_dev_id)].copy()

        print(f"  Rows for '{split_name}': {len(df_split)}")

        split_dir     = Config.DATA_ROOT / split_name
        pairs         = []
        missing_audio = []

        for _, row in df_split.iterrows():
            text_id  = row['source_filename']
            eng_text = row['english']

            for col in USER_COLS:
                user_id = row.get(col, '')
                if not user_id or pd.isna(user_id) or str(user_id).strip() == '':
                    continue
                user_id  = str(user_id).strip()
                user_id_format = f"{user_id}_updated"
                wav_path = split_dir / user_id_format / f"{text_id}.wav"

                if not wav_path.exists():
                    missing_audio.append(str(wav_path))
                    continue

                pairs.append({
                    'src_text_id': text_id,
                    'user_id':     user_id,
                    'user_col':    col,
                    'wav_path':    str(wav_path),
                    'tgt_text':    eng_text,
                    'sheet':       row.get('_sheet', ''),
                })

        print(f"  Created {len(pairs)} pairs ({len(missing_audio)} skipped — audio not found)")
        if missing_audio:
            print(f"  First 10 missing audio files:")
            for p in missing_audio[:10]:
                print(f"    {p}")
            if len(missing_audio) > 10:
                print(f"    ... and {len(missing_audio) - 10} more")

        os.makedirs(Config.METADATA_DIR, exist_ok=True)
        metadata = {
            'split':     split_name,
            'num_pairs': len(pairs),
            'pairs':     pairs,
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        print(f"  Saved metadata to {metadata_file}")

        return metadata

    # =========================================================================
    # DATASET
    # =========================================================================

    class LocalPidginDataset(torch.utils.data.Dataset):
        """
        Loads pidgin .wav files from disk and returns processed features
        paired with english text labels.
        """

        def __init__(self, metadata: dict, processor, max_audio_length: int = 30):
            self.pairs            = metadata['pairs']
            self.processor        = processor
            self.max_audio_length = max_audio_length
            self._error_count     = 0
            self._rejection_counts: Dict[str, int] = {}
            print(f"✓ Dataset '{metadata['split']}': {len(self.pairs)} pairs")

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            pair = self.pairs[idx]
            try:
                wav_path = pair['wav_path']

                # Load audio
                audio_array, sr = librosa.load(wav_path, sr=None, mono=True)

                # Resample to 16 kHz if needed
                if sr != 16000:
                    audio_array = librosa.resample(
                        audio_array.astype('float32'),
                        orig_sr=sr,
                        target_sr=16000,
                    )
                    sr = 16000

                duration = len(audio_array) / sr

                if duration < 0.5:
                    self._reject(idx, f"audio too short ({duration:.2f}s) | {pair['src_text_id']}")
                    return None
                if duration > self.max_audio_length:
                    self._reject(idx, f"audio too long ({duration:.2f}s) | {pair['src_text_id']}")
                    return None

                # Audio features
                inputs = self.processor(
                    audio=audio_array,
                    sampling_rate=16000,
                    src_lang=Config.SRC_LANG,
                    return_tensors="pt",
                )
                input_features = inputs.input_features.squeeze(0)  # [T, 160]
                attention_mask = inputs.attention_mask.squeeze(0)   # [T]

                if torch.isnan(input_features).any() or torch.isinf(input_features).any():
                    self._reject(idx, f"NaN/Inf in input_features | {pair['src_text_id']}")
                    return None

                # Text labels — condition on target language
                tgt_text = pair['tgt_text'].strip()
                if not tgt_text:
                    self._reject(idx, f"empty tgt_text | {pair['src_text_id']}")
                    return None

                conditioned_text = f"<{Config.TGT_LANG}> {tgt_text}"
                labels = self.processor.tokenizer(
                    text=conditioned_text,
                    padding=False,
                    truncation=True,
                    max_length=224,
                    return_tensors="pt",
                ).input_ids.squeeze(0)

                if labels.shape[0] == 0:
                    self._reject(idx, f"empty labels | {pair['src_text_id']}")
                    return None

                return {
                    'input_features': input_features,
                    'attention_mask': attention_mask,
                    'labels':         labels,
                }

            except Exception as e:
                self._error_count += 1
                if self._error_count <= 20:
                    import traceback
                    print(f"  ⚠ Exception at pair {idx} ({pair.get('src_text_id','?')}): {e}")
                    traceback.print_exc()
                return None

        def _reject(self, idx: int, reason: str):
            self._rejection_counts[reason] = self._rejection_counts.get(reason, 0) + 1
            if self._rejection_counts[reason] <= 5:
                print(f"  ↩ Pair {idx} rejected: {reason}")

    # =========================================================================
    # DATA COLLATOR
    # =========================================================================

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict]) -> Dict:
            features = [f for f in features if f is not None]

            if not features:
                return {
                    'input_features': torch.zeros(1, 100, 160),
                    'attention_mask': torch.zeros(1, 100, dtype=torch.long),
                    'labels':         torch.full((1, 1), -100, dtype=torch.long),
                }

            input_feats = [f['input_features'] for f in features]
            attn_masks  = [f['attention_mask']  for f in features]
            labels_list = [f['labels']          for f in features]

            # Pad input features to longest in batch
            max_T = max(f.shape[0] for f in input_feats)
            padded_feats, padded_masks = [], []
            for feat, mask in zip(input_feats, attn_masks):
                pad = max_T - feat.shape[0]
                padded_feats.append(
                    torch.nn.functional.pad(feat, (0, 0, 0, pad)) if pad > 0 else feat
                )
                padded_masks.append(
                    torch.nn.functional.pad(mask, (0, pad), value=0) if pad > 0 else mask
                )

            # Pad labels with -100 (ignored in cross-entropy)
            max_L = max(l.shape[0] for l in labels_list)
            padded_labels = []
            for l in labels_list:
                pad = max_L - l.shape[0]
                padded_labels.append(
                    torch.nn.functional.pad(l, (0, pad), value=-100) if pad > 0 else l
                )

            return {
                'input_features': torch.stack(padded_feats),
                'attention_mask': torch.stack(padded_masks),
                'labels':         torch.stack(padded_labels),
            }

    # =========================================================================
    # CALLBACK
    # =========================================================================

    class NaNLossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            for key in ('loss', 'eval_loss'):
                val = logs.get(key)
                if val is not None and (val != val or val == float('inf')):
                    print(f"\n{'!'*70}")
                    print(f"  ⚠ {key} is {val} at step {state.global_step}")
                    print(f"  Check: label token IDs, empty batches, audio NaNs")
                    print(f"{'!'*70}\n")
                    sys.stdout.flush()

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None:
                return
            print(f"\n{'='*50}")
            print(f" Evaluation at step {state.global_step}")
            for k, v in metrics.items():
                print(f"     {k}: {v:.4f}" if isinstance(v, float) else f"     {k}: {v}")
            print(f"{'='*50}\n")
            sys.stdout.flush()

    # =========================================================================
    # PRE-FILTER (check audio length using sf.info, no full decode)
    # =========================================================================

    def prefilter_dataset(dataset: LocalPidginDataset, label: str = 'train') -> LocalPidginDataset:
        """Drop pairs whose wav is too long or too short (per ``sf.info``).

        Args:
            dataset (LocalPidginDataset): Dataset to filter (mutated in
                place).
            label (str): Human-friendly split name used in log output.

        Returns:
            LocalPidginDataset: The same dataset restricted to valid
            pairs.
        """
        print(f"\n🔎 Pre-filtering {label} dataset...")
        valid_pairs      = []
        rejected_long    = []
        rejected_short   = []
        rejected_error   = []

        for i, pair in enumerate(dataset.pairs):
            if i % 5000 == 0:
                print(f"  Scanning {i}/{len(dataset.pairs)}...", end='\r')
            try:
                info     = sf.info(pair['wav_path'])
                duration = info.frames / info.samplerate

                if duration < 0.5:
                    rejected_short.append((i, pair['src_text_id'], duration))
                elif duration > dataset.max_audio_length:
                    rejected_long.append((i, pair['src_text_id'], duration))
                else:
                    valid_pairs.append(pair)
            except Exception as e:
                rejected_error.append((i, pair['src_text_id'], str(e)))

        total_rejected = len(rejected_long) + len(rejected_short) + len(rejected_error)
        print(f"  ✓ {len(valid_pairs)} valid, {total_rejected} rejected\n")

        if rejected_long:
            print(f"    Too long  ({len(rejected_long)}, max={dataset.max_audio_length}s):")
            for idx, tid, dur in rejected_long[:10]:
                print(f"      {idx:6d}  {tid:20s}  {dur:.2f}s")
        if rejected_short:
            print(f"    Too short ({len(rejected_short)}):")
            for idx, tid, dur in rejected_short[:10]:
                print(f"      {idx:6d}  {tid:20s}  {dur:.2f}s")
        if rejected_error:
            print(f"    Errors    ({len(rejected_error)}):")
            for idx, tid, err in rejected_error[:10]:
                print(f"      {idx:6d}  {tid:20s}  {err}")

        dataset.pairs = valid_pairs
        return dataset

    # =========================================================================
    # BUILD METADATA
    # =========================================================================

    print("\nLoading xlsx metadata...")
    df_all = load_xlsx_metadata()

    print("\nCreating/loading pair metadata...")
    train_metadata = create_pair_metadata('train', df_all)

    # After: train_metadata = create_pair_metadata('train', df_all)
    print("\nDiagnosing paths...")
    if train_metadata['num_pairs'] == 0:
        # Show what paths we're trying to build
        df_sample = df_all[~df_all['source_filename'].apply(is_dev_id)].head(3)
        split_dir = Config.DATA_ROOT / 'train'
        print(f"  split_dir exists: {split_dir.exists()} -> {split_dir}")
        print(f"  Contents of split_dir (first 5):")
        if split_dir.exists():
            for p in list(split_dir.iterdir())[:5]:
                print(f"    {p}")
                if p.is_dir():
                    for f in list(p.iterdir())[:3]:
                        print(f"      {f}")
        print(f"\n  Sample rows from xlsx:")
        for _, row in df_sample.iterrows():
            for col in ['user 1', 'user 2', 'user 3']:
                uid = row.get(col, '')
                tid = row['source_filename']
                if uid and str(uid).strip():
                    tried = split_dir / str(uid).strip() / f"{tid}.wav"
                    print(f"    col={col}  user_id={uid}  text_id={tid}")
                    print(f"    -> {tried}  exists={tried.exists()}")

    val_metadata   = create_pair_metadata('dev',   df_all)

    # =========================================================================
    # LOAD MODEL & PROCESSOR
    # =========================================================================

    print("\nLoading processor and model...")
    processor = SeamlessM4TProcessor.from_pretrained(Config.MODEL_NAME)
    model     = SeamlessM4Tv2ForSpeechToText.from_pretrained(Config.MODEL_NAME)

    if args.lora:
        print(f"Applying LoRA (rank={args.lora_r})...")
        print("LoRA support is commented out — uncomment peft imports to enable.")
        # from peft import LoraConfig, get_peft_model
        # lora_config = LoraConfig(
        #     r=args.lora_r,
        #     lora_alpha=args.lora_r * 2,
        #     target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        #     lora_dropout=0.1,
        #     bias="none",
        #     task_type="SEQ_2_SEQ_LM",
        # )
        # model = get_peft_model(model, lora_config)
        # model.print_trainable_parameters()
        # model.config.use_cache = False

    print("Model loaded.")

    # =========================================================================
    # DATASETS & COLLATOR
    # =========================================================================

    print("\nCreating datasets...")
    train_dataset = LocalPidginDataset(train_metadata, processor, Config.MAX_AUDIO_LENGTH)
    val_dataset   = LocalPidginDataset(val_metadata,   processor, Config.MAX_AUDIO_LENGTH)

    train_dataset = prefilter_dataset(train_dataset, 'train')
    val_dataset   = prefilter_dataset(val_dataset,   'dev')

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # =========================================================================
    # VALIDATE A FEW SAMPLES
    # =========================================================================

    print("\nValidating first few training samples...")
    good, bad = 0, 0
    for i in range(min(20, len(train_dataset))):
        s = train_dataset[i]
        if s is None:
            bad += 1
            continue
        good += 1
        if good <= 5:
            print(f"\n  Sample {i}:")
            print(f"    wav_path        : {train_dataset.pairs[i]['wav_path']}")
            print(f"    input_features  : {s['input_features'].shape}  "
                  f"min={s['input_features'].min():.3f}  max={s['input_features'].max():.3f}")
            print(f"    attention_mask  : {s['attention_mask'].shape}")
            print(f"    labels          : {s['labels'].shape}  "
                  f"min={s['labels'].min()}  max={s['labels'].max()}")
            print(f"    NaN in features : {torch.isnan(s['input_features']).any()}")

    print(f"\n  Good samples: {good}/{good+bad}")

    if train_dataset._rejection_counts:
        print("\n  Rejection reasons:")
        for reason, count in sorted(train_dataset._rejection_counts.items(), key=lambda x: -x[1]):
            print(f"    {count:5d}x  {reason}")
    else:
        print(f"  No rejections in first {good+bad} samples ✓")

    samples = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
    samples = [s for s in samples if s is not None]
    if samples:
        batch = data_collator(samples)
        print(f"\n  Collated batch:")
        print(f"    input_features : {batch['input_features'].shape}")
        print(f"    attention_mask : {batch['attention_mask'].shape}")
        print(f"    labels         : {batch['labels'].shape}")
        print(f"    Non-(-100) labels: {(batch['labels'] != -100).sum().item()}")
        print(f"    NaN in batch   : {torch.isnan(batch['input_features']).any().item()}")

    # =========================================================================
    # TRAINING ARGS
    # =========================================================================

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = Config.OUTPUT_DIR,
        per_device_train_batch_size = Config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size  = Config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate               = Config.LEARNING_RATE,
        warmup_steps                = Config.WARMUP_STEPS,
        num_train_epochs            = Config.NUM_TRAIN_EPOCHS,
        eval_strategy               = "steps",
        eval_steps                  = Config.EVAL_STEPS,
        save_steps                  = Config.SAVE_STEPS,
        logging_steps               = Config.LOGGING_STEPS,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        fp16                        = False,
        bf16                        = Config.BF16,
        gradient_checkpointing      = False if args.lora else Config.GRADIENT_CHECKPOINTING,
        predict_with_generate       = False,
        report_to                   = "tensorboard" if not args.no_tensorboard else "none",
        dataloader_num_workers      = Config.DATALOADER_NUM_WORKERS,
        dataloader_pin_memory       = Config.DATALOADER_NUM_WORKERS > 0,
        remove_unused_columns       = False,
        max_grad_norm               = 1.0,
        optim                       = "adamw_torch",
        save_strategy               = "steps",
        save_only_model             = True,
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        seed                        = 42,
        data_seed                   = 42,
        group_by_length             = False,
        ignore_data_skip            = True,
    )

    # =========================================================================
    # TRAINER
    # =========================================================================

    trainer = Seq2SeqTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        data_collator    = data_collator,
        processing_class = processor.tokenizer,
        callbacks        = [NaNLossCallback()],
    )

    # =========================================================================
    # TRAIN
    # =========================================================================

    print("\nStarting training...")
    print(f"  Epochs      : {Config.NUM_TRAIN_EPOCHS}")
    print(f"  Train pairs : {len(train_dataset)}")
    print(f"  Val pairs   : {len(val_dataset)}")
    print("=" * 70 + "\n")
    sys.stdout.flush()

    trainer.train(resume_from_checkpoint=args.resume)

    # =========================================================================
    # SAVE
    # =========================================================================

    print("\nSaving model...")
    final_dir = f"{Config.OUTPUT_DIR}/final"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"✓ Saved to: {final_dir}")

    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Model saved to: {final_dir}")


if __name__ == "__main__":
    main()