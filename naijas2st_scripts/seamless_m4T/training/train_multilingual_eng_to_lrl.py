"""
SeamlessM4T Fine-Tuning Script
Configured for English -> Low-Resource Language (Yoruba, Hausa, Igbo) translation.
Supports both STT (Speech-to-Text) and STS (Speech-to-Speech) tasks.
"""

import argparse
import os
import sys
import torch
from pathlib import Path
import librosa
import random


def main():
    """Parse CLI args and run the English -> LRL SeamlessM4T training loop.

    Workflow:
        1. Configure HuggingFace cache directories under
           ``SCRATCH_CACHE`` so the model and dataset cache live on
           scratch storage.
        2. Parse CLI args (task, model dir, epochs, batch sizes,
           grad-accum, learning rate, warmup/eval/save/log steps,
           max audio length, resume, num workers, gradient
           checkpointing, force reprocess, base model, LoRA flags,
           tensorboard).
        3. Print an effective-config banner.
        4. Define the ``Config`` wrapper around the parsed args.
        5. Build (or load cached) per-split pair metadata via
           ``create_pair_metadata``: the metadata is keyed by base ID
           and pairs an English source audio with a target-language
           audio + text for Yoruba/Hausa/Igbo.
        6. Load the base ``SeamlessM4Tv2`` model + processor
           (optionally LoRA-wrapped).
        7. Construct ``OnDemandPairedDataset`` instances and apply
           :func:`prefilter_dataset` to drop unusable pairs.
        8. Validate the first few samples and a collated batch.
        9. Build ``Seq2SeqTrainingArguments`` (bf16, gradient
           checkpointing optionally on, ``adamw_torch``, linear LR
           schedule).
        10. Instantiate ``Seq2SeqTrainer`` (or ``STSTrainer`` for
            STS), train with the NaN-loss callback, save the final
            model to ``<output_dir>/final``.

    Outputs:
        Checkpoints under ``--model-dir`` and a final saved model.

    Returns:
        None.
    """
    print('parsing arguments...')
    sys.stdout.flush()

    scratch_cache = os.environ.get("SCRATCH_CACHE", "./hf_cache")
    os.makedirs(scratch_cache, exist_ok=True)

    os.environ['HF_HOME'] = scratch_cache
    os.environ['HF_DATASETS_CACHE'] = f"{scratch_cache}/datasets"
    os.environ['HUGGINGFACE_HUB_CACHE'] = f"{scratch_cache}/hub"
    os.environ['HF_DATASETS_OFFLINE'] = '0'
    os.environ['XET_CACHE_DIR'] = f"{scratch_cache}/xet"
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    print(f"HuggingFace cache directory: {scratch_cache}")
    sys.stdout.flush()

    parser = argparse.ArgumentParser(
        description='Quick Start Training for SeamlessM4T (English -> Low-Resource)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start_train.py --task stt --model-dir ./my-model --epochs 5
  python quick_start_train.py --task sts --model-dir ./my-model --epochs 10
  python quick_start_train.py --task stt --model-dir ./my-model --resume ./my-model/checkpoint-1000
        """
    )

    parser.add_argument('--task',           type=str, required=True, choices=['stt', 'sts'])
    parser.add_argument('--model-dir',      type=str, required=True)
    parser.add_argument('--epochs',         type=int,   default=3)
    parser.add_argument('--batch-size',     type=int,   default=2)
    parser.add_argument('--grad-accum',     type=int,   default=8)
    parser.add_argument('--lr',             type=float, default=3e-5)
    parser.add_argument('--warmup-steps',   type=int,   default=500)
    parser.add_argument('--max-audio-length', type=int, default=30)
    parser.add_argument('--save-steps',     type=int,   default=1000)
    parser.add_argument('--eval-steps',     type=int,   default=500)
    parser.add_argument('--logging-steps',  type=int,   default=50)
    parser.add_argument('--resume',         type=str,   default=None)
    parser.add_argument('--num-workers',    type=int,   default=0)
    parser.add_argument('--no-fp16',        action='store_true')
    parser.add_argument('--no-gradient-checkpointing', action='store_true')
    parser.add_argument('--metadata-dir',   type=str,   default='./pair_metadata')
    parser.add_argument('--force-reprocess', action='store_true')
    parser.add_argument('--base-model',     type=str,   default='facebook/seamless-m4t-v2-large')
    parser.add_argument('--lora',           action='store_true')
    parser.add_argument('--lora-r',         type=int,   default=16)
    parser.add_argument('--no-tensorboard', action='store_true')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("SEAMLESSM4T FINE-TUNING - QUICK START")
    print("="*70)
    print(f"  Task:             {args.task.upper()}")
    print(f"  Base Model:       {args.base_model}")
    print(f"  Output Directory: {args.model_dir}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch Size:       {args.batch_size} x{args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print(f"  Learning Rate:    {args.lr}")
    print(f"  Num Workers:      {args.num_workers}")

    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n  WARNING: No GPU detected")


    import time
    try:
        for i in range(5, 0, -1):
            print(f"  Starting in {i}...", end='\r')
            time.sleep(1)
        print(" " * 30, end='\r')
    except KeyboardInterrupt:
        print("\n\nTraining cancelled.")
        return

    # ========================================================================
    # IMPORTS
    # ========================================================================

    from datasets import load_dataset
    from transformers import (
        SeamlessM4Tv2ForSpeechToText,
        SeamlessM4Tv2ForSpeechToSpeech,
        SeamlessM4TProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        TrainerCallback,
    )
    from dataclasses import dataclass
    from typing import Any, Dict, List
    from collections import defaultdict, Counter
    import gc
    import json
    # ========================================================================
    # CONFIG
    # ========================================================================

    class Config:
        """Global configuration mapping from CLI args to internal variables."""
        MODEL_NAME                  = args.base_model
        TASK                        = args.task
        OUTPUT_DIR                  = args.model_dir
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
        FP16                        = False          # disabled in favour of bf16
        BF16                        = True
        GRADIENT_CHECKPOINTING      = not args.no_gradient_checkpointing
        DATALOADER_NUM_WORKERS      = args.num_workers
        SEAMLESS_LANG_CODES = {
            'yoruba':  'swh',
            'hausa':   'arb', # seamless doesn't natively support hausa, but Somali is a close proxy in the same language family
            'igbo':    'swh',
            'english': 'eng',
        }

    # ========================================================================
    # PARSING HELPERS
    # ========================================================================

    def extract_base_id(text_id):
        """Drop the leading language-letter from a text ID.

        Args:
            text_id (str | None): Full text ID.

        Returns:
            str | None: Stripped ID or ``None``.
        """
        if not text_id or len(text_id) < 2:
            return None
        return text_id[1:]

    def get_language_from_text_id(text_id):
        """Map a text ID's first letter to the language name.

        Args:
            text_id (str | None): Text ID.

        Returns:
            str | None: Language name or ``None``.
        """
        if not text_id:
            return None
        lang_map = {'Y': 'yoruba', 'H': 'hausa', 'I': 'igbo', 'E': 'english'}
        return lang_map.get(text_id[0].upper())

    def get_english_accent_from_user_id(user_id):
        """Classify an English speaker as ``south``/``north``/``None``.

        Args:
            user_id (str | None): Speaker user ID.

        Returns:
            str | None: ``"south"``, ``"north"`` or ``None``.
        """
        if not user_id:
            return None
        if user_id.startswith('EY'):
            return 'south'
        if user_id.startswith('EN'):
            return 'north'
        return None

    def should_pair(african_lang, english_accent):
        """Return ``True`` when this African language should pair with this accent.

        Args:
            african_lang (str): African language name.
            english_accent (str | None): English accent.

        Returns:
            bool: Whether the pair is allowed.
        """
        if english_accent is None:
            return True
        if english_accent == 'south':
            return african_lang in ['yoruba', 'igbo']
        if english_accent == 'north':
            return african_lang == 'hausa'
        return False

    # ========================================================================
    # METADATA CREATION
    # ========================================================================

    def create_pair_metadata(split_name='train'):
        """
        Creates and caches metadata pairing English source audio with 
        Low-Resource (Yoruba, Hausa, Igbo) target text/audio.
        
        Args:
            split_name (str): The dataset split to process (e.g., 'train', 'dev').
            
        Returns:
            dict: Processed metadata containing dataset pair indices.
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

        print(f"\nLoading dataset (non-streaming)...")
        sys.stdout.flush()

        dataset = load_dataset(
            "McGill-NLP/NaijaS2ST",
            streaming=False,
            trust_remote_code=True,
            cache_dir=scratch_cache,
        )

        is_dev = (split_name == 'dev')
        buffer = defaultdict(lambda: {
            'yoruba': [], 'hausa': [], 'igbo': [],
            'english': [], 'english_south': [], 'english_north': [],
        })

        split_data  = dataset[split_name]
        total       = len(split_data)
        print(f"  Total examples in {split_name}: {total}")

        for idx in range(total):
            example  = split_data[idx]
            text_id  = example.get('text_id', '')
            user_id  = example.get('user_id', '')
            base_id  = extract_base_id(text_id)
            language = get_language_from_text_id(text_id)

            if not base_id or not language:
                continue

            if language == 'english':
                if is_dev:
                    accent     = get_english_accent_from_user_id(user_id)
                    buffer_key = f'english_{accent}' if accent else 'english'
                else:
                    buffer_key = 'english'
            else:
                buffer_key = language

            raw_text = example.get('transcription', example.get('text', ''))
            text     = raw_text[0] if isinstance(raw_text, list) else raw_text

            buffer[base_id][buffer_key].append({
                'idx':     idx,
                'text_id': text_id,
                'text':    text,
            })

            if idx % 5000 == 0:
                print(f"  Indexed {idx}/{total} examples, {len(buffer)} base_ids...")

        print(f"\nIndexed {total} examples across {len(buffer)} base_ids")

        # Create pairs: English -> African Language
        pair_metadata = []
        for base_id, langs in buffer.items():
            for african_lang in ['igbo']:
            # for african_lang in ['yoruba', 'hausa', 'igbo']:
                african_items = langs[african_lang]
                if not african_items:
                    continue

                if is_dev:
                    english_items = []
                    if should_pair(african_lang, 'south'):
                        english_items.extend(langs['english_south'])
                    if should_pair(african_lang, 'north'):
                        english_items.extend(langs['english_north'])
                else:
                    english_items = langs['english']

                if not english_items:
                    continue

                MAX_PAIRS_PER_ITEM = 2   # keeps balance, reduces dominance

                # Iterate through pairs, but swap assigning English to src and African to tgt
                for african_item in african_items:
                    sampled_english = english_items
                    if len(english_items) > MAX_PAIRS_PER_ITEM:
                        sampled_english = random.sample(english_items, MAX_PAIRS_PER_ITEM)

                    for english_item in sampled_english:
                        pair_metadata.append({
                            'src_idx':     english_item['idx'],
                            'src_text_id': english_item['text_id'],
                            'src_text':    english_item['text'],
                            'src_lang':    'english',
                            'tgt_idx':     african_item['idx'],
                            'tgt_text_id': african_item['text_id'],
                            'tgt_text':    african_item['text'],
                            'tgt_lang':    african_lang,
                        })

        lang_counts = Counter(p['tgt_lang'] for p in pair_metadata)
        print(f"\n✓ Created {len(pair_metadata)} pairs:")
        for lang, count in sorted(lang_counts.items()):
            print(f"    english → {lang}: {count}")

        os.makedirs(Config.METADATA_DIR, exist_ok=True)
        metadata = {'split': split_name, 'num_pairs': len(pair_metadata), 'pairs': pair_metadata}
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        print(f"✓ Saved metadata to {metadata_file}")

        del buffer
        gc.collect()
        return metadata

    # ========================================================================
    # DATASET
    # ========================================================================

    class OnDemandPairedDataset(torch.utils.data.Dataset):
        """
        PyTorch Dataset that lazy-loads audio only when requested via __getitem__.
        Pre-processes pairs for either Speech-to-Text or Speech-to-Speech mapping.
        """

        def __init__(self, metadata, processor, split_name='train', max_audio_length=80):
            self.pairs            = metadata['pairs']
            self.processor        = processor
            self.split_name       = split_name
            self.max_audio_length = max_audio_length
            self.seamless_codes   = Config.SEAMLESS_LANG_CODES
            self._dataset_split   = None   # loaded once, shared
            self._error_count     = 0
            print(f"✓ Dataset '{split_name}': {len(self.pairs)} pairs")

        # ------------------------------------------------------------------
        # Lazy-load the split ONCE per process (workers get a forked copy)
        # ------------------------------------------------------------------
        def _get_split(self):
            if self._dataset_split is None:
                ds = load_dataset(
                    "McGill-NLP/NaijaS2ST",
                    streaming=False,
                    trust_remote_code=True,
                    cache_dir=scratch_cache,
                )
                self._dataset_split = ds[self.split_name]
            return self._dataset_split

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            pair = self.pairs[idx]
            try:
                split       = self._get_split()
                # Source is now English
                src_example = split[pair['src_idx']]
                src_audio   = src_example['audio']
                original_sr = src_audio['sampling_rate']
                
                lang_id = pair['src_lang']  # Now always 'english'
                tgt_lang_id = pair['tgt_lang'] # e.g. 'yoruba', 'hausa', 'igbo'
                
                # Resample to 16 kHz if needed
                if original_sr != 16000:
                    src_audio['array'] = librosa.resample(
                        src_audio['array'].astype('float32'),
                        orig_sr=original_sr,
                        target_sr=16000,
                    )
                    src_audio['sampling_rate'] = 16000

                # Duration is now always computed on the 16kHz array
                duration = len(src_audio['array']) / 16000

                if duration < 0.5:
                    self._log_rejection(idx,
                        f"audio too short ({duration:.2f}s) | "
                        f"text_id={pair.get('src_text_id','?')} | "
                        f"original_sr={original_sr}")
                    return None
                if duration > self.max_audio_length:
                    # Show original duration at original SR to expose SR bugs
                    orig_duration = len(src_audio['array']) / original_sr if original_sr != 16000 \
                                    else duration
                    self._log_rejection(idx,
                        f"audio too long ({duration:.2f}s, orig {orig_duration:.2f}s@{original_sr}Hz) | "
                        f"text_id={pair.get('src_text_id','?')}")
                    return None

                src_lang = self.seamless_codes[pair['src_lang']]
                tgt_lang_code = self.seamless_codes[pair['tgt_lang']]

                # --- Audio features ---
                inputs = self.processor(
                    audio=src_audio['array'],
                    sampling_rate=16000,
                    src_lang=src_lang,
                    return_tensors="pt",
                )
                input_features = inputs.input_features.squeeze(0)   # [T, 160]
                attention_mask = inputs.attention_mask.squeeze(0)    # [T]

                if torch.isnan(input_features).any() or torch.isinf(input_features).any():
                    self._log_rejection(idx, 'NaN/Inf in input_features')
                    return None

                # --- Text labels ---
                if Config.TASK == 'stt':
                    tgt_text = pair['tgt_text'].strip()
                    if not tgt_text:
                        self._log_rejection(idx, 'empty tgt_text')
                        return None

                    # explicit target language conditioning based on the African target language
                    conditioned_text = f"<{tgt_lang_code}> {tgt_text}"

                    labels = self.processor.tokenizer(
                        text=conditioned_text,
                        padding=False,
                        truncation=True,
                        max_length=224,
                        return_tensors="pt",
                    ).input_ids.squeeze(0)
                    
                    if labels.shape[0] == 0:
                        self._log_rejection(idx, 'empty labels')
                        return None

                    return {
                        'input_features': input_features,
                        'attention_mask': attention_mask,
                        'labels':         labels,
                        'src_lang':       lang_id,
                        'tgt_lang':       tgt_lang_id,
                    }
                elif Config.TASK == 'sts':
                    # -------------------------------------------------------
                    # TARGET AUDIO (STS supervision)
                    # -------------------------------------------------------
                    tgt_example = split[pair['tgt_idx']]
                    tgt_audio   = tgt_example['audio']
                    tgt_sr      = tgt_audio['sampling_rate']

                    if tgt_sr != 16000:
                        tgt_audio['array'] = librosa.resample(
                            tgt_audio['array'].astype('float32'),
                            orig_sr=tgt_sr,
                            target_sr=16000,
                        )
                        tgt_audio['sampling_rate'] = 16000

                    # The processor handles speech input for STS automatically.
                    # Tell processor the actual language of the target audio labels.
                    labels = self.processor(
                        audio=tgt_audio['array'],
                        sampling_rate=16000,
                        src_lang=tgt_lang_code, 
                        return_tensors="pt",
                    )

                    speech_labels = labels.input_features.squeeze(0)

                    if torch.isnan(speech_labels).any() or torch.isinf(speech_labels).any():
                        self._log_rejection(idx, "NaN/Inf in target speech labels")
                        return None
                    return {
                        'input_features': input_features,
                        'attention_mask': attention_mask,
                        'labels':         speech_labels,
                    }

            except Exception as e:
                self._error_count += 1
                if self._error_count <= 20:
                    import traceback
                    print(f" Exception at pair {idx}: {e}")
                    print(f"     tgt_text: '{pair.get('tgt_text', '')[:80]}'")
                    print(f"     tgt_lang: {pair.get('tgt_lang', '?')}")
                    traceback.print_exc()
                return None

        def _log_rejection(self, idx, reason):
            """Log first 5 of each unique rejection reason."""
            if not hasattr(self, '_rejection_counts'):
                self._rejection_counts = {}
            self._rejection_counts[reason] = self._rejection_counts.get(reason, 0) + 1
            if self._rejection_counts[reason] <= 5:
                print(f"  ↩ Pair {idx} rejected: {reason}")


    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        """
        Dynamically pads batched inputs and labels based on the max sequence 
        length present in a given batch to optimize compute.
        """
        processor: Any
        task: str = 'stt'  # 'stt' or 'sts'

        def __call__(self, features: List[Dict]):
            features = [f for f in features if f is not None]

            if not features:
                # dummy batch to avoid Trainer crash
                return {
                    'input_features': torch.zeros(1, 100, 160),
                    'attention_mask': torch.zeros(1, 100, dtype=torch.long),
                    'labels': torch.zeros(1, 100, 160) if self.task=='sts' else torch.full((1, 1), -100, dtype=torch.long),
                }

            # ---- pad input features ----
            input_feats = [f['input_features'] for f in features]
            attn_masks  = [f['attention_mask'] for f in features]

            max_T = max(f.shape[0] for f in input_feats)
            padded_feats = []
            padded_masks = []

            for feat, mask in zip(input_feats, attn_masks):
                pad = max_T - feat.shape[0]
                padded_feats.append(
                    torch.nn.functional.pad(feat, (0,0,0,pad)) if pad>0 else feat
                )
                padded_masks.append(
                    torch.nn.functional.pad(mask, (0,pad), value=0) if pad>0 else mask
                )

            batch = {
                'input_features': torch.stack(padded_feats),
                'attention_mask': torch.stack(padded_masks),
            }

            # ---- pad labels ----
            if self.task == 'stt':
                labels_list = [f['labels'] for f in features]
                max_L = max(l.shape[0] for l in labels_list)
                padded_labels = []
                for l in labels_list:
                    pad = max_L - l.shape[0]
                    padded_labels.append(
                        torch.nn.functional.pad(l, (0,pad), value=-100) if pad>0 else l
                    )
                batch['labels'] = torch.stack(padded_labels)
            elif self.task == 'sts':
                # labels are audio features [T', 160] → pad to max_T_label
                labels_list = [f['labels'] for f in features]
                max_T_label = max(l.shape[0] for l in labels_list)
                padded_labels = []
                for l in labels_list:
                    pad = max_T_label - l.shape[0]
                    padded_labels.append(
                        torch.nn.functional.pad(l, (0,0,0,pad)) if pad>0 else l
                    )
                batch['labels'] = torch.stack(padded_labels)  # [B, T_max, 160]

            batch.pop('src_lang', None)
            batch.pop('tgt_lang', None)

            return batch
        
    # ========================================================================
    # CALLBACK: print a warning whenever loss is NaN/Inf
    # ========================================================================

    class NaNLossCallback(TrainerCallback):
        """Detect NaN/Inf losses early and print diagnostics."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            for key in ('loss', 'eval_loss'):
                val = logs.get(key)
                if val is not None and (val != val or val == float('inf')):   # NaN or Inf
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

    # ========================================================================
    # BUILD METADATA
    # ========================================================================

    print("Creating/loading pair metadata")
    train_metadata = create_pair_metadata('train')
    val_metadata   = create_pair_metadata('dev')

    # ========================================================================
    # LOAD MODEL
    # ========================================================================

    print("Loading model and processor")
    processor = SeamlessM4TProcessor.from_pretrained(Config.MODEL_NAME)

    ModelClass = SeamlessM4Tv2ForSpeechToText if Config.TASK == 'stt' \
                 else SeamlessM4Tv2ForSpeechToSpeech

    # After loading the model
    model = ModelClass.from_pretrained(Config.MODEL_NAME)
    
    if args.lora:
        print(f"Applying LoRA (rank={args.lora_r})...")
        print("Removed LoRA code to avoid new dependency issues. Uncomment following code for LoRA support")
        # from peft import LoraConfig, get_peft_model
        
        # # CRITICAL: Prepare model for LoRA BEFORE anything else
        # model.requires_grad_(True)  # Ensure gradients are enabled
        
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
        
        # # CRITICAL: Disable gradient checkpointing with LoRA (incompatible)
        # model.config.use_cache = False
        # if hasattr(model, 'gradient_checkpointing_enable'):
        #     # Don't enable - LoRA doesn't work well with it
        #     pass

    print("Model loaded")

    # ========================================================================
    # DATASETS
    # ========================================================================

    print("Creating datasets")
    train_dataset = OnDemandPairedDataset(train_metadata, processor, 'train',  Config.MAX_AUDIO_LENGTH)
    val_dataset   = OnDemandPairedDataset(val_metadata,   processor, 'dev',    Config.MAX_AUDIO_LENGTH)

    # -----------------------------------------------------------------------
    # Pre-filter: scan all pairs for bad audio duration using ONLY metadata
    # (no audio loading, just check array length vs sampling_rate from a
    #  quick streaming pass).  Removes the "entire batch is None" problem.
    # -----------------------------------------------------------------------
    def prefilter_dataset(dataset, label='train', scan_limit=None):
        """
        Remove pairs whose audio is definitely too long/short without
        loading the full audio array — just use the cached split index.
        Returns a new dataset with only valid pairs.
        """
        print(f"\nPre-filtering {label} dataset...")
        split     = dataset._get_split()
        valid_pairs = []
        rejected_too_long  = []   # (idx, text_id, duration)
        rejected_too_short = []
        rejected_error     = []

        total = len(dataset.pairs) if scan_limit is None else min(scan_limit, len(dataset.pairs))

        for i, pair in enumerate(dataset.pairs[:total]):
            if i % 5000 == 0:
                print(f"  Scanning {i}/{total}...", end='\r')
            try:
                example  = split[pair['src_idx']]
                audio    = example['audio']
                sr       = audio['sampling_rate']
                n        = len(audio['array'])
                duration = n / sr   # duration in seconds at original SR

                if duration < 0.5:
                    rejected_too_short.append((i, pair.get('src_text_id', '?'), duration, sr))
                elif duration > dataset.max_audio_length:
                    rejected_too_long.append((i, pair.get('src_text_id', '?'), duration, sr))
                else:
                    valid_pairs.append(pair)
            except Exception as e:
                rejected_error.append((i, pair.get('src_text_id', '?'), str(e)))

        total_rejected = len(rejected_too_long) + len(rejected_too_short) + len(rejected_error)
        print(f"  ✓ {len(valid_pairs)} valid, {total_rejected} rejected\n")

        # Report too-long
        if rejected_too_long:
            print(f"    Too long ({len(rejected_too_long)} pairs, max={dataset.max_audio_length}s):")
            for idx, text_id, dur, sr in rejected_too_long[:10]:
                print(f"      pair_idx={idx:6d}  text_id={text_id:15s}  "
                      f"duration={dur:.2f}s  sr={sr}Hz")
            if len(rejected_too_long) > 10:
                durations = [d for _, _, d, _ in rejected_too_long]
                print(f"      ... and {len(rejected_too_long)-10} more  "
                      f"(min={min(durations):.1f}s  max={max(durations):.1f}s  "
                      f"mean={sum(durations)/len(durations):.1f}s)")

        # Report too-short
        if rejected_too_short:
            print(f"\n    Too short ({len(rejected_too_short)} pairs):")
            for idx, text_id, dur, sr in rejected_too_short[:10]:
                print(f"      pair_idx={idx:6d}  text_id={text_id:15s}  "
                      f"duration={dur:.2f}s  sr={sr}Hz")

        # Report errors
        if rejected_error:
            print(f"\n    Errors ({len(rejected_error)} pairs):")
            for idx, text_id, err in rejected_error[:10]:
                print(f"      pair_idx={idx:6d}  text_id={text_id:15s}  error={err}")

        # Replace pairs in-place
        dataset.pairs    = valid_pairs
        dataset.metadata = {'split': label, 'num_pairs': len(valid_pairs), 'pairs': valid_pairs}
        return dataset

    train_dataset = prefilter_dataset(train_dataset, 'train')
    val_dataset   = prefilter_dataset(val_dataset,   'dev')
    
    if Config.TASK == 'stt':
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, task='stt')

    elif Config.TASK == 'sts':
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, task='sts')

    # ========================================================================
    # VALIDATE A FEW SAMPLES BEFORE TRAINING
    # ========================================================================

    print("\n Validating first 5 training samples...")
    good, bad = 0, 0
    for i in range(min(20, len(train_dataset))):
        s = train_dataset[i]
        if s is None:
            bad += 1
            continue
        good += 1
        if good <= 5:
            print(f"\n  Sample {i}:")
            print(f"    input_features: {s['input_features'].shape}  "
                  f"min={s['input_features'].min():.3f}  max={s['input_features'].max():.3f}")
            print(f"    attention_mask: {s['attention_mask'].shape}")
            print(f"    labels:         {s['labels'].shape}  "
                  f"min={s['labels'].min()}  max={s['labels'].max()}")
            has_nan = torch.isnan(s['input_features']).any()
            print(f"    NaN in features: {has_nan}")

    print(f"\n  Good samples: {good}/{good+bad}")

    # Print rejection reason breakdown
    if hasattr(train_dataset, '_rejection_counts') and train_dataset._rejection_counts:
        print(f"\n  Rejection reasons:")
        for reason, count in sorted(train_dataset._rejection_counts.items(),
                                    key=lambda x: -x[1]):
            print(f"    {count:5d}x  {reason}")
    else:
        print(f"  No rejections in first {good+bad} samples ✓")

    # Check a collated batch
    samples = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
    samples = [s for s in samples if s is not None]
    if samples:
        batch = data_collator(samples)
        if batch:
            print(f"\n Collated batch:")
            print(f"    input_features: {batch['input_features'].shape}")
            print(f"    attention_mask: {batch['attention_mask'].shape}")
            print(f"    labels:         {batch['labels'].shape}")
            print(f"    Non-(-100) labels: {(batch['labels'] != -100).sum().item()}")
            any_nan = torch.isnan(batch['input_features']).any().item()
            print(f"    NaN in batch:   {any_nan}")
        else:
            print("Collator returned None for test batch — check data")

    # ========================================================================
    # TRAINING ARGS
    # ========================================================================

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
        bf16                        = True,
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

    # ========================================================================
    # TRAINER
    # ========================================================================


    class STSTrainer(Seq2SeqTrainer):
        """
        Custom Seq2SeqTrainer instance to compute MSE loss directly on the 
        speech features when training on the Speech-to-Speech (STS) task.
        """
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """
            Custom loss for STS (speech-to-speech).
            inputs: dict with keys: input_features, attention_mask, labels
            labels: target speech features [B, T, F]
            """
            labels = inputs.pop("labels")  # [B, T, F]
            
            # inside compute_loss
            proj = model.text_decoder.embed_tokens  # or a linear layer from 160->1024
            decoder_inputs_embeds = proj(labels)    # convert to decoder hidden size
            outputs = model(
                input_features=inputs['input_features'],
                attention_mask=inputs['attention_mask'],
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=False
            )
            
            # predicted speech features
            preds = outputs.decoder_output  # [B, T, F]
            
            # Align lengths
            min_len = min(preds.shape[1], labels.shape[1])
            preds = preds[:, :min_len, :]
            labels = labels[:, :min_len, :]
            
            # L2 loss
            loss = torch.nn.functional.mse_loss(preds, labels)
            
            return (loss, outputs) if return_outputs else loss

    if Config.TASK == 'stt':
        trainer = Seq2SeqTrainer(
            model           = model,
            args            = training_args,
            train_dataset   = train_dataset,
            eval_dataset    = val_dataset,
            data_collator   = data_collator,
            processing_class = processor.tokenizer,
            callbacks       = [NaNLossCallback()],
        )

    elif Config.TASK == 'sts':
        trainer = STSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[NaNLossCallback()],
    )

    # ========================================================================
    # TRAIN
    # ========================================================================

    print("\n Starting training...")
    print(f"  Epochs:      {Config.NUM_TRAIN_EPOCHS}")
    print(f"  Train pairs: {len(train_dataset)}")
    print(f"  Val pairs:   {len(val_dataset)}")
    print("="*70 + "\n")
    sys.stdout.flush()

    trainer.train(resume_from_checkpoint=args.resume)

    # ========================================================================
    # SAVE
    # ========================================================================

    print("\n Saving model...")
    final_dir = f"{Config.OUTPUT_DIR}/final"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"✓ Saved to: {final_dir}")

    print("\n" + "="*70)
    print(" TRAINING COMPLETE")
    print("="*70)
    print(f"\n Model saved to: {final_dir}")

if __name__ == "__main__":
    main()