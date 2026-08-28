"""Batch SeamlessM4T LRL audio -> English text inference using a custom trainer.

CLI script that walks ``<data-dir>/<language>/<user>/recordings/*.wav``
for each configured language, runs zero-shot SeamlessM4T v2 large for
speech-to-text translation into English, and saves per-language plus a
combined translations JSON.
"""

import argparse
import torch
import librosa
import json
from pathlib import Path
from tqdm import tqdm

from transformers import (
    SeamlessM4TProcessor,
    SeamlessM4Tv2ForSpeechToText,
)

SEAMLESS_LANG_CODES = {
    "yoruba": "yor",
    "hausa": "hau",
    "igbo": "ibo",
    "english": "eng",
}

def load_audio(path, target_sr=16000):
    """Load a wav file as mono and resample to ``target_sr``.

    Args:
        path (str | os.PathLike): Path to the audio file.
        target_sr (int): Desired sample rate in Hz.

    Returns:
        tuple[numpy.ndarray, int] | tuple[None, None]: ``(audio_array,
        sample_rate)`` on success, or ``(None, None)`` if the file
        could not be loaded.
    """
    try:
        audio, sr = librosa.load(path, sr=None, mono=True)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return audio, sr
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None

def process_directory(model, processor, audio_dir, src_lang, device):
    """Run SeamlessM4T speech-to-text over every wav under one language directory.

    Args:
        model (transformers.SeamlessM4Tv2ForSpeechToText): Loaded model.
        processor (transformers.SeamlessM4TProcessor): Matching processor.
        audio_dir (str | os.PathLike): Root directory; either
            ``*.wav`` directly or ``<user>/recordings/*.wav``.
        src_lang (str): Source language name (used to look up the
            SeamlessM4T code in ``SEAMLESS_LANG_CODES``).
        device (str | torch.device): Torch device string
            (``"cuda"`` or ``"cpu"``).

    Returns:
        list[dict]: One dict per processed wav with keys
        ``file_name``/``file_path``/``language``/``translation``/
        ``duration``, plus ``error`` on failures.
    """
    audio_path = Path(audio_dir)
    
    if not audio_path.exists():
        print(f"Directory not found: {audio_dir}")
        return []
    
    # Find all wav files recursively in recordings subdirectories
    wav_files = sorted(audio_path.glob("*/recordings/*.wav"))
    
    if not wav_files:
        # Fallback: try direct wav files
        wav_files = sorted(audio_path.glob("*.wav"))
    
    if not wav_files:
        print(f" No wav files found in {audio_dir}")
        return []
    
    print(f"\n Processing {src_lang.upper()} - Found {len(wav_files)} files")
    
    results = []
    src_lang_code = SEAMLESS_LANG_CODES[src_lang]
    
    for wav_file in tqdm(wav_files, desc=f"Processing {src_lang}"):
        # Load audio
        audio, sr = load_audio(str(wav_file))
        if audio is None:
            continue
        
        try:
            # Process audio
            inputs = processor(
                audio=audio,
                sampling_rate=sr,
                src_lang=src_lang_code,
                return_tensors="pt",
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    tgt_lang=SEAMLESS_LANG_CODES["english"],
                    max_new_tokens=256,
                )
            
            # Decode text
            text = processor.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
            
            results.append({
                "file_name": wav_file.name,
                "file_path": str(wav_file),
                "language": src_lang,
                "translation": text,
                "duration": len(audio) / sr,
            })
            
        except Exception as e:
            print(f"Error processing {wav_file.name}: {e}")
            results.append({
                "file_name": wav_file.name,
                "file_path": str(wav_file),
                "language": src_lang,
                "translation": "",
                "error": str(e),
            })
    
    return results

def main():
    """Parse CLI args and run SeamlessM4T batch STT for each requested language.

    Workflow:
        1. Parse CLI args: ``--task`` (only ``stt`` is currently
           supported), ``--model-path``, ``--data-dir``,
           ``--output-dir``, ``--device``, ``--languages``,
           ``--zero-shot``.
        2. Load ``SeamlessM4TProcessor`` and
           ``SeamlessM4Tv2ForSpeechToText`` from ``--model-path`` and
           move the model to the chosen device (in eval mode).
        3. For each requested language, call
           :func:`process_directory` on ``<data-dir>/<language>``;
           the helper returns a list of
           ``{file_name, file_path, language, translation, duration}``
           records (with ``error`` set on failures).
        4. Save per-language results to
           ``<output-dir>/<language>_translations.json`` and add the
           list into ``all_results[language]``.
        5. Print a summary table (success vs. error counts) and write
           ``<output-dir>/all_translations.json`` with the combined
           results.

    Inputs:
        Nested ``<language>/<user>/recordings/*.wav`` structure.

    Outputs:
        Per-language plus combined translation JSONs.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser("SeamlessM4T Batch Inference (Zero-Shot)")
    parser.add_argument("--task", required=True, choices=["stt", "sts"],
                        help="Task: speech-to-text or speech-to-speech")
    parser.add_argument("--model-path", default="facebook/seamless-m4t-v2-large",
                        help="Model path or HuggingFace model ID (default: facebook/seamless-m4t-v2-large)")
    parser.add_argument("--data-dir", required=True,
                        help="Root directory containing language subdirectories (yoruba, hausa, igbo)")
    parser.add_argument("--output-dir", default="./batch_results",
                        help="Output directory for JSON results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--languages", default="yoruba,hausa,igbo",
                        help="Comma-separated list of languages to process")
    parser.add_argument("--zero-shot", action="store_true", default=True,
                        help="Use zero-shot inference (no fine-tuning)")
    
    args = parser.parse_args()
    
    # Validate
    if args.task != "stt":
        print("Currently only STT (speech-to-text) is supported")
        return
    
    languages = [lang.strip() for lang in args.languages.split(",")]
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("SeamlessM4T Batch Inference (Zero-Shot)")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Languages: {', '.join(languages)}")
    print(f"Device: {args.device}")
    print("="*70)
    
    # Load model and processor
    print("\nLoading model...")
    processor = SeamlessM4TProcessor.from_pretrained(args.model_path)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()
    print("Model loaded")
    
    # Process each language
    all_results = {}
    for language in languages:
        lang_dir = data_dir / language
        
        results = process_directory(
            model=model,
            processor=processor,
            audio_dir=str(lang_dir),
            src_lang=language,
            device=args.device,
        )
        
        all_results[language] = results
        
        # Save results for this language
        output_file = output_dir / f"{language}_translations.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(results)} results to {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    total_files = sum(len(results) for results in all_results.values())
    print(f"Total files processed: {total_files}")
    
    for language, results in all_results.items():
        success_count = sum(1 for r in results if "error" not in r)
        error_count = sum(1 for r in results if "error" in r)
        print(f"\n{language.upper()}:")
        print(f"  Success: {success_count}/{len(results)}")
        if error_count > 0:
            print(f" Errors: {error_count}/{len(results)}")
    
    # Save combined results
    combined_file = output_dir / "all_translations.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n Saved combined results to {combined_file}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
