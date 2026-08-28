"""Batch SeamlessM4T English audio -> LRL text inference using a custom trainer.

Counterpart to ``batch_inference_custom_trainer.py`` in the English ->
LRL direction. Picks the appropriate English accent test set per
target LRL and translates each wav into the target language.
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
    "hausa": "som", # som for Somalian, as SeamlessM4T does not natively support Hausa and is relatively similar linguistically
    "igbo": "ibo", 
    "english": "eng"
}

accent_dictionary = {"yoruba": "english_south_accent", "igbo":"english_south_accent", "hausa": "english_north_accent"}


def load_audio(path, target_sr=16000):
    """Load a wav file as mono and resample to ``target_sr``.

    Args:
        path (str | os.PathLike): Path to the audio file.
        target_sr (int): Desired sample rate in Hz.

    Returns:
        tuple[numpy.ndarray, int] | tuple[None, None]:
        ``(audio_array, sample_rate)`` on success, or
        ``(None, None)`` on failure.
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

def process_directory(model, processor, audio_dir, tgt_lang, device):
    """Translate every wav under ``audio_dir`` into ``tgt_lang`` text.

    Args:
        model (transformers.SeamlessM4Tv2ForSpeechToText): Loaded model.
        processor (transformers.SeamlessM4TProcessor): Matching processor.
        audio_dir (str | os.PathLike): Directory containing English
            wavs (searched recursively).
        tgt_lang (str): Target LRL name; used to pick the SeamlessM4T
            language code from ``SEAMLESS_LANG_CODES``.
        device (str | torch.device): Torch device string.

    Returns:
        list[dict]: One dict per processed wav with keys
        ``file_name``/``file_path``/``source_language``/
        ``target_language``/``translation``/``duration``, plus
        ``error`` on failures.
    """
    audio_path = Path(audio_dir)
    
    if not audio_path.exists():
        print(f"Directory not found: {audio_dir}")
        return []
    
    # Find all wav files recursively
    wav_files = sorted(audio_path.glob("**/*.wav"))
    
    if not wav_files:
        print(f" No wav files found in {audio_dir}")
        return []
    
    print(f"\n Processing translations to {tgt_lang.upper()} - Found {len(wav_files)} files")
    
    results = []
    tgt_lang_code = SEAMLESS_LANG_CODES[tgt_lang]
    
    # For speech-to-text, source is always English here
    src_lang_code = SEAMLESS_LANG_CODES["english"] 
    
    for wav_file in tqdm(wav_files, desc=f"Translating to {tgt_lang}"):
        # Load audio
        audio, sr = load_audio(str(wav_file))
        if audio is None:
            continue
        
        try:
            # Process audio (source is English)
            inputs = processor(
                audios=audio,
                sampling_rate=sr,
                return_tensors="pt",
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    tgt_lang=tgt_lang_code,
                    max_new_tokens=256,
                    num_beams=5,               # Use beam search instead of greedy decoding
                    do_sample=False,           # Keep it deterministic, but explore more paths
                    repetition_penalty=1.2,    # Help prevent it from getting stuck in loops
                    # Optional: You can force it to not output English by suppressing English tokens, 
                    # but Beam Search is usually enough to fix this specific leakage.
                )
            
            # Decode text
            text = processor.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
            
            results.append({
                "file_name": wav_file.name,
                "file_path": str(wav_file),
                "source_language": "english",
                "target_language": tgt_lang,
                "translation": text,
                "duration": len(audio) / sr,
            })
            
        except Exception as e:
            print(f"Error processing {wav_file.name}: {e}")
            results.append({
                "file_name": wav_file.name,
                "file_path": str(wav_file),
                "source_language": "english",
                "target_language": tgt_lang,
                "translation": "",
                "error": str(e),
            })
    
    return results

def main():
    """Parse CLI args and run English -> LRL SeamlessM4T batch inference.

    Workflow:
        1. Parse CLI args (``--task`` only supports ``stt``,
           ``--model-path``, ``--data-dir`` for English wavs,
           ``--output-dir``, ``--device``, ``--target-languages``,
           ``--zero-shot``).
        2. Load ``SeamlessM4TProcessor`` +
           ``SeamlessM4Tv2ForSpeechToText`` from ``--model-path`` and
           place the model on the chosen device in eval mode.
        3. For each target language, call :func:`process_directory`
           passing the English audio dir and the target language;
           SeamlessM4T language codes come from
           ``SEAMLESS_LANG_CODES`` (Hausa is mapped to Somali
           ``"som"`` since SeamlessM4T does not natively support
           Hausa).
        4. Write per-language and combined results JSONs to
           ``--output-dir``.

    Notes:
        ``accent_map`` is defined alongside but currently unused; the
        script translates whatever English audio is supplied without
        special accent routing.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser("SeamlessM4T Batch Inference (English to African Languages)")
    parser.add_argument("--task", required=True, choices=["stt"],
                        help="Task: speech-to-text")
    parser.add_argument("--model-path", default="facebook/seamless-m4t-v2-large",
                        help="Model path or HuggingFace model ID")
    parser.add_argument("--data-dir", required=True,
                        help="Root directory containing English wav files")
    parser.add_argument("--output-dir", default="./batch_results",
                        help="Output directory for JSON results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--target-languages", default="yoruba,hausa,igbo",
                        help="Comma-separated list of target languages to translate to")
    parser.add_argument("--zero-shot", action="store_true", default=True,
                        help="Use zero-shot inference")
    
    args = parser.parse_args()
    
    target_languages = [lang.strip().lower() for lang in args.target_languages.split(",")]
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("SeamlessM4T Batch Inference (Eng -> African)")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Input English Audio: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target Languages: {', '.join(target_languages)}")
    print(f"Device: {args.device}")
    print("="*70)
    
    # Load model and processor
    print("\nLoading model...")
    processor = SeamlessM4TProcessor.from_pretrained(args.model_path)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()
    print("✓ Model loaded")
    
    # Process English audio into each target language
    all_results = {}
    for tgt_lang in target_languages:
        if tgt_lang not in SEAMLESS_LANG_CODES:
            print(f"Warning: {tgt_lang} not found in SEAMLESS_LANG_CODES. Skipping.")
            continue
        english_accent = accent_dictionary[tgt_lang]
        accent_data_dir = data_dir / english_accent

        results = process_directory(
            model=model,
            processor=processor,
            audio_dir=str(accent_data_dir),
            tgt_lang=tgt_lang,
            device=args.device,
        )
        
        all_results[tgt_lang] = results
        
        # Save results for this language
        output_file = output_dir / f"eng_to_{tgt_lang}_translations.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved {len(results)} {tgt_lang} translations to {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    total_files = sum(len(results) for results in all_results.values())
    print(f"Total translations generated: {total_files}")
    
    for language, results in all_results.items():
        success_count = sum(1 for r in results if "error" not in r)
        error_count = sum(1 for r in results if "error" in r)
        print(f"\nENG -> {language.upper()}:")
        print(f"  Success: {success_count}/{len(results)}")
        if error_count > 0:
            print(f"  Errors: {error_count}/{len(results)}")
    
    # Save combined results
    combined_file = output_dir / "all_eng_to_african_translations.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n Saved combined results to {combined_file}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()