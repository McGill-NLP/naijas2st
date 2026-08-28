"""SeamlessM4T LRL audio -> English speech (S2ST) batch inference.

Loads a (possibly fine-tuned) SeamlessM4T v2 checkpoint via the
``seamless_communication`` ``Translator`` API, walks the local test
set per language, runs S2ST sequentially per file (to avoid OOM), and
writes the translated wavs plus a results JSON per language.
"""

import os
import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from seamless_communication.inference import Translator

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "facebook/seamless-m4t-v2-large"

# Base directories (script will append language names to these)
BASE_INPUT_DIR = "naijas2st_scripts/test"
BASE_OUTPUT_DIR = "RESULTS/naijas2st/seamless_lrl_to_eng_zero_shot_v2"

BATCH_SIZE = 10  # Adjust based on your GPU VRAM
TGT_LANG = "eng"
TASK = "s2st"

# Language mapping: Folder Name -> Seamless Code
LANGUAGES = {
    # "hausa": "arb",
    "yoruba": "yor",
    "igbo": "ibo"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_custom_translator(model_path, device):
    """Load a SeamlessM4T checkpoint and patch the state dict layout.

    Maps custom-trainer key prefixes (``t2u_encoder.``, ``t2u_decoder.``,
    ``t2u_final_proj.weight``) onto the layout expected by the upstream
    ``Translator`` model.

    Args:
        model_path (str | os.PathLike): Path to a torch ``.pt``
            checkpoint or HF model card.
        device (torch.device | str): Torch device.

    Returns:
        seamless_communication.Translator: Initialised ``Translator``
        in eval mode.
    """
    print("\n Loading model checkpoint...")
    ckpt = torch.load(model_path, map_location=device)
    model_name = ckpt["model_name"] 

    print(f"Initializing Translator for {model_name}...")
    translator = Translator(
        model_name_or_card=model_name, 
        vocoder_name_or_card="vocoder_36langs", 
        device=device
    )

    raw_state_dict = ckpt["model"]
    clean_state_dict = {}

    for key, value in raw_state_dict.items():
        new_key = key.replace("model.", "")
        if new_key.startswith("t2u_encoder."):
            new_key = new_key.replace("t2u_encoder.", "t2u_model.encoder.")
        elif new_key.startswith("t2u_decoder."):
            new_key = new_key.replace("t2u_decoder.", "t2u_model.decoder.")
        elif new_key.startswith("t2u_decoder_frontend."):
            new_key = new_key.replace("t2u_decoder_frontend.", "t2u_model.decoder_frontend.")
        elif new_key == "t2u_final_proj.weight":
            new_key = "t2u_model.final_proj.weight"
            
        clean_state_dict[new_key] = value

    msg = translator.model.load_state_dict(clean_state_dict, strict=False)
    print(f"✓ State dict loaded successfully!")
    translator.model.eval() 
    return translator


def process_language(translator, lang_name, lang_code):
    """Translate every wav for one language and save audios + metadata.

    Args:
        translator (seamless_communication.Translator): A loaded
            SeamlessM4T ``Translator``.
        lang_name (str): Folder name under ``BASE_INPUT_DIR`` to
            process.
        lang_code (str): SeamlessM4T language code (e.g. ``"yor"``).

    Returns:
        None. Writes one translated wav per input plus a results JSON.
    """
    input_dir = Path(BASE_INPUT_DIR) / lang_name
    output_audio_dir = Path(BASE_OUTPUT_DIR) / lang_name / "audio"
    output_json_path = Path(BASE_OUTPUT_DIR) / lang_name / f"{lang_name}_results.json"
    
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Find Audio Files
    wav_files = sorted(list(input_dir.glob("**/*.wav"))) 
    
    if not wav_files:
        print(f"\n No .wav files found in {input_dir}. Skipping {lang_name.upper()}...")
        return
        
    print(f"\n" + "="*50)
    print(f"🌍 Processing {lang_name.upper()} ({lang_code}) - {len(wav_files)} files")
    print("="*50)

    # 3. Sequential Processing Loop
    all_results = []

    for file_path in tqdm(wav_files, desc=f"Translating {lang_name}"):
        try:
            # Predict takes a SINGLE string file path
            out_texts, out_audios = translator.predict(
                input=str(file_path),
                task_str=TASK,
                src_lang=lang_code, 
                tgt_lang=TGT_LANG
            )
            
            # 4. Unpack and Save Results
            translated_text = str(out_texts[0])
            
            # Extract Audio Tensor
            audio_tensor = out_audios.audio_wavs[0][0].cpu().to(torch.float32)
            sample_rate = out_audios.sample_rate
            
            # Save Audio
            out_filename = f"translated_{file_path.name}"
            out_filepath = output_audio_dir / out_filename
            torchaudio.save(str(out_filepath), audio_tensor, sample_rate)
            
            # Store Metadata
            all_results.append({
                "original_file": file_path.name,
                "original_path": str(file_path),
                "translated_text": translated_text,
                "output_audio_path": str(out_filepath)
            })
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[!] OOM Error on {file_path.name}. Skipping. Try shorter audio.")
                torch.cuda.empty_cache()
            else:
                print(f"\n[!] Error processing {file_path.name}: {e}")
            continue
            
    # 5. Save JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
    print(f"✓ {lang_name.upper()} complete! Saved to {output_json_path}")


def main():
    """Run LRL -> English S2ST inference for every configured language.

    Workflow:
        1. Print the run header (model + script banner).
        2. Load the SeamlessM4T ``Translator`` once from ``MODEL_PATH``
           via :func:`load_custom_translator` (which patches the
           fine-tuned state dict layout).
        3. For each ``(lang_name, lang_code)`` in ``LANGUAGES``, call
           :func:`process_language`, which iterates over the wav files
           under ``BASE_INPUT_DIR/<lang_name>``, calls
           ``translator.predict(task_str="s2st", src_lang=...,
           tgt_lang="eng")`` per file (sequential to avoid OOM), and
           saves translated wavs plus a ``<lang_name>_results.json``
           under ``BASE_OUTPUT_DIR/<lang_name>``.
        4. Print a completion banner.

    Outputs:
        Translated wav files under
        ``BASE_OUTPUT_DIR/<lang_name>/audio`` and a
        ``<lang_name>_results.json`` per language.

    Returns:
        None.
    """
    print("\n" + "="*70)
    print("SeamlessM4T Multi-Language Batch Inference (S2ST)")
    print("="*70)
    
    # Load Model Once
    translator = load_custom_translator(MODEL_PATH, device)
    
    # Iterate through each language
    for lang_name, lang_code in LANGUAGES.items():
        process_language(translator, lang_name, lang_code)
        
    print("All languages processed successfully!")

if __name__ == "__main__":
    main()