"""SeamlessM4T LRL audio -> English text (S2TT) or speech (S2ST) batch inference.

Uses the ``seamless_communication`` ``Translator`` API with a custom
fine-tuned checkpoint. ``TASK`` controls whether the script does
speech-to-text or speech-to-speech translation. Writes per-language
results JSONs (and audio files in S2ST mode).
"""

import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from seamless_communication.inference import Translator

# ==========================================
# CONFIGURATION
# ==========================================
# Point this directly to your single saved file
MODEL_PATH = "./finetuning_output_stt"

BASE_INPUT_DIR = "NaijaS2ST_dev_test/test"
BASE_OUTPUT_DIR = "RESULTS/naijas2st/seamless_cli_jonah/"

# Toggle this depending on what you trained!
# "s2tt" = Speech-to-Text
# "s2st" = Speech-to-Speech
TASK = "s2tt"  
TGT_LANG = "eng"

LANGUAGES = {
    "hausa": "hau",
    "yoruba": "yor",
    "igbo": "ibo"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_custom_translator(model_path, device):
    """Load a fine-tuned SeamlessM4T checkpoint and remap state-dict keys.

    Args:
        model_path (str | os.PathLike): Path to a torch ``.pt``
            checkpoint.
        device (torch.device | str): Torch device.

    Returns:
        seamless_communication.Translator: Initialised ``Translator``
        in eval mode.
    """
    print(f"Loading model from {model_path}...")
    
    # Load the single file directly
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
        # Step A: Remove the global 'model.' prefix
        new_key = key.replace("model.", "")
        
        # Step B: Map T2U sub-module names (Safely ignored if purely STT)
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
    print(f"✓ State dict loaded successfully! (Missing/Unexpected keys: {msg})")
    translator.model.eval() 
    return translator

def process_language(translator, lang_name, lang_code, task):
    """Run S2TT or S2ST translation for every wav under one language directory.

    Args:
        translator (seamless_communication.Translator): A loaded
            SeamlessM4T ``Translator``.
        lang_name (str): Folder name under ``BASE_INPUT_DIR``.
        lang_code (str): Source-language SeamlessM4T code.
        task (str): ``"s2tt"`` for text-only or ``"s2st"`` for
            text+audio output.

    Returns:
        None. Writes a per-language results JSON (and wavs for S2ST).
    """
    input_dir = Path(BASE_INPUT_DIR) / lang_name
    output_json_path = Path(BASE_OUTPUT_DIR) / lang_name / f"{lang_name}_{task}_results.json"
    
    # Only create an audio directory if we are doing STS
    if task == "s2st":
        output_audio_dir = Path(BASE_OUTPUT_DIR) / lang_name / "audio"
        output_audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the parent directory for the JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 2. Find Audio Files
    wav_files = sorted(list(input_dir.glob("**/*.wav"))) 
    
    if not wav_files:
        print(f"No .wav files found in {input_dir}. Skipping {lang_name.upper()}...")
        return
        
    print(f"\n" + "="*50)
    print(f"Processing {lang_name.upper()} ({lang_code}) - {len(wav_files)} files")
    print("="*50)

    # 3. Sequential Processing Loop
    all_results = []

    for file_path in tqdm(wav_files, desc=f"Translating {lang_name}"):
        try:
            # Predict takes a SINGLE string file path
            out_texts, out_audios = translator.predict(
                input=str(file_path),
                task_str=task,
                src_lang=lang_code, 
                tgt_lang=TGT_LANG
            )
            
            # 4. Unpack Results
            translated_text = str(out_texts[0])
            
            result_dict = {
                "original_file": file_path.name,
                "original_path": str(file_path),
                "translated_text": translated_text
            }
            
            # 5. Handle Audio only if doing STS
            if task == "s2st" and out_audios is not None:
                audio_tensor = out_audios.audio_wavs[0][0].cpu().to(torch.float32)
                sample_rate = out_audios.sample_rate
                
                out_filename = f"translated_{file_path.name}"
                out_filepath = output_audio_dir / out_filename
                torchaudio.save(str(out_filepath), audio_tensor, sample_rate)
                
                result_dict["output_audio_path"] = str(out_filepath)
                
            all_results.append(result_dict)
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[!] OOM Error on {file_path.name}. Skipping. Try shorter audio.")
                torch.cuda.empty_cache()
            else:
                print(f"\n[!] Error processing {file_path.name}: {e}")
            continue
            
    # 6. Save JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
    print(f"✓ {lang_name.upper()} complete! Saved to {output_json_path}")

def main():
    """Run SeamlessM4T inference for every configured language with one model.

    Workflow:
        1. Print the run header showing the chosen ``TASK``
           (``s2tt`` or ``s2st``).
        2. Load the fine-tuned ``Translator`` from ``MODEL_PATH`` via
           :func:`load_custom_translator` (handles the custom
           state-dict key remapping).
        3. For each ``(lang_name, lang_code)`` in ``LANGUAGES``, call
           :func:`process_language` to translate every wav under
           ``BASE_INPUT_DIR/<lang_name>``. In ``s2st`` mode, also save
           per-utterance translated wavs under
           ``BASE_OUTPUT_DIR/<lang_name>/audio``.
        4. Print a completion banner.

    Inputs:
        ``BASE_INPUT_DIR/<language>/**/*.wav`` plus the fine-tuned
        checkpoint at ``MODEL_PATH``.

    Outputs:
        Per-language ``<lang_name>_<task>_results.json`` (and wav
        files in ``s2st`` mode) under ``BASE_OUTPUT_DIR``.

    Returns:
        None.
    """
    print("\n" + "="*70)
    print(f"SeamlessM4T Multi-Language Inference (Task: {TASK.upper()})")
    print("="*70)
    
    # Load Model Once
    translator = load_custom_translator(MODEL_PATH, device)
    
    # Iterate through each language
    for lang_name, lang_code in LANGUAGES.items():
        process_language(translator, lang_name, lang_code, TASK)
        
    print("All languages processed successfully!")

if __name__ == "__main__":
    main()