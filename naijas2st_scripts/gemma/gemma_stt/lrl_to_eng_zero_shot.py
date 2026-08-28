"""Zero-shot end-to-end LRL audio -> English translation with multimodal Gemma.

Uses the multimodal Gemma 4 checkpoint to perform speech translation
directly from audio. For each test wav, resamples to 16 kHz, constructs
a chat prompt with the audio attached, and asks the model for the
English translation. Writes per-language prediction JSONs.
"""

from pathlib import Path
import os
import json
from transformers import AutoProcessor, AutoModelForMultimodalLM
import tempfile
import soundfile as sf
import librosa


MODEL_ID = "google/gemma-4-E4B-it"
# MODEL_ID = "google/gemma-3n-E2B-it"

language_list = [
                'yoruba',
                'igbo',
                'hausa', 
                'pidgin'
                ]
test_base_dir = Path("./naijas2st_scripts/test/")
results_dir = "./RESULTS/naijas2st/gemma3_stt/lrl_to_eng_zero_shot"


def resample_audio_to_16k(input_path):
    """Resample an audio file to 16 kHz and write it to a temp WAV.

    Args:
        input_path (str | os.PathLike): Path to the input audio file.

    Returns:
        str: Path to the newly written 16 kHz temporary WAV file.
    """
    input_path = Path(input_path)

    audio, sr = librosa.load(input_path, sr=16000)

    tmp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    sf.write(tmp_file.name, audio, 16000)

    return tmp_file.name


def main():
    """Zero-shot multimodal Gemma LRL audio -> English translation.

    Workflow:
        1. Ensure ``results_dir`` exists and load ``MODEL_ID`` via
           ``AutoProcessor`` + ``AutoModelForMultimodalLM`` with
           ``device_map="auto"``.
        2. For each language in ``language_list``:
            - Pick the test directory ``./naijas2st_scripts/test/
              <language>`` and a per-language temp dir.
            - Walk every user dir (Pidgin uses ``<user>/*.wav``;
              everything else uses ``<user>/recordings/*.wav``).
            - For each wav, resample to 16 kHz via
              :func:`resample_audio_to_16k`, build a multimodal chat
              prompt with the audio attached, tokenise via
              ``processor.apply_chat_template``, and generate up to
              64 new tokens.
            - Decode the output, split on ``"model"`` to strip the
              chat template prefix, and append
              ``{"ID", "prediction"}``.
            - NOTE: the inner loop currently ``break``s after the
              first file per user, so this script behaves like a
              smoke test for each user rather than a full sweep.
        3. Write per-language predictions to
           ``<results_dir>/<language>.json``.

    Outputs:
        One predictions JSON per language under
        ``./RESULTS/naijas2st/gemma3_stt/lrl_to_eng_zero_shot``.

    Returns:
        None.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Load model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto"
    )

    for language in language_list:
        print(f"\n→ Processing {language}...")
        # Load streaming test dataset
        test_set = test_base_dir / language


        tmp_dir = tempfile.mkdtemp(prefix=f"local_{language}_")
        out_path = os.path.join(results_dir, f"{language}.json")
        results = []
        
        for user in test_set.iterdir():
            if language == "pidgin":
                user_dir = user
            else:
                user_dir = user / "recordings"

            for file in user_dir.iterdir():
                    file_name = file.stem
                    file_path = file
                    if language=="pidgin":
                        print(f"Processing file PIDGIN: {file_path}")
                    resampled_file = resample_audio_to_16k(file_path)
                    # Prompt
                    messages = [
                        {"role": "system",
                        "content": f"Your are an expert {language} to English translator. Translate the following {language} audio to English. Only output the English translation, without any additional text or formatting."},
                        {"role": "user", 
                        "content":[{
                            "type": "audio",
                            "audio": resampled_file}]},
                    ]

                    input_ids = processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True, return_dict=True,
                            return_tensors="pt",
                    )
                    inputs = input_ids.to(model.device, dtype=model.dtype)

                    outputs = model.generate(**inputs, max_new_tokens=64)

                    decoded = processor.decode(
                        outputs[0],
                        skip_special_tokens=True
                    )

                    translation = decoded.split("model")[-1].strip()

                    results.append({
                        "ID": f"{user}_{file_name}",
                        "prediction": translation
                    })
                    print(results)
                    break
        # Save results to JSON
        output_json = Path(f"{results_dir}/{language}.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Saved results to {output_json}")


if __name__ == "__main__":
    main()
