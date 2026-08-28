"""Synthesise LRL audio from English->LRL text predictions via Gemini TTS.

Counterpart to ``gemini_sts.py`` for the English->LRL direction. Reads
per-language ``RESULTS/.../<language>.json`` text predictions and asks
Gemini TTS to read each line aloud in the target LRL, saving one WAV
per item.
"""

from google import genai
from google.genai import types
import os
import json
from pathlib import Path
import wave


def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """Write raw PCM bytes to disk as a WAV file.

    Args:
        filename (str | os.PathLike): Destination ``.wav`` path.
        pcm (bytes): Raw PCM bytes returned by the TTS API.
        channels (int): Number of audio channels.
        rate (int): Sample rate in Hz.
        sample_width (int): Sample width in bytes.

    Returns:
        None.
    """
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def main():
    """Synthesise LRL audio from few-shot English->LRL text predictions.

    Workflow (per language in ``language_list``):
        1. Set up the Gemini client and per-language results directory
           under ``base_results_dir/<language>/``.
        2. Load
           ``RESULTS/naijas2st/few_shot_S2T_eng_to_lrl_gemini3/<language>.json``
           whose ``prediction`` field is the model's LRL translation.
        3. For every item:
            - Recover a clean ``stem_name`` from the original audio path.
            - Build the prompt ``"Say the following sentence in
              <language>: <translation>"`` (no accent override).
            - Call ``gemini-2.5-flash-preview-tts`` requesting AUDIO
              with the ``Kore`` voice (up-to-5 retries).
            - Decode the inline PCM bytes and write
              ``<results_dir>/<stem>.wav`` via :func:`wave_file`.
        4. Print how many wavs were generated.

    Outputs:
        WAV files under
        ``RESULTS/naijas2st/sts_mt31_gemini_eng_to_lrl/<language>/``.

    Returns:
        None.
    """
    language_list = ["yoruba","hausa","igbo"]
    language_accent_dictonary = {
        "yoruba": "southern", 
        "hausa": "northern", 
        "igbo": "southern"
    }
 
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

    client = genai.Client(api_key=GOOGLE_API_KEY)


    base_results_dir = "./RESULTS/naijas2st/sts_mt31_gemini_eng_to_lrl/"
    os.makedirs(base_results_dir, exist_ok=True)
    test_base_dir = Path(f"./RESULTS/naijas2st/few_shot_S2T_eng_to_lrl_gemini3")

    for language in language_list:
        print(f"\n→ Processing {language}...")
        test_json = test_base_dir / f"{language}.json"
        print('test json path:', test_json)
        results_dir = os.path.join(base_results_dir, f"{language}/")
        os.makedirs(results_dir, exist_ok=True)

        results = []
        counter = 0
        with open (test_json, "r", encoding="utf-8") as fp:
            test_set = json.load(fp)
            for test_item in test_set:
                file_name = test_item['file_name']
                # file_name = test_item['ID']
                stem_name = file_name.split("/")[3]
                stem_name = stem_name.split(".")[0]
                print(f"→ Processing {stem_name}...")
                translation = test_item['prediction']
                # translation = test_item['translation']
                prompt_parts = [f"Say the following sentence in {language}: {translation}"]
                
                success = False
                number_of_retries = 0
                while not success:
                    print(f"    ↳ Attempt {number_of_retries + 1}")
                    number_of_retries += 1
                    if number_of_retries > 5:
                        print(f"  ↳ failed to process, moving on...")
                        break
                    try:
                        resp = client.models.generate_content(
                                    model="gemini-2.5-flash-preview-tts",
                                    contents=prompt_parts,
                                    config=types.GenerateContentConfig(
                                        response_modalities=["AUDIO"],
                                        speech_config=types.SpeechConfig(
                                            voice_config=types.VoiceConfig(
                                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                                voice_name='Kore',
                                                )
                                            )
                                        ),
                                    )
                                    )

                        data = resp.candidates[0].content.parts[0].inline_data.data
                        counter += 1
                        wav_file_name=f"{results_dir}{stem_name}.wav"
                        wave_file(wav_file_name, data)

                    except Exception as e:
                        success = False
                        print(f"caught error, retrying: {e}")

        print(f"  ↳ saved {counter} for {language} translations to {results_dir}")


if __name__ == "__main__":
    main()
