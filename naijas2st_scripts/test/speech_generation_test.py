"""End-to-end Gemini test: LRL audio -> English text -> Nigerian TTS.

Reads one local Hausa wav, asks Gemini 2.5 Flash to translate it to
English, then re-synthesises that English text in a Nigerian accent
with Gemini TTS, writing the result to ``out_test_no_accent.wav``.
"""

import tempfile
from google import genai
from google.genai import types
import wave
import soundfile as sf
import os


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
    """End-to-end smoke test: LRL wav -> English text -> Nigerian-accent TTS.

    Workflow:
        1. Instantiate a Gemini client from ``GOOGLE_API_KEY``.
        2. Read a hard-coded Hausa wav with ``soundfile`` and round-trip
           it into a temp WAV so the header is clean.
        3. Upload the temp wav to the Gemini file store.
        4. Call ``gemini-2.5-flash`` asking the model to transcribe the
           Hausa audio and translate it into English; print the
           response.
        5. Re-prompt ``gemini-2.5-flash-preview-tts`` with
           ``"Say the following in a Nigerian accent: <english>"`` and
           the ``Kore`` voice; decode the inline PCM payload and write
           it to ``out_test_no_accent.wav`` via :func:`wave_file`.

    Outputs:
        ``out_test_no_accent.wav`` in the current working directory.
        Stdout contains the intermediate English translation. Intended
        only as a smoke test for both Gemini models.

    Returns:
        None.
    """
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    tmp_dir = tempfile.mkdtemp(prefix="local_hausa_")

    file_path = "path/to/audio.wav"
    arr, sr = sf.read(file_path)
    tmp_path = os.path.join(tmp_dir, f"test_{file_path.split('/')[-1]}")
    sf.write(tmp_path, arr, sr)
    test_audio = client.files.upload(file=tmp_path)

    prompt_parts = ["Transcribe the following Hausa audio, and then give the exact translation in English. Only provide the English translation as output, no additional text."]
    prompt_parts.append(test_audio)

    response1 = client.models.generate_content(
       model="gemini-2.5-flash",
       contents=prompt_parts,
    )
    print("Transcription and Translation:", response1.text.strip())

    response2 = client.models.generate_content(
       model="gemini-2.5-flash-preview-tts",
       contents=f"Say the following in a Nigerian accent: {response1.text.strip()}",
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

    data = response2.candidates[0].content.parts[0].inline_data.data

    file_name='out_test_no_accent.wav'
    wave_file(file_name, data) # Saves the file to current directory


if __name__ == "__main__":
    main()
