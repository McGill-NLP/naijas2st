"""Convert African/Celtic wav recordings into a partitioned Parquet dataset.

Walks the per-language folders under ``TEST_ROOT``, resamples each wav
to 16 kHz mono and re-encodes it as FLAC into a byte buffer, and writes
out one Parquet file per language under
``<corpus=...>/<split=test>/<language=...>``. Also writes a TSV
manifest that maps each row's ``file_id`` back to its source wav path.
"""

from pathlib import Path
import soundfile as sf
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import librosa
import io
from collections import defaultdict

TEST_ROOT = "./RESULTS/naijas2st/sts_openai_gemini_tts_lrl_to_eng/hausa"
OUTPUT_ROOT = "./huggingface_cache/naijas2st_parquet/openai_gemini_tts_eng_accented/hausa"
CORPUS = "naijas2st_test"
TARGET_SR = 16_000

# Map folder names to BCP-47 codes
LANG_MAP = {
    # "english": "eng_Latn",
    # "english_north_accent": "eng_Latn",
    # "english_south_accent": "eng_Latn",
    "hausa": "eng_Latn",
    "igbo": "eng_Latn",
    "yoruba": "eng_Latn",
}

SCHEMA = pa.schema([
    pa.field("text", pa.string()), # empty — no transcripts
    pa.field("audio_bytes", pa.list_(pa.int8())),
    pa.field("audio_size", pa.int64()),
    pa.field("corpus", pa.dictionary(pa.int32(), pa.string())),
    pa.field("split", pa.dictionary(pa.int32(), pa.string())),
    pa.field("language", pa.dictionary(pa.int32(), pa.string())),
])


def main():
    """Convert wavs under ``TEST_ROOT`` to per-language Parquet shards.

    Workflow:
        1. Walk ``TEST_ROOT/<lang_folder>/**/*.wav`` for every entry in
           ``LANG_MAP``; skip empty or unreadable files with a logged
           warning.
        2. For each wav:
            - Load with ``soundfile``, downmix to mono if stereo, and
              resample to ``TARGET_SR`` (16 kHz) with ``librosa`` when
              the source rate differs.
            - Re-encode the float array as FLAC bytes into an in-memory
              ``BytesIO`` buffer so the audio can be stored as a list
              of ``int8`` inside the Parquet row.
            - Build a record with ``text=file_id`` (stem of the wav),
              ``audio_bytes``, ``audio_size``, ``corpus``, ``split``,
              ``language``, plus housekeeping fields for the manifest.
            - Append to the per-language bucket and the manifest list.
        3. For each language bucket, build a PyArrow table matching
           ``SCHEMA`` (dictionary-encoding ``corpus``/``split``/``language``)
           and write a Parquet shard at
           ``<OUTPUT_ROOT>/corpus=.../split=test/language=.../part-00000.parquet``.
        4. Write ``test_manifest.tsv`` (next to ``OUTPUT_ROOT``'s parent)
           with ``file_id``, ``lang_folder``, ``lang_code``, ``path``
           columns so model outputs can be mapped back to source wavs.

    Inputs:
        Per-language wav trees under ``TEST_ROOT``.

    Outputs:
        Parquet shards under ``OUTPUT_ROOT`` and a single
        ``test_manifest.tsv`` mapping file_id back to source paths.

    Returns:
        None.
    """
    manifest_rows = []
    buckets = defaultdict(list)

    for lang_folder, lang_code in LANG_MAP.items():
        lang_dir = Path(TEST_ROOT) / lang_folder
        if not lang_dir.exists():
            print('lang dir does not exist')
            continue

        wav_files = sorted(lang_dir.rglob("*.wav"))
        print(f"{lang_folder}: {len(wav_files)} files")

        for wav_path in wav_files:
            # Skip empty files
            if wav_path.stat().st_size == 0:
                print(f"Skipping empty file: {wav_path}")
                continue
            
            try:
                audio, sr = sf.read(str(wav_path))
            except Exception as e:
                print(f"Error reading {wav_path}: {e}")
                continue
            
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != TARGET_SR:
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)

            audio_size = len(audio)
            buf = io.BytesIO()
            sf.write(buf, audio, TARGET_SR, format="FLAC")
            audio_bytes = np.frombuffer(buf.getvalue(), dtype=np.int8).tolist()

            # Use filename as the text placeholder so we can recover it later
            file_id = wav_path.stem # e.g. ETE_0251

            buckets[lang_code].append({
                "text": file_id, # store filename as text — useful for matching outputs
                "audio_bytes": audio_bytes,
                "audio_size": audio_size,
                "corpus": CORPUS,
                "split": "test",
                "language": lang_code,
                "source_lang_folder": lang_folder,
                "source_path": str(wav_path),
            })

            manifest_rows.append({
                "file_id": file_id,
                "lang_folder": lang_folder,
                "lang_code": lang_code,
                "path": str(wav_path),
            })
            print('file id:', file_id)

    # Write parquet files
    for lang_code, rows in buckets.items():
        out_dir = (
            Path(OUTPUT_ROOT)
            / f"corpus={CORPUS}"
            / "split=test"
            / f"language={lang_code}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        table = pa.table({
            "text": pa.array([r["text"] for r in rows], type=pa.string()),
            "audio_bytes": pa.array([r["audio_bytes"] for r in rows], type=pa.list_(pa.int8())),
            "audio_size": pa.array([r["audio_size"] for r in rows], type=pa.int64()),
            "corpus": pa.array([r["corpus"] for r in rows]).dictionary_encode().cast(pa.dictionary(pa.int32(), pa.string())),
            "split": pa.array([r["split"] for r in rows]).dictionary_encode().cast(pa.dictionary(pa.int32(), pa.string())),
            "language": pa.array([r["language"] for r in rows]).dictionary_encode().cast(pa.dictionary(pa.int32(), pa.string())),
        }, schema=SCHEMA)

        pq.write_table(table, out_dir / "part-00000.parquet", row_group_size=100)
        print(f"Wrote {len(rows)} rows → {out_dir}")

    # Write manifest TSV so you can map model output back to original files
    manifest_path = Path(OUTPUT_ROOT).parent / "test_manifest.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        f.write("file_id\tlang_folder\tlang_code\tpath\n")
        for r in manifest_rows:
            f.write(f"{r['file_id']}\t{r['lang_folder']}\t{r['lang_code']}\t{r['path']}\n")
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
