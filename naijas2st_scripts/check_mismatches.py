"""Audit per-user prediction text files against the source wav recordings.

Walks a directory of submission ``.txt`` prediction files (named
``SLC.st.unconstrained.primary.<lang>-eng.<USER>.txt``), counts how many
prediction lines are present, compares to the number of wav recordings
for that user/language under ``<wav_base>/<lang>/<user>/recordings``,
and prints a per-user mismatch report.
"""

import os

def extract_user_id(filename):
    """Return the trailing user ID component of a submission filename.

    Args:
        filename (str): Submission filename, e.g.
            ``SLC.st.unconstrained.primary.yor-eng.Y041.txt``.

    Returns:
        str: The last dot-separated token before ``.txt``
        (e.g. ``"Y041"``).
    """
    parts = filename.replace('.txt', '').split('.')
    return parts[-1]

def get_language_from_filename(filename):
    """Recover the source-language code from a submission filename.

    Args:
        filename (str): Submission filename, e.g.
            ``SLC.st.unconstrained.primary.yor-eng.Y041.txt``.

    Returns:
        str | None: The source side of the language pair
        (e.g. ``"yor"``), or ``None`` if the filename does not match
        the expected dot-separated layout.
    """
    filename_no_ext = filename.replace('.txt', '')
    parts = filename_no_ext.split('.')
    if len(parts) >= 5:
        lang_pair = parts[-2]
        lang = lang_pair.split('-')[0]
        return lang
    return None

def get_wav_files(user_id, language, wav_base_path):
    """List the wav recordings for a given user and language.

    Args:
        user_id (str): User identifier (e.g. ``"Y041"``).
        language (str): Three-letter ISO-style language code
            (e.g. ``"yor"``).
        wav_base_path (str | os.PathLike): Root of the
            ``<lang>/<user>/recordings`` tree.

    Returns:
        list[str]: Sorted list of wav filenames under
        ``<wav_base_path>/<lang_folder>/<user_id>/recordings``, or an
        empty list if that directory does not exist.
    """
    lang_map = {
        'yor': 'yoruba',
        'hau': 'hausa',
        'ibo': 'igbo',
        'eng': 'english'
    }
    
    lang_folder = lang_map.get(language, language)
    user_path = os.path.join(wav_base_path, lang_folder, user_id, 'recordings')
    
    if not os.path.exists(user_path):
        return []
    
    wav_files = sorted([f for f in os.listdir(user_path) if f.endswith('.wav')])
    return wav_files

def check_mismatches(txt_dir, wav_base_dir):
    """Print a per-user audit comparing wav recordings to predicted lines.

    Workflow:
        1. Sort every ``.txt`` file in ``txt_dir``; each file is one
           submission, named like
           ``SLC.st.unconstrained.primary.<lang>-eng.<USER>.txt``.
        2. Print a header for a fixed-width table:
           ``User ID | Language | WAV Files | Predictions | Mismatch | Status``.
        3. For each prediction file:
            - Extract the user ID (:func:`extract_user_id`) and source
              language (:func:`get_language_from_filename`).
            - Count the wav files under
              ``<wav_base_dir>/<lang_folder>/<user>/recordings``
              with :func:`get_wav_files`.
            - Read the file line-by-line preserving empty lines (each
              line is one prediction, blanks included on purpose).
            - Compute ``mismatch = num_wav - num_pred`` and pick a
              status string (``"OK"`` if zero, ``"MISSING"`` otherwise).
            - Print one table row and accumulate totals.
        4. Print a separator line, the totals row, and a final
           "all match" / total-missing summary line.

    Args:
        txt_dir (str | os.PathLike): Directory with ``.txt`` prediction
            files (one per user).
        wav_base_dir (str | os.PathLike): Root containing
            ``<lang>/<user>/recordings`` wavs.

    Returns:
        None. The diagnostic table and totals go to stdout.
    """
    txt_files = sorted([f for f in os.listdir(txt_dir) if f.endswith('.txt')])
    
    print(f"\n{'User ID':<10} {'Language':<10} {'WAV Files':<12} {'Predictions':<12} {'Mismatch':<10} {'Status'}")
    print("=" * 75)
    
    total_wav = 0
    total_pred = 0
    total_missing = 0
    
    for txt_file in txt_files:
        txt_path = os.path.join(txt_dir, txt_file)
        user_id = extract_user_id(txt_file)
        language = get_language_from_filename(txt_file)
        
        # Get wav files
        wav_files = get_wav_files(user_id, language, wav_base_dir)
        num_wav = len(wav_files)
        
        # Read predictions (preserve empty lines)
        with open(txt_path, 'r', encoding='utf-8') as f:
            predictions = [line.rstrip('\n') for line in f.readlines()]
        num_pred = len(predictions)
        
        mismatch = num_wav - num_pred
        status = "✓ OK" if mismatch == 0 else "✗ MISSING"
        
        print(f"{user_id:<10} {language:<10} {num_wav:<12} {num_pred:<12} {mismatch:<10} {status}")
        
        total_wav += num_wav
        total_pred += num_pred
        total_missing += mismatch
    
    print("=" * 75)
    print(f"{'TOTAL':<10} {'':<10} {total_wav:<12} {total_pred:<12} {total_missing:<10}")
    
    if total_missing > 0:
        print(f"\n⚠ Total missing predictions: {total_missing}")
    else:
        print(f"\n✓ All predictions match!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check for mismatches between wav files and predictions")
    parser.add_argument("txt_dir", help="Directory containing txt prediction files")
    parser.add_argument("wav_base_dir", help="Base directory containing language/user_id/recordings structure")
    
    args = parser.parse_args()
    
    check_mismatches(args.txt_dir, args.wav_base_dir)
