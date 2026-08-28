"""Bundle per-user txt predictions and recording lists into one JSON.

Pairs each prediction line in submission ``.txt`` files with its
corresponding wav filename from ``<wav_base>/<lang>/<user>/recordings``
and writes the union as a single JSON of
``{user_id, file_name, prediction}`` records.
"""

import json
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

def get_language_from_path(txt_file_path):
    """Recover the source-language code from a submission file path.

    Args:
        txt_file_path (str | os.PathLike): Path or filename of a
            submission ``.txt`` file.

    Returns:
        str | None: The source side of the language pair
        (e.g. ``"yor"``), or ``None`` if the filename does not match
        the expected dot-separated layout.
    """
    filename = os.path.basename(txt_file_path).replace('.txt', '')
    parts = filename.split('.')
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
        print(f"Warning: Path not found {user_path}")
        return []
    
    wav_files = sorted([f for f in os.listdir(user_path) if f.endswith('.wav')])
    return wav_files

def create_predictions_json(txt_dir, wav_base_dir, output_file):
    """Pair submission ``.txt`` predictions with their wav files into one JSON.

    Workflow:
        1. List every ``*.txt`` file under ``txt_dir`` in sorted order;
           each file represents one user's submission.
        2. For each file:
            - Recover the user ID (:func:`extract_user_id`) and source
              language (:func:`get_language_from_path`).
            - Look up the wav files for that user/language under
              ``<wav_base_dir>/<lang>/<user>/recordings`` with
              :func:`get_wav_files`. Skip the file if no wavs are found.
            - Read the prediction file line-by-line preserving empty
              lines (so blank predictions stay aligned).
            - Warn (but continue) if ``len(wav_files) != len(predictions)``.
            - ``zip`` the wavs and predictions and append one
              ``{"user_id", "file_name", "prediction"}`` dict per
              ``(wav, pred)`` pair.
        3. Write the combined list to ``output_file`` as UTF-8 JSON.

    Args:
        txt_dir (str | os.PathLike): Directory with per-user ``.txt``
            prediction files.
        wav_base_dir (str | os.PathLike): Root containing
            ``<lang>/<user>/recordings`` wavs.
        output_file (str | os.PathLike): Destination JSON path.

    Returns:
        None. The combined records are written to ``output_file``.
    """
    all_predictions = []
    
    # Process each txt file
    txt_files = sorted([f for f in os.listdir(txt_dir) if f.endswith('.txt')])
    
    for txt_file in txt_files:
        txt_path = os.path.join(txt_dir, txt_file)
        user_id = extract_user_id(txt_file)
        language = get_language_from_path(txt_path)
        
        print(f"Processing: {txt_file} (user_id: {user_id}, language: {language})")
        
        # Get wav files for this user
        wav_files = get_wav_files(user_id, language, wav_base_dir)
        
        if not wav_files:
            print(f"  Warning: No wav files found for user {user_id}")
            continue
        
        # Read predictions from txt file (preserve empty lines)
        with open(txt_path, 'r', encoding='utf-8') as f:
            predictions = [line.rstrip('\n') for line in f.readlines()]
        
        print(f"  Found {len(wav_files)} wav files and {len(predictions)} predictions")
        
        # Match predictions to wav files
        if len(predictions) != len(wav_files):
            print(f"  Warning: Mismatch between wav files ({len(wav_files)}) and predictions ({len(predictions)})")
        
        for i, (wav_file, prediction) in enumerate(zip(wav_files, predictions)):
            entry = {
                "user_id": user_id,
                "file_name": wav_file,
                "prediction": prediction
            }
            all_predictions.append(entry)
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
    
    print(f"\nJSON file created: {output_file}")
    print(f"Total entries: {len(all_predictions)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create JSON file from txt predictions and wav files")
    parser.add_argument("txt_dir", help="Directory containing txt prediction files")
    parser.add_argument("wav_base_dir", help="Base directory containing language/user_id/recordings structure")
    parser.add_argument("--output", "-o", default="predictions.json", help="Output JSON file path")
    
    args = parser.parse_args()
    
    create_predictions_json(args.txt_dir, args.wav_base_dir, args.output)
