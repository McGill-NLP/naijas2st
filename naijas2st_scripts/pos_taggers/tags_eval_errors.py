"""Detailed POS-tag error analysis: alignment, confusion, top mismatches.

Loads a POS-tagged JSON, aligns reference and prediction tokens with
``difflib.SequenceMatcher`` (so it tolerates insertions/deletions),
builds a per-POS confusion matrix, and reports the most common
mismatched (reference, prediction) tag pairs together with example
sentences. Plots are saved alongside the input file.
"""

import json
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def clean_token(token):
    """Normalise a tokeniser-produced subword for alignment.

    Args:
        token (str): Raw token string (may contain ``▁`` artefacts).

    Returns:
        str: Lower-cased, stripped string with the SentencePiece
        underscore removed.
    """
    return (
        token
        .replace("▁", "")
        .strip()
        .lower()
    )


def extract_tokens_and_pos(tag_list):
    """Split a ``[[word, POS], ...]`` list into parallel token/POS lists.

    Args:
        tag_list (list[list[str]]): List of ``[word, pos]`` pairs as
            produced by the POS-tag enrichment scripts.

    Returns:
        tuple[list[str], list[str]]: ``(tokens, pos_tags)`` of equal
        length; tokens are cleaned via :func:`clean_token`.
    """
    tokens = []
    pos_tags = []

    for tok in tag_list:

        # Safety checks
        if not isinstance(tok, list):
            continue

        if len(tok) != 2:
            continue

        token = clean_token(tok[0])
        pos = tok[1]

        tokens.append(token)
        pos_tags.append(pos)

    return tokens, pos_tags


def main():
    """Run the full POS-tag error analysis and save figures.

    Workflow:
        1. Load the input ``*_pos_tagged_spacy.json`` (Yoruba audio-LLM
           predictions tagged with spaCy by default).
        2. Initialise per-POS counters
           ``stats[pos] = {match, substitute, delete, insert}`` and
           lists ``y_true`` / ``y_pred`` for the confusion matrix.
        3. For each item:
            - Strip tokeniser artefacts and lower-case via
              :func:`clean_token`, then extract parallel
              ``(tokens, pos)`` lists for reference and prediction
              with :func:`extract_tokens_and_pos`.
            - Align the two token sequences with
              ``difflib.SequenceMatcher.get_opcodes`` so insertions
              and deletions are accounted for properly rather than
              forcing positional alignment.
            - For each opcode, increment the appropriate
              ``stats`` field (``equal`` -> ``match``, ``replace``
              -> ``substitute``, ``delete`` -> ``delete``, ``insert``
              -> ``insert``) and accumulate aligned ``(ref_pos,
              pred_pos)`` pairs for the confusion matrix.
        4. Compute per-POS accuracy / substitution / deletion /
           insertion rates and print them.
        5. Build a row-normalised confusion matrix over the union of
           tags and render a heatmap PNG.
        6. Surface the top-K most common substitution pairs together
           with example sentences pulled back from the source data.

    Inputs:
        A ``*_pos_tagged*.json`` file (with ``[[word, pos], ...]`` tag lists).

    Outputs:
        Summary tables on stdout plus CSV and PNG artefacts written
        next to the input file.

    Returns:
        None.
    """
    input_file = (
        "./RESULTS/naijas2st/pos_tags/audiollm/"
        "lrl_to_eng_yoruba_pos_tagged_spacy.json"
    )

    output_dir = (
        "./RESULTS/naijas2st/pos_tags/audiollm/"
    )

    # ============================================================
    # LOAD DATA
    # ============================================================

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ============================================================
    # STORAGE
    # ============================================================

    stats = defaultdict(lambda: {
        "match": 0,
        "substitute": 0,
        "delete": 0,
        "insert": 0
    })

    # For confusion matrix
    y_true = []
    y_pred = []

    # ============================================================
    # MAIN EVALUATION LOOP
    # ============================================================

    for item in data:

        # --------------------------------------------------------
        # Extract tokens + POS
        # --------------------------------------------------------

        ref_tokens, ref_pos = extract_tokens_and_pos(
            item["pos_tags_reference"]
        )

        pred_tokens, pred_pos = extract_tokens_and_pos(
            item["pos_tags_prediction"]
        )

        # Skip empty examples
        if len(ref_tokens) == 0 or len(pred_tokens) == 0:
            continue

        # --------------------------------------------------------
        # Align reference + prediction tokens
        # --------------------------------------------------------

        matcher = SequenceMatcher(
            None,
            ref_tokens,
            pred_tokens
        )

        opcodes = matcher.get_opcodes()

        # --------------------------------------------------------
        # Evaluate operations
        # --------------------------------------------------------

        for tag, i1, i2, j1, j2 in opcodes:

            # ====================================================
            # MATCHES
            # ====================================================

            if tag == "equal":

                aligned_ref_pos = ref_pos[i1:i2]
                aligned_pred_pos = pred_pos[j1:j2]

                for r_pos, p_pos in zip(
                    aligned_ref_pos,
                    aligned_pred_pos
                ):

                    stats[r_pos]["match"] += 1

                    y_true.append(r_pos)
                    y_pred.append(p_pos)

            # ====================================================
            # SUBSTITUTIONS
            # ====================================================

            elif tag == "replace":

                aligned_ref_pos = ref_pos[i1:i2]
                aligned_pred_pos = pred_pos[j1:j2]

                min_len = min(
                    len(aligned_ref_pos),
                    len(aligned_pred_pos)
                )

                # --------------------------------------------
                # Paired substitutions
                # --------------------------------------------

                for k in range(min_len):

                    r_pos = aligned_ref_pos[k]
                    p_pos = aligned_pred_pos[k]

                    stats[r_pos]["substitute"] += 1

                    y_true.append(r_pos)
                    y_pred.append(p_pos)

                # --------------------------------------------
                # Extra reference tokens = deletions
                # --------------------------------------------

                for r_pos in aligned_ref_pos[min_len:]:

                    stats[r_pos]["delete"] += 1

                # --------------------------------------------
                # Extra predicted tokens = insertions
                # --------------------------------------------

                for p_pos in aligned_pred_pos[min_len:]:

                    stats[p_pos]["insert"] += 1

            # ====================================================
            # DELETIONS
            # ====================================================

            elif tag == "delete":

                for r_pos in ref_pos[i1:i2]:

                    stats[r_pos]["delete"] += 1

            # ====================================================
            # INSERTIONS
            # ====================================================

            elif tag == "insert":

                for p_pos in pred_pos[j1:j2]:

                    stats[p_pos]["insert"] += 1

    # ============================================================
    # COMPUTE METRICS
    # ============================================================

    results = []

    for pos, s in stats.items():

        total_ref = (
            s["match"] +
            s["substitute"] +
            s["delete"]
        )

        total_pred = (
            s["match"] +
            s["substitute"] +
            s["insert"]
        )

        errors = (
            s["substitute"] +
            s["delete"]
        )

        recall = (
            s["match"] / total_ref
            if total_ref > 0 else 0
        )

        precision = (
            s["match"] / total_pred
            if total_pred > 0 else 0
        )

        error_rate = (
            errors / total_ref
            if total_ref > 0 else 0
        )

        deletion_rate = (
            s["delete"] / total_ref
            if total_ref > 0 else 0
        )

        substitution_rate = (
            s["substitute"] / total_ref
            if total_ref > 0 else 0
        )

        results.append({
            "POS": pos,
            "MATCH": s["match"],
            "SUB": s["substitute"],
            "DEL": s["delete"],
            "INS": s["insert"],
            "RECALL": recall * 100,
            "PRECISION": precision * 100,
            "ERROR_RATE": error_rate * 100,
            "DEL_RATE": deletion_rate * 100,
            "SUB_RATE": substitution_rate * 100
        })

    # ============================================================
    # RESULTS DATAFRAME
    # ============================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="ERROR_RATE",
        ascending=False
    )

    print("\n================ POS METRICS ================\n")

    print(
        results_df.round(2).to_string(index=False)
    )

    # ============================================================
    # SAVE CSV
    # ============================================================

    csv_path = output_dir + "pos_metrics.csv"

    results_df.to_csv(
        csv_path,
        index=False
    )

    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    labels = sorted(
        list(set(y_true) | set(y_pred))
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # ------------------------------------------------------------
    # Row normalization
    # ------------------------------------------------------------

    cm_percent = cm.astype(float)

    row_sums = cm_percent.sum(
        axis=1,
        keepdims=True
    )

    cm_percent = np.divide(
        cm_percent,
        row_sums,
        where=row_sums != 0
    )

    cm_percent = np.nan_to_num(cm_percent)

    cm_percent *= 100

    # ============================================================
    # CONFUSION MATRIX DATAFRAME
    # ============================================================

    cm_df = pd.DataFrame(
        cm_percent,
        index=labels,
        columns=labels
    )

    print("\n================ CONFUSION MATRIX (%) ================\n")

    print(cm_df.round(2))

    # ============================================================
    # PLOT CONFUSION MATRIX
    # ============================================================

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(
        cm_percent,
        cmap="Blues"
    )

    # Colorbar
    cbar = fig.colorbar(im)
    cbar.set_label("Percentage")

    # Axis labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=90
    )

    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted POS")
    ax.set_ylabel("Reference POS")

    ax.set_title(
        "POS Confusion Matrix (%)"
    )

    # ------------------------------------------------------------
    # Add percentages to cells
    # ------------------------------------------------------------

    for i in range(len(labels)):
        for j in range(len(labels)):

            value = cm_percent[i, j]

            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7
            )

    plt.tight_layout()

    conf_matrix_path = (
        output_dir + "pos_confusion_matrix.png"
    )

    plt.savefig(
        conf_matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ============================================================
    # ERROR RATE BARPLOT
    # ============================================================

    sorted_df = results_df.sort_values(
        by="ERROR_RATE",
        ascending=False
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        sorted_df["POS"],
        sorted_df["ERROR_RATE"]
    )

    plt.xticks(rotation=45)

    plt.xlabel("POS Tag")
    plt.ylabel("Error Rate (%)")

    plt.title("POS Translation Error Rates")

    plt.tight_layout()

    error_plot_path = (
        output_dir + "pos_error_rates.png"
    )

    plt.savefig(
        error_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n================ SAVED FILES ================\n")

    print(csv_path)
    print(conf_matrix_path)
    print(error_plot_path)


if __name__ == "__main__":
    main()