"""English POS-tag evaluation: frequency comparison + confusion matrix.

Sibling of ``tags_eval.py`` adapted for the English NLTK/spaCy tag
format (``[word, POS]`` lists rather than dicts with ``entity``).
Reads an English ``*_pos_tagged.json``, prints reference vs. prediction
counts and per-tag deltas, then computes and saves a row-normalised
POS confusion matrix as PNG.
"""

import json
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def main():
    """English POS-tag distribution check + confusion matrix.

    Workflow:
        1. Load
           ``./RESULTS/naijas2st/pos_tags/cascaded/nllb_lrl_to_eng_igbo_pos_tagged.json``
           where ``pos_tags_reference`` / ``pos_tags_prediction`` are
           ``[[word, POS], ...]`` lists (the NLTK / spaCy English
           format, not the HuggingFace dict format used by
           :file:`tags_eval.py`).
        2. Aggregate per-tag counts and print
           ``POS | REF | PRED | DIFF | ERROR%`` for the union of tags.
        3. Build positional ``y_true``/``y_pred`` arrays (truncated to
           the shorter sequence per pair), compute the raw confusion
           matrix and row-normalise it to percentages.
        4. Print the rounded percentage matrix.
        5. Render a Blues heatmap (with per-cell percentages, rotated
           x ticks) and save it as
           ``nllb_lrl_to_eng_igbo_pos_tagged.png`` at 300 dpi.

    Outputs:
        Stdout summary tables and a saved PNG confusion matrix.

    Returns:
        None.
    """
    # =========================
    # LOAD DATA
    # =========================

    json_path = "./RESULTS/naijas2st/pos_tags/cascaded/nllb_lrl_to_eng_igbo_pos_tagged.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    # =========================
    # POS FREQUENCY ANALYSIS
    # =========================

    reference_counts = Counter()
    prediction_counts = Counter()

    for item in data:

        ref_tags = item["pos_tags_reference"]
        pred_tags = item["pos_tags_prediction"]

        # tag format:
        # ["word", "POS"]

        for tag in ref_tags:
            pos = tag[1]
            reference_counts[pos] += 1

        for tag in pred_tags:
            pos = tag[1]
            prediction_counts[pos] += 1

    all_pos = sorted(
        set(reference_counts.keys()) |
        set(prediction_counts.keys())
    )

    print(
        f"{'POS':<10} "
        f"{'REF':<10} "
        f"{'PRED':<10} "
        f"{'DIFF':<10} "
        f"{'ERROR%':<10}"
    )

    for pos in all_pos:

        ref = reference_counts[pos]
        pred = prediction_counts[pos]

        diff = pred - ref

        error_pct = (
            abs(diff) / ref * 100
            if ref > 0 else 0
        )

        print(
            f"{pos:<10} "
            f"{ref:<10} "
            f"{pred:<10} "
            f"{diff:<10} "
            f"{error_pct:.2f}"
        )

    # =========================
    # BUILD CONFUSION MATRIX
    # =========================

    y_true = []
    y_pred = []

    for item in data:

        ref_tags = [
            tag[1]
            for tag in item["pos_tags_reference"]
        ]

        pred_tags = [
            tag[1]
            for tag in item["pos_tags_prediction"]
        ]

        # positional alignment
        min_len = min(len(ref_tags), len(pred_tags))

        y_true.extend(ref_tags[:min_len])
        y_pred.extend(pred_tags[:min_len])

    # POS labels
    labels = sorted(
        list(set(y_true) | set(y_pred))
    )

    # Raw confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # =========================
    # NORMALIZE TO PERCENTAGES
    # =========================

    cm_percent = (
        cm.astype("float") /
        cm.sum(axis=1, keepdims=True)
    )

    cm_percent = np.nan_to_num(cm_percent)

    cm_percent *= 100

    # =========================
    # DATAFRAME
    # =========================

    cm_df = pd.DataFrame(
        cm_percent,
        index=labels,
        columns=labels
    )

    print("\nNormalized Confusion Matrix (%):")
    print(cm_df.round(2))

    # =========================
    # PLOT
    # =========================

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(
        cm_percent,
        cmap="Blues"
    )

    # Colorbar
    cbar = fig.colorbar(im)
    cbar.set_label("Percentage")

    # Axis ticks
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=90
    )

    ax.set_yticklabels(labels)

    # Labels
    ax.set_xlabel("Predicted POS")
    ax.set_ylabel("Reference POS")

    ax.set_title(
        "Normalized POS Confusion Matrix (%)"
    )

    # Add percentages to cells
    for i in range(len(labels)):
        for j in range(len(labels)):

            value = cm_percent[i, j]

            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=6
            )

    plt.tight_layout()

    # Save BEFORE show
    output_png = "./RESULTS/naijas2st/pos_tags/cascaded/nllb_lrl_to_eng_igbo_pos_tagged.png"

    plt.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight"
    )

    # plt.show()

    print(f"\nSaved confusion matrix to:\n{output_png}")


if __name__ == "__main__":
    main()
