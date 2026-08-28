"""Summarise and visualise POS-tag agreement between reference and prediction.

Reads a per-language ``*_pos_tagged.json`` (entries with
``pos_tags_reference`` and ``pos_tags_prediction``), prints a table of
tag counts and per-tag deltas, then computes and plots a row-normalised
POS confusion matrix.
"""

import json
from collections import Counter


def main():
    """Summarise POS-tag distribution and plot a confusion matrix.

    Workflow:
        1. Load
           ``./RESULTS/naijas2st/pos_tags/cascaded/nllb_lrl_to_eng_yoruba_pos_tagged.json``
           (the Yoruba cascaded NLLB output with HuggingFace-style
           POS dicts, i.e. ``[{"entity": "...", "score": ...}, ...]``).
        2. Build ``Counter`` objects ``reference_counts`` and
           ``prediction_counts`` over each item's ``pos_tags_reference``
           and ``pos_tags_prediction`` entities, then print a per-tag
           table ``POS | REF | PRED | DIFF | ERROR%`` for the union of
           tags.
        3. Re-iterate the data to build positionally-aligned
           ``y_true`` / ``y_pred`` lists (truncating each pair to the
           shorter sequence length), then compute the raw
           ``sklearn.metrics.confusion_matrix`` over the sorted tag
           union.
        4. Row-normalise the matrix to percentages, replacing NaNs
           where a reference row has zero entries, and print the
           rounded DataFrame.
        5. Render the matrix as a ``matplotlib`` heatmap (Blues
           colormap, percentage cell labels, rotated x ticks) and save
           it as
           ``nllb_lrl_to_eng_yoruba_pos_tagged.png``.

    Inputs:
        One ``*_pos_tagged.json`` file with ``pos_tags_reference`` /
        ``pos_tags_prediction`` dicts.

    Outputs:
        Stdout tables and a saved confusion-matrix PNG.

    Returns:
        None.
    """
    with open("./RESULTS/naijas2st/pos_tags/cascaded/nllb_lrl_to_eng_yoruba_pos_tagged.json", "r") as f:
        data = json.load(f)

    reference_counts = Counter()
    prediction_counts = Counter()

    for item in data:

        ref_tags = item["pos_tags_reference"]
        pred_tags = item["pos_tags_prediction"]

        for tag in ref_tags:
            reference_counts[tag["entity"]] += 1

        for tag in pred_tags:
            prediction_counts[tag["entity"]] += 1

    all_pos = sorted(
        set(reference_counts.keys()) |
        set(prediction_counts.keys())
    )

    print(f"{'POS':<10} {'REF':<10} {'PRED':<10} {'DIFF':<10} {'ERROR%':<10}")

    for pos in all_pos:

        ref = reference_counts[pos]
        pred = prediction_counts[pos]

        diff = pred - ref

        error_pct = abs(diff) / ref * 100 if ref > 0 else 0

        print(
            f"{pos:<10} "
            f"{ref:<10} "
            f"{pred:<10} "
            f"{diff:<10} "
            f"{error_pct:.2f}"
        )

    import json
    import pandas as pd
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    # Load data
    with open("./RESULTS/naijas2st/pos_tags/cascaded/nllb_lrl_to_eng_yoruba_pos_tagged.json", "r") as f:
        data = json.load(f)

    y_true = []
    y_pred = []

    for item in data:

        ref_tags = [
            tag["entity"]
            for tag in item["pos_tags_reference"]
        ]

        pred_tags = [
            tag["entity"]
            for tag in item["pos_tags_prediction"]
        ]

        # positional alignment
        min_len = min(len(ref_tags), len(pred_tags))

        y_true.extend(ref_tags[:min_len])
        y_pred.extend(pred_tags[:min_len])

    # POS labels
    labels = sorted(list(set(y_true) | set(y_pred)))

    # Raw confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # Convert to percentages (row-normalized)
    cm_percent = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    # Replace NaNs if any row sums to zero
    cm_percent = np.nan_to_num(cm_percent)

    # Convert to percentages
    cm_percent *= 100

    # DataFrame
    cm_df = pd.DataFrame(
        cm_percent,
        index=labels,
        columns=labels
    )

    print(cm_df.round(2))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(cm_percent, cmap="Blues")

    # Colorbar
    cbar = fig.colorbar(im)
    cbar.set_label("Percentage")

    # Ticks
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted POS")
    ax.set_ylabel("Reference POS")
    ax.set_title("Normalized POS Confusion Matrix (%)")

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
                fontsize=7
            )

    plt.tight_layout()
    plt.show()
    plt.savefig("./RESULTS/naijas2st/pos_tags/cascaded/nllb_lrl_to_eng_yoruba_pos_tagged.png")

if __name__ == "__main__":
    main()