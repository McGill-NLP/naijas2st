"""Inter-annotator agreement and correlation analysis for human eval CSVs.

Reads a per-annotator CSV of speech-translation quality ratings,
pivots it into ``item x annotator`` scores, computes pairwise
Spearman/Pearson correlations and a 2-way random ICC via ``pingouin``,
and writes the correlation matrix plus ICC results to CSV.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import pingouin as pg


def extract_number(x):
    """Parse a stored JSON-ish score string into a numeric score.

    Args:
        x (str): String like ``'[{"number":79}]'`` or
            ``'{"number":42}'``.

    Returns:
        float: The embedded ``number`` field, or ``numpy.nan`` on
        parse failure.
    """
    try:
        parsed = json.loads(x)

        # Handle list format
        if isinstance(parsed, list):
            return parsed[0]["number"]

        # Handle dict format
        if isinstance(parsed, dict):
            return parsed["number"]

    except Exception:
        return np.nan


def main():
    """Compute inter-annotator agreement and correlation for human-eval scores.

    Workflow:
        1. Load the per-annotator CSV (``naija_human_eval_hausa_to_english.csv``)
           with latin-1 encoding (to handle stray non-UTF-8 bytes).
        2. Add a ``has_source_text`` boolean column and parse the
           per-row JSON quality scores via :func:`extract_number`,
           dropping rows where parsing failed.
        3. Print a quick data summary: distribution of
           ``quality_score`` and the annotator counts.
        4. Create a stable ``item_id`` column and pivot to
           ``item_id x annotator`` of mean quality scores.
        5. Compute pairwise Spearman and Pearson correlations between
           annotators, store them as a square correlation matrix and
           print and write it to ``corr_csv``.
        6. Reshape the pivot to long form and run a 2-way random ICC
           via ``pingouin.intraclass_corr`` to estimate
           inter-annotator reliability; print the ICC results and
           write them to ``icc_csv``.
        7. Optionally render heatmaps/seaborn plots of the correlation
           matrix (subsequent code in the file).

    Inputs:
        ``human_eval_/naija_human_eval_hausa_to_english.csv``.

    Outputs:
        ``annotator_correlation_matrix.csv`` and ``icc_results.csv``
        (plus stdout summaries and any seaborn plots).

    Returns:
        None.
    """
    csv_path = "human_eval_/naija_human_eval_hausa_to_english.csv"

    # Use the shared annotation target as item_id
    item_id_col = "id"

    # Output files
    corr_csv = "annotator_correlation_matrix.csv"
    icc_csv = "icc_results.csv"

    # ============================================================
    # LOAD DATA
    # ============================================================

    df = pd.read_csv(csv_path, encoding="latin1")

    print(f"Loaded {len(df)} rows")


    df["has_source_text"] = (
        df["source_text"]
        .fillna("N/A")
        .ne("N/A")
    )

    # ============================================================
    # EXTRACT NUMERIC SCORES
    # ============================================================

    df["quality_score"] = df["target_audio_quality_score"].apply(extract_number)

    # Remove invalid rows
    df = df.dropna(subset=["quality_score"])

    print(f"Valid scores: {len(df)}")

    # ============================================================
    # BASIC DATA SUMMARY
    # ============================================================

    print("\n========== DATA SUMMARY ==========")
    print(df["quality_score"].describe())

    print("\nAnnotators:")
    print(df["annotator"].value_counts())

    # ============================================================
    # CREATE ITEM ID
    # ============================================================

    df["item_id"] = df[item_id_col]

    # ============================================================
    # PIVOT TABLE
    # Rows   = audio/sample
    # Columns= annotators
    # Values = quality scores
    # ============================================================

    pivot = df.pivot_table(
        index="item_id",
        columns="annotator",
        values="quality_score",
        aggfunc="mean"
    )

    print("\nPivot shape:")
    print(pivot.shape)

    # ============================================================
    # SPEARMAN CORRELATION MATRIX
    # ============================================================

    spearman_corr = pivot.corr(method="spearman")

    print("\n========== SPEARMAN CORRELATION ==========")
    print(spearman_corr)

    spearman_corr.to_csv(corr_csv)

    # ============================================================
    # PEARSON CORRELATION MATRIX
    # ============================================================

    pearson_corr = pivot.corr(method="pearson")

    print("\n========== PEARSON CORRELATION ==========")
    print(pearson_corr)

    # ============================================================
    # PAIRWISE CORRELATION WITH P-VALUES
    # ============================================================

    annotators = pivot.columns.tolist()

    pairwise_results = []

    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):

            a1 = annotators[i]
            a2 = annotators[j]

            pair = pivot[[a1, a2]].dropna()

            if len(pair) < 2:
                continue

            spear_corr, spear_p = spearmanr(pair[a1], pair[a2])
            pear_corr, pear_p = pearsonr(pair[a1], pair[a2])

            pairwise_results.append({
                "annotator_1": a1,
                "annotator_2": a2,
                "n_shared_items": len(pair),
                "spearman_r": spear_corr,
                "spearman_p": spear_p,
                "pearson_r": pear_corr,
                "pearson_p": pear_p
            })

    pairwise_df = pd.DataFrame(pairwise_results)

    print("\n========== PAIRWISE CORRELATIONS ==========")
    print(pairwise_df)

    pairwise_df.to_csv("pairwise_correlations.csv", index=False)

    # ============================================================
    # INTRACLASS CORRELATION (ICC)
    # ============================================================

    icc_input = df[["item_id", "annotator", "quality_score"]]

    icc = pg.intraclass_corr(
        data=icc_input,
        targets="item_id",
        raters="annotator",
        ratings="quality_score", 
        nan_policy="omit"
    )

    print("\n========== ICC RESULTS ==========")
    print(icc)

    icc.to_csv(icc_csv, index=False)

    # ============================================================
    # ANNOTATOR BIAS ANALYSIS
    # ============================================================

    annotator_stats = df.groupby("annotator")["quality_score"].agg([
        "count",
        "mean",
        "std",
        "median",
        "min",
        "max"
    ])

    print("\n========== ANNOTATOR STATS ==========")
    print(annotator_stats)

    annotator_stats.to_csv("annotator_stats.csv")

    # ============================================================
    # ITEM DISAGREEMENT ANALYSIS
    # ============================================================

    item_disagreement = df.groupby("item_id")["quality_score"].agg([
        "count",
        "mean",
        "std",
        "min",
        "max"
    ])

    item_disagreement = item_disagreement.sort_values(
        by="std",
        ascending=False
    )

    print("\n========== MOST CONTROVERSIAL ITEMS ==========")
    print(item_disagreement.head(20))

    item_disagreement.to_csv("item_disagreement.csv")

    # ============================================================
    # AGGREGATE TO ITEM LEVEL (mean score across annotators)
    # ============================================================

    item_level = df.groupby("item_id").agg(
        quality_score=("quality_score", "mean"),
        method=("method", "first"),
        has_source_text=("has_source_text", "first")
    ).reset_index()

    print(f"\nUnique items: {len(item_level)}")
    print(item_level["method"].value_counts())

    # ============================================================
    # METHOD-LEVEL ANALYSIS (per unique item)
    # ============================================================

    item_level = df.drop_duplicates(subset=["item_id"])

    method_stats = item_level.groupby("method")["quality_score"].agg([
        "count", "mean", "std", "median", "min", "max"
    ])

    print("\n========== METHOD STATS (per unique item) ==========")
    print(method_stats)
    method_stats.to_csv("method_stats.csv")


    # ============================================================
    # TEXT CONDITION ANALYSIS (per unique item)
    # ============================================================

    text_condition_stats = item_level.groupby("has_source_text")["quality_score"].agg([
        "count", "mean", "std", "median"
    ])

    print("\n========== TEXT CONDITION STATS (per unique item) ==========")
    print(text_condition_stats)
    text_condition_stats.to_csv("text_condition_stats.csv")


    # ============================================================
    # METHOD x TEXT CONDITION (per unique item)
    # ============================================================

    method_text_stats = item_level.groupby(["method", "has_source_text"])["quality_score"].agg([
        "count", "mean", "std", "median"
    ])

    print("\n========== METHOD x TEXT (per unique item) ==========")
    print(method_text_stats)
    method_text_stats.to_csv("method_text_stats.csv")


    from scipy.stats import kruskal

    groups = [
        group["quality_score"].values
        for _, group in df.groupby("method")
    ]

    stat, p = kruskal(*groups)

    print("\n========== METHOD SIGNIFICANCE ==========")
    print(f"Kruskal-Wallis H: {stat:.4f}")
    print(f"p-value: {p:.10f}")


    # ============================================================
    # Duplicate keys
    # ============================================================

    duplicate_key = (
        df["target_audio"].astype(str)
        + "||"
        + df["method"].astype(str)
        + "||"
        + df["has_source_text"].astype(str)
    )

    df["duplicate_key"] = duplicate_key


    dup_counts = df.groupby(
        ["annotator", "duplicate_key"]
    ).size()

    true_duplicates = dup_counts[
        dup_counts > 1
    ]

    print("\nDuplicate items:")
    print(true_duplicates.head())

    duplicate_consistency = []

    for (annotator, key), count in true_duplicates.items():

        subset = df[
            (df["annotator"] == annotator)
            & (df["duplicate_key"] == key)
        ]

        scores = subset["quality_score"].values

        if len(scores) >= 2:

            diff = abs(scores[0] - scores[1])

            duplicate_consistency.append({
                "annotator": annotator,
                "duplicate_key": key,
                "score_1": scores[0],
                "score_2": scores[1],
                "absolute_difference": diff
            })

    duplicate_df = pd.DataFrame(
        duplicate_consistency
    )

    print("\n========== DUPLICATE CONSISTENCY ==========")
    print(duplicate_df.head())

    print("\nMean duplicate difference:")
    print(
        duplicate_df["absolute_difference"].mean()
    )

    duplicate_df.to_csv(
        "duplicate_consistency.csv",
        index=False
    )

    annotator_dup_stats = duplicate_df.groupby(
        "annotator"
    )["absolute_difference"].agg([
        "count",
        "mean",
        "std",
        "max"
    ])

    print("\n========== DUPLICATE RELIABILITY ==========")
    print(annotator_dup_stats)

    annotator_dup_stats.to_csv(
        "annotator_duplicate_stats.csv"
    )

    # ============================================================
    # VISUALIZATIONS
    # ============================================================

    # ------------------------------------------------------------
    # Correlation Heatmap
    # ------------------------------------------------------------

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="method",
        y="quality_score",
        hue="has_source_text"
    )

    plt.title("Quality Scores by Method and Text Condition")
    plt.tight_layout()
    plt.savefig(
        "method_text_boxplots.png",
        dpi=300
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        spearman_corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.title("Annotator Spearman Correlation")
    plt.tight_layout()
    plt.savefig("annotator_correlation_heatmap.png", dpi=300)

    # ------------------------------------------------------------
    # Annotator Score Distributions
    # ------------------------------------------------------------

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df,
        x="annotator",
        y="quality_score"
    )

    plt.xticks(rotation=45)
    plt.title("Annotator Score Distributions")
    plt.tight_layout()
    plt.savefig("annotator_boxplots.png", dpi=300)

    # ------------------------------------------------------------
    # Histogram of All Scores
    # ------------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["quality_score"],
        bins=20,
        kde=True
    )

    plt.title("Distribution of Quality Scores")
    plt.xlabel("Quality Score")
    plt.tight_layout()
    plt.savefig("quality_score_distribution.png", dpi=300)

    # ------------------------------------------------------------
    # Item Disagreement Histogram
    # ------------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.histplot(
        item_disagreement["std"].dropna(),
        bins=20,
        kde=True
    )

    plt.title("Distribution of Inter-Annotator Disagreement")
    plt.xlabel("Std Dev Across Annotators")
    plt.tight_layout()
    plt.savefig("inter_annotator_disagreement.png", dpi=300)

    # ============================================================
    # SUMMARY INTERPRETATION
    # ============================================================

    print("\n========== SUMMARY ==========")

    avg_corr = (
        spearman_corr.where(~np.eye(spearman_corr.shape[0], dtype=bool))
        .stack()
        .mean()
    )

    print(f"Average Spearman correlation: {avg_corr:.3f}")

    icc2 = icc[icc["Type"] == "ICC2"]

    if not icc2.empty:
        icc_value = icc2["ICC"].values[0]

        print(f"ICC2: {icc_value:.3f}")

        if icc_value < 0.5:
            print("Interpretation: Poor agreement")
        elif icc_value < 0.75:
            print("Interpretation: Moderate agreement")
        elif icc_value < 0.9:
            print("Interpretation: Good agreement")
        else:
            print("Interpretation: Excellent agreement")

    print("\nSaved outputs:")
    print(f"- {corr_csv}")
    print("- pairwise_correlations.csv")
    print(f"- {icc_csv}")
    print("- annotator_stats.csv")
    print("- item_disagreement.csv")
    print("- annotator_correlation_heatmap.png")
    print("- annotator_boxplots.png")
    print("- quality_score_distribution.png")
    print("- inter_annotator_disagreement.png")


if __name__ == "__main__":
    main()
