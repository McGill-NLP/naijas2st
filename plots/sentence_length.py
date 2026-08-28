"""Violin plot of prediction-vs-reference length ratios per system.

For each LRL, loads the end-to-end (SeamlessM4T), cascaded (NLLB) and
audio-LLM (Gemini) predictions JSONs, computes per-sample
``len(pred)/len(ref)``, and renders a seaborn violin plot grouped by
language and method.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    """Render a violin plot of prediction-to-reference length ratios per system.

    Workflow:
        1. Define the three target languages and a colour palette for
           the three system families (``Audio LLM``, ``Cascaded``,
           ``End-to-End``).
        2. Apply a clean seaborn whitegrid theme with custom rcParams
           (spineless axes, soft grid, serif-friendly tick colour).
        3. For each language:
            - Load the per-system prediction JSONs (SeamlessM4T
              end-to-end, NLLB-200 cascaded, Gemini few-shot audio
              LLM) for the English -> LRL direction.
            - Compute per-sample reference lengths (in words) from the
              cascaded file's ``reference``.
            - Compute per-system prediction lengths in words and
              align them with the references via ``zip`` (clipped to
              the shorter list).
            - For every aligned pair with a non-zero reference,
              append a ``{Language, Method, Length Ratio}`` row to
              the global ``rows`` list.
        4. Materialise ``rows`` as a DataFrame and clip extreme
           outliers (``Length Ratio < 3.0``) for readability.
        5. Render a seaborn ``violinplot`` (``x=Language``,
           ``y=Length Ratio``, ``hue=Method``), draw a dashed
           reference line at ``y=1`` (perfect length match), set
           titles/labels and a custom legend, and save the figure to
           ``./plots/figures/length_ratio_violinplot_eng_to_lrl.png``.

    Inputs:
        Per-system English -> LRL prediction JSONs.

    Outputs:
        ``length_ratio_violinplot_eng_to_lrl.png`` under
        ``./plots/figures/``.

    Returns:
        None.
    """
    language_list = ['yoruba', 'igbo', 'hausa']

    PALETTE = {
        "Audio LLM":  "#E07B39",
        "Cascaded":   "#3A7EBF",
        "End-to-End": "#5BAD72",
    }

    # ── Style ──────────────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid")

    plt.rcParams.update({
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,
        "axes.grid":         True,
        "grid.color":        "#E5E5E5",
        "grid.linewidth":    0.8,
        "axes.axisbelow":    True,
        "xtick.color":       "#666666",
        "ytick.color":       "#666666",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "axes.labelcolor":   "#444444",
        "axes.labelsize":    10,
        "legend.frameon":    False,
        "legend.fontsize":   9,
    })

    # ── Collect all rows into one dataframe ────────────────────────────────────────
    rows = []

    for language in language_list:

        print(f"→ Processing {language}...")

        # Load files
        with open(
            f"./RESULTS/naijas2st/seamless_stt_all/eng_to_lrl/seamless_eng_to_lrl_finetuned_mono_{language}/eng_to_{language}_translations_reformatted.json"
        ) as f:
            end_to_end_results = json.load(f)

        with open(
            f"./RESULTS/naijas2st/cascaded/"
            f"eng_to_lrl_LLM_1B_nllb-200-3.3B_{language}_reformatted.json"
        ) as f:
            cascaded_results = json.load(f)

        with open(
            f"./RESULTS/naijas2st/few_shot_S2T_eng_to_lrl_gemini3/"
            f"{language}_reformatted.json"
        ) as f:
            audio_llm_results = json.load(f)

        # Ground-truth lengths
        gt_len = [
            len(r["reference"].split())
            for r in cascaded_results
        ]

        # Method outputs
        method_outputs = {
            "Audio LLM": [
                len(r["prediction"].split())
                for r in audio_llm_results
            ],

            "Cascaded": [
                len(r["prediction"].split())
                for r in cascaded_results
            ],

            "End-to-End": [
                len(r["translation"].split())
                for r in end_to_end_results
            ],
        }

        # Build rows
        for method, pred_lengths in method_outputs.items():

            n_pairs = min(len(gt_len), len(pred_lengths))

            for gt, pred in zip(gt_len[:n_pairs], pred_lengths[:n_pairs]):

                # avoid divide-by-zero
                if gt == 0:
                    continue

                ratio = pred / gt

                rows.append({
                    "Language": language.capitalize(),
                    "Method": method,
                    "Length Ratio": ratio,
                })

    # ── DataFrame ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    # Optional: clip extreme outliers for readability
    df = df[df["Length Ratio"] < 3.0]

    # ── Figure layout ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(
        figsize=(8.5, 4.8),
        dpi=150
    )

    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # ── Violin plot ────────────────────────────────────────────────────────────────
    sns.violinplot(
        data=df,
        x="Language",
        y="Length Ratio",
        hue="Method",
        palette=PALETTE,
        inner="box",
        linewidth=1.0,
        cut=0,
        density_norm="width",
        ax=ax,
        alpha=0.7
    )

    # ── Reference line ─────────────────────────────────────────────────────────────
    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
        color="#666666",
        alpha=0.8,
    )

    # ── Labels / titles ────────────────────────────────────────────────────────────
    ax.set_title(
        "Prediction Length Relative to Ground Truth for English to Naija Translation",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    ax.set_ylabel("Predicted Length / Ground Truth Length")
    ax.set_xlabel("")

    # ── Legend ─────────────────────────────────────────────────────────────────────
    ax.legend(
        title="",
        loc="upper right",
        bbox_to_anchor=(1.2, 1.0)
    )

    # ── Y-axis tuning ──────────────────────────────────────────────────────────────
    ax.set_ylim(0, 2.5)

    # ── Save ───────────────────────────────────────────────────────────────────────
    out_path = "./plots/figures/length_ratio_violinplot_eng_to_lrl.png"

    plt.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )

    print(f"\nSaved plot to:\n{out_path}")


if __name__ == "__main__":
    main()
