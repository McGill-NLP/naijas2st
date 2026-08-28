"""Plot per-method spBLEU heatmaps across language pairs.

Builds an inline DataFrame of spBLEU scores for each method/model
across Hausa/Igbo/Yoruba in both XX->Eng and Eng->XX directions and
renders one paper-ready heatmap per method.
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


data = [
    # Method, Model, XX->Eng (Hausa, Igbo, Yoruba), Eng->XX (Hausa, Igbo, Yoruba)
    ["Cascaded", "ASR (CTC) + NLLB-200", 17.7, 8.4, 15.1, 0.0, 0.0, 0.0],
    ["Cascaded", "ASR (LLM) + NLLB-200", 17.3, 11.0, 16.1, 16.7, 29.2, 23.4],
    ["Cascaded", "ASR (LLM) + AfriqueQwen", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    ["End-to-End", "Seamless Zero-Shot", 1.3, 4.0, 19.5, np.nan, 7.5, 2.5],
    ["End-to-End", "Seamless Mono", 18.6, 17.6, 21.1, 0.0, 0.0, 0.0],
    ["End-to-End", "Seamless Multi", 13.9, 14.8, 18.5, 0.0, 0.0, 0.0],
    ["End-to-End", "Seamless Multi+Mono", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

    ["AudioLLM", "Gem. 2.5 Zero", 19.2, 7.2, 12.5, 26.0, 36.1, 25.2],
    ["AudioLLM", "Gem. 2.5 Few", 23.3, 10.9, 17.0, 26.7, 37.4, 31.2],
    ["AudioLLM", "Gem. 3.1 Zero", 30.0, 19.6, 28.4, 30.3, 39.0, 35.6],
    ["AudioLLM", "Gem. 3.1 Few", 35.6, 25.2, 33.3, 32.4, 40.6, 36.3],
]

columns = [
    "Method", "Model",
    "XX_Eng_Hausa", "XX_Eng_Igbo", "XX_Eng_Yoruba",
    "Eng_XX_Hausa", "Eng_XX_Igbo", "Eng_XX_Yoruba"
]

df = pd.DataFrame(data, columns=columns)


# Use a clean, paper-friendly style
sns.set_theme(style="whitegrid")

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_heatmap(df, method=None, save=False):
    """Render an spBLEU heatmap for one method (or all rows).

    Args:
        df (pandas.DataFrame): DataFrame with one row per
            ``(Method, Model)`` and language pair columns.
        method (str | None): Optional filter; if set, only rows
            matching this method are kept.
        save (bool): If ``True``, write the figure as an EPS file
            alongside showing it.

    Returns:
        None.
    """
    subset = df.copy()

    # Optionally filter by method
    if method is not None:
        subset = subset[subset["Method"] == method]

    languages = ["Hausa", "Igbo", "Yoruba"]

    # Build full set of language pairs (columns)
    pairs = (
        [f"{lang}→Eng" for lang in languages] +
        [f"Eng→{lang}" for lang in languages]
    )

    # Create a new dataframe with proper column names
    heatmap_df = pd.DataFrame()
    heatmap_df["Model"] = subset["Model"]

    for lang in languages:
        heatmap_df[f"{lang}→Eng"] = subset[f"XX_Eng_{lang}"]
        heatmap_df[f"Eng→{lang}"] = subset[f"Eng_XX_{lang}"]

    heatmap_df = heatmap_df.set_index("Model")

    # Ensure column order
    heatmap_df = heatmap_df[pairs]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".1f",
        cmap="OrRd",          # good perceptual colormap
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "spBLEU"},
        ax=ax
    )

    title = "All Methods" if method is None else method
    ax.set_title(f"{title} Performance")
    ax.set_xlabel("Language Pair")
    ax.set_ylabel("Model")

    plt.tight_layout()

    if save:
        fname = f"{title}_heatmap.eps".replace(" ", "_")
        plt.savefig(fname, bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    for method in df["Method"].unique():
        plot_heatmap(df, method, save=True)
        plot_heatmap(df, method, save=True)