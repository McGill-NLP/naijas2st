"""Paper-ready per-method spBLEU line plots across language pairs.

For each ``(Method, direction)`` pair, draws a line chart with one line
per model showing how spBLEU varies across the three language pairs.
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def plot_method_line(df, method, direction="XX_Eng", save=False):
    """Render a per-method line plot of spBLEU across the three language pairs.

    Args:
        df (pandas.DataFrame): DataFrame with ``Method``/``Model``/
            per-language score columns.
        method (str): Method name to filter to.
        direction (str): ``"XX_Eng"`` or ``"Eng_XX"``.
        save (bool): If ``True``, save the figure as an SVG file.

    Returns:
        None.
    """
    subset = df[df["Method"] == method].copy()

    languages = ["Hausa", "Igbo", "Yoruba"]

    # Create ordered language pair labels
    if direction == "XX_Eng":
        pairs = [f"{lang}→Eng" for lang in languages]
    else:
        pairs = [f"Eng→{lang}" for lang in languages]

    # Reshape to long format
    long_df = subset.melt(
        id_vars=["Model"],
        value_vars=[f"{direction}_{lang}" for lang in languages],
        var_name="Lang",
        value_name="Score"
    )

    # Map to clean x-axis labels
    long_df["Language Pair"] = long_df["Lang"].apply(
        lambda x: pairs[languages.index(x.split("_")[-1])]
    )

    # Ensure correct ordering
    long_df["Language Pair"] = pd.Categorical(
        long_df["Language Pair"], categories=pairs, ordered=True
    )

    fig, ax = plt.subplots(figsize=(6, 3))

    # Plot one line per model
    for model, group in long_df.groupby("Model"):
        group = group.sort_values("Language Pair")
        ax.plot(
            group["Language Pair"],
            group["Score"],
            marker="o",
            label=model
        )

    # Titles and labels
    title_dir = "XX → Eng" if direction == "XX_Eng" else "Eng → XX"
    ax.set_title(f"{method} ({title_dir})")
    ax.set_ylabel("spBLEU")
    ax.set_xlabel("Language Pair")

    # Grid styling
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.grid(axis="x", visible=False)

    # Legend
    ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()

    if save:
        fname = f"{method}_{direction}_lineplot.svg".replace(" ", "_")
        plt.savefig(fname, bbox_inches="tight")

    plt.show()


def main():
    """Render paper-ready spBLEU line plots for every (method, direction).

    Workflow:
        1. Build an inline ``data`` table of spBLEU scores for the
           three method families across ``XX -> Eng`` and
           ``Eng -> XX`` directions.
        2. Construct a ``pandas.DataFrame`` with explicit column names
           and apply a paper-friendly seaborn theme + matplotlib
           rcParams (serif font, hidden top/right spines, 300 dpi).
        3. For each unique ``Method`` in the DataFrame, call
           :func:`plot_method_line` twice (once per direction) which
           reshapes the rows to long form and plots one line per model
           across the three language pairs.

    Outputs:
        One SVG per ``(method, direction)`` pair in the current
        working directory.

    Returns:
        None.
    """
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

    for method in df["Method"].unique():
        plot_method_line(df, method, "XX_Eng", save=True)
        plot_method_line(df, method, "Eng_XX", save=True)


if __name__ == "__main__":
    main()
