"""Paper-ready spBLEU grouped bar plots per method and direction.

For each ``(Method, direction)`` pair, renders a compact bar chart of
spBLEU across Hausa/Igbo/Yoruba and saves it to SVG.
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def plot_method(df, method, direction="XX_Eng", save=False):
    """Render a grouped bar chart for one method in one translation direction.

    Args:
        df (pandas.DataFrame): DataFrame with ``Method``/``Model``/
            per-language score columns.
        method (str): Method name to filter to (e.g. ``"Cascaded"``).
        direction (str): ``"XX_Eng"`` or ``"Eng_XX"``.
        save (bool): If ``True``, save the figure as an SVG file.

    Returns:
        None.
    """
    subset = df[df["Method"] == method]
    languages = ["Hausa", "Igbo", "Yoruba"]

    x = range(len(subset))
    width = 0.22

    fig, ax = plt.subplots(figsize=(6, 3))  # compact paper size

    for i, lang in enumerate(languages):
        values = subset[f"{direction}_{lang}"]
        ax.bar(
            [p + i * width for p in x],
            values,
            width=width,
            label=lang
        )

    # X ticks centered
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(subset["Model"], rotation=25, ha="right")

    # Labels
    title_dir = "XX → Eng" if direction == "XX_Eng" else "Eng → XX"
    ax.set_title(f"{method} ({title_dir})")
    ax.set_ylabel("spBLEU")

    # Subtle grid
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.grid(axis="x", visible=False)

    # Legend
    ax.legend(frameon=False, ncol=3, loc="upper left")

    plt.tight_layout()

    if save:
        fname = f"{method}_{direction}.svg".replace(" ", "_")
        plt.savefig(fname, bbox_inches="tight")

    plt.show()


def main():
    """Render paper-ready spBLEU bar plots for every (method, direction).

    Workflow:
        1. Build an inline ``data`` table of spBLEU scores for the
           three method families (``Cascaded``, ``End-to-End``,
           ``AudioLLM``) across the three target languages in both
           directions (``XX -> Eng`` and ``Eng -> XX``).
        2. Convert ``data`` into a ``pandas.DataFrame`` with explicit
           column names.
        3. Apply a paper-friendly seaborn theme + matplotlib rcParams
           (serif font, hidden top/right spines, 300 dpi).
        4. For each unique ``Method`` in the DataFrame, call
           :func:`plot_method` twice (once with ``direction="XX_Eng"``
           and once with ``"Eng_XX"``) to render and save the
           grouped-bar charts as ``<method>_<direction>.svg``.

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
        ["End-to-End", "Seamless Monolingual", 18.6, 17.6, 21.1, 0.0, 0.0, 0.0],
        ["End-to-End", "Seamless Multilingual", 13.9, 14.8, 18.5, 0.0, 0.0, 0.0],
        ["End-to-End", "Seamless Multi+Mono", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

        ["AudioLLM", "Gemini 2.5 Zero", 19.2, 7.2, 12.5, 26.0, 36.1, 25.2],
        ["AudioLLM", "Gemini 2.5 Few", 23.3, 10.9, 17.0, 26.7, 37.4, 31.2],
        ["AudioLLM", "Gemini 3.1 Zero", 30.0, 19.6, 28.4, 30.3, 39.0, 35.6],
        ["AudioLLM", "Gemini 3.1 Few", 35.6, 25.2, 33.3, 32.4, 40.6, 36.3],
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
        plot_method(df, method, "XX_Eng", save=True)
        plot_method(df, method, "Eng_XX", save=True)


if __name__ == "__main__":
    main()
