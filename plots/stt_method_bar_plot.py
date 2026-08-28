"""Simple per-method XX->Eng spBLEU bar plot.

Builds an inline scores DataFrame and renders one PNG/SVG per method
showing spBLEU bars for Hausa/Igbo/Yoruba.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    """Plot per-method XX -> Eng spBLEU bar charts (one PNG/SVG per method).

    Workflow:
        1. Define the inline ``data`` table of spBLEU scores covering
           ``XX -> Eng`` and ``Eng -> XX`` for the three method
           families.
        2. Wrap it in a ``pandas.DataFrame`` with explicit columns
           and group rows by ``Method``.
        3. For each method group, render one ``matplotlib`` figure
           with three bars per model (Hausa/Igbo/Yoruba), rotated x
           labels, an spBLEU y-axis and a horizontal legend.
        4. ``plt.show()`` the figure and then save it as
           ``<method>_XX_Eng.svg``.

    Outputs:
        One SVG per method in the current working directory plus the
        live ``plt.show()`` window for interactive use.

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


    languages = ["Hausa", "Igbo", "Yoruba"]

    for method, group in df.groupby("Method"):
        x = range(len(group))

        plt.figure()
        
        # Example: plot XX → Eng
        for i, lang in enumerate(languages):
            plt.bar(
                [p + i*0.2 for p in x],
                group[f"XX_Eng_{lang}"],
                width=0.2,
                label=f"{lang}"
            )

        plt.xticks([p + 0.2 for p in x], group["Model"], rotation=30, ha='right')
        plt.title(f"{method} (XX → Eng)")
        plt.ylabel("spBLEU")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(f"{method}_XX_Eng.svg")


if __name__ == "__main__":
    main()
