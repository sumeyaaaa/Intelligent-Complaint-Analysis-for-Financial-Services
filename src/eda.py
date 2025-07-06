# src/eda_clean.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def product_distribution(
    clean_csv: str | Path = r"C:\Users\ABC\Desktop\10Acadamy\week_6\Intelligent-Complaint-Analysis-for-Financial-Services\data\clean\complaints_clean.csv",
    show_plot: bool = True,
) -> pd.Series:
    """
    Count complaints for each unique Product in `clean_csv`.

    Parameters
    ----------
    clean_csv : str | Path
        Path to the cleaned subset CSV (must contain a 'Product' column).
    show_plot : bool
        If True, display a bar-chart inline (Jupyter / VS Code).

    Returns
    -------
    pd.Series
        Index = Product, value = number of complaints.
    """
    counts = (
        pd.read_csv(clean_csv, usecols=["Product"])
          .value_counts(subset=["Product"])
          .sort_values(ascending=False)
          .rename("n_complaints")
    )

    if show_plot:
        counts.plot(kind="bar", figsize=(8, 4))
        plt.ylabel("# complaints")
        plt.title("Complaints per Product – clean dataset")
        plt.tight_layout()
        plt.show()

    return counts

# src/eda_narrative.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

def narrative_stats(
    clean_csv: str | Path = r"C:\Users\ABC\Desktop\10Acadamy\week_6\Intelligent-Complaint-Analysis-for-Financial-Services\data\clean\complaints_clean.csv",
    show_hist: bool = True,
    bins: int = 80,
) -> Dict:
    """
    Compute word-length distribution of 'Consumer complaint narrative'
    and counts of rows with / without narrative.

    Parameters
    ----------
    clean_csv : str | Path
        CSV that contains a 'Consumer complaint narrative' column.
    show_hist : bool
        Plot a histogram inline if True.
    bins : int
        Number of histogram bins (width ≈ range/ bins).

    Returns
    -------
    dict
        {
          "total"          : int,
          "with_narrative" : int,
          "without_narrative": int,
          "word_length_percentiles": {0:…, 25:…, 50:…, …, 100:…}
        }
    """
    col = "Consumer complaint narrative"
    df  = pd.read_csv(clean_csv, usecols=[col])

    mask_text   = df[col].notna() & (df[col].str.strip() != "")
    with_text   = int(mask_text.sum())
    without_txt = int((~mask_text).sum())
    total       = with_text + without_txt

    lengths = (
        df.loc[mask_text, col]
          .str.split()
          .str.len()
          .to_numpy(dtype=int)
    )

    if show_hist and len(lengths):
        plt.figure(figsize=(9,4))
        plt.hist(lengths, bins=bins)
        plt.yscale("log")
        plt.xlabel("Narrative length (words)")
        plt.ylabel("Frequency (log-scale)")
        plt.title("Word-count distribution – clean dataset")
        plt.tight_layout()
        plt.show()

    pctiles = {p: int(np.percentile(lengths, p)) for p in (0,25,50,75,90,95,99,100)}

    return {
        "total"              : total,
        "with_narrative"     : with_text,
        "without_narrative"  : without_txt,
        "word_length_percentiles": pctiles,
    }
