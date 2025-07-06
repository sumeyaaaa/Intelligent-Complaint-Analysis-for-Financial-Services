# rag_finance/src/filter_clean.py
"""
Stream-filter to the five target products and deep-clean narratives.
Writes TWO files in one pass:
  • complaints_filtered.csv  – only product / non-empty rows, minimal clean
  • complaints_clean.csv     – same rows but boilerplate + special chars removed
"""

import gc, os, re, yaml, pathlib, pandas as pd

_CFG = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "config.yaml"))

# boiler-plate phrases to strip (add more as needed)
BOILER_PATTERNS = [
    r"i am writing to file a complaint",
    r"to whom it may concern",
    r"dear sir or madam",
    r"please help me resolve",
    r"thank you for your time"
]
_boiler_re  = re.compile("|".join(BOILER_PATTERNS), re.IGNORECASE)
_special_re = re.compile(r"[^a-z0-9.,!? \n]")
_multi_ws   = re.compile(r"\s+")

def _clean(text: str) -> str:
    text = text.lower()
    text = _boiler_re.sub(" ", text)
    text = text.replace("&amp;", "&")
    text = _special_re.sub(" ", text)
    text = _multi_ws.sub(" ", text)
    return text.strip()

def filter_and_clean(raw_csv: str | None = None,
                     filtered_csv: str | None = None,
                     clean_csv: str | None = None,
                     products: list[str] | None = None,
                     chunk_size: int | None = None) -> tuple[pathlib.Path, pathlib.Path]:
    raw_csv      = raw_csv      or _CFG["raw_csv"]
    filtered_csv = pathlib.Path(filtered_csv or _CFG["filtered_csv"])
    clean_csv    = pathlib.Path(clean_csv    or _CFG["clean_csv"])
    products     = set(products or _CFG["target_products"])
    chunk_size   = chunk_size   or _CFG["chunk_size"]

    filtered_csv.parent.mkdir(parents=True, exist_ok=True)
    for f in (filtered_csv, clean_csv):
        if f.exists(): f.unlink()

    header_written = False
    reader = pd.read_csv(raw_csv, chunksize=chunk_size, low_memory=False)

    for i, chunk in enumerate(reader, 1):
        narr = chunk["Consumer complaint narrative"]
        mask = chunk["Product"].isin(products) & narr.notna() & (narr.str.strip() != "")
        sub  = chunk.loc[mask, ["Complaint ID", "Product", "Consumer complaint narrative"]]
        if sub.empty:
            continue

        # minimal clean → filtered file
        sub.to_csv(filtered_csv, mode="a", index=False, header=not header_written)

        # deep clean → clean file
        sub = sub.copy()
        sub["Consumer complaint narrative"] = sub["Consumer complaint narrative"].apply(_clean)
        sub.to_csv(clean_csv, mode="a", index=False, header=not header_written)

        header_written = True
        size_mb = clean_csv.stat().st_size // 1_048_576
        print(f"chunk {i}  written  (clean file ≈ {size_mb} MB)", end="\r")

    gc.collect()
    return filtered_csv, clean_csv
