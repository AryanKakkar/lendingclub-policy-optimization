# src/data_utils.py
import pandas as pd
import numpy as np

DEFAULTS = {
    "Charged Off","Default","Late (31-120 days)","Late (16-30 days)",
    "Does not meet the credit policy. Status:Charged Off"
}
PAID = {"Fully Paid","Does not meet the credit policy. Status:Fully Paid"}

# --- helpers that ALWAYS return a 1D Series ---
def _to_series(x, name: str):
    """Return a 1D Series for column 'name' even if a 2D frame sneaks in."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        # More than one column with the same name; take the first non-null per row
        return x.bfill(axis=1).iloc[:, 0]
    # last resort: construct a Series
    return pd.Series(x, name=name)

def pct_to_float(s):
    s = _to_series(s, "pct")
    return pd.to_numeric(s.astype(str).str.replace("%","", regex=False), errors="coerce")/100.0

def term_to_months(s):
    s = _to_series(s, "term")
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)")[0], errors="coerce")

def emp_to_years(s):
    s = _to_series(s, "emp_length")
    s = s.astype(str).str.lower().str.strip()
    s = s.replace({"n/a": np.nan})
    s = s.str.replace("< 1 year","0", regex=False)
    s = s.str.replace("10+ years","10", regex=False)
    return pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")

def load_and_clean(path, num_cols, cat_cols, meta_cols):
    df = pd.read_csv(path, low_memory=False)

    # Normalize headers and drop dup headers from source
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    # Filter statuses and add target
    df = df[df["loan_status"].isin(DEFAULTS | PAID)].copy()
    df["y"] = df["loan_status"].isin(DEFAULTS).astype(int)

    # Parse dates
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_d"]).sort_values("issue_d")

    # UNIQUE keep list (preserve order)
    raw_keep = list(num_cols + cat_cols + meta_cols + ["y"])
    keep = []
    seen = set()
    for c in raw_keep:
        if c in df.columns and c not in seen:
            keep.append(c)
            seen.add(c)

    df = df[keep].copy()
    # Paranoia: remove any accidental dups post-subset
    df = df.loc[:, ~df.columns.duplicated()]

    # Conversions using safe Series extractor
    if "int_rate"   in df.columns: df["int_rate"]   = pct_to_float(df.loc[:, ["int_rate"]] if isinstance(df["int_rate"], pd.DataFrame) else df["int_rate"])
    if "revol_util" in df.columns: df["revol_util"] = pct_to_float(df.loc[:, ["revol_util"]] if "revol_util" in df.columns else df.get("revol_util"))
    if "term"       in df.columns: df["term"]       = term_to_months(df.loc[:, ["term"]] if isinstance(df["term"], pd.DataFrame) else df["term"])
    if "emp_length" in df.columns: df["emp_length"] = emp_to_years(df.loc[:, ["emp_length"]] if "emp_length" in df.columns else df.get("emp_length"))

    # Coerce numerics
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(_to_series(df[c], c), errors="coerce")

    return df

def time_split(df, train_end="2016-12-31", val_year="2017"):
    train = df[df["issue_d"] <= train_end].copy()
    valid = df[(df["issue_d"] >= f"{val_year}-01-01") & (df["issue_d"] < f"{int(val_year)+1}-01-01")].copy()
    test  = df[df["issue_d"] >= f"{int(val_year)+1}-01-01"].copy()
    return train, valid, test

def get_blocks(frame, num_cols, cat_cols):
    X = frame[num_cols + cat_cols].copy()
    y = frame["y"].values.astype(np.int64)
    meta = frame[["loan_amnt","int_rate"]].copy()
    return X, y, meta
