# flags.py
import pandas as pd
import numpy as np

def apply_flags(df: pd.DataFrame, pctl_risk=25, pctl_high=10) -> pd.DataFrame:
    out = df.copy()

    out["flag_high"] = out["percentile"].notna() & (out["percentile"] < pctl_high)
    out["flag_risk"] = out["percentile"].notna() & (out["percentile"] < pctl_risk)

    # A single field that explains why
    reasons = []
    for _, r in out.iterrows():
        why = []
        if r.get("flag_high"): why.append(f"Percentile < {pctl_high}")
        elif r.get("flag_risk"): why.append(f"Percentile < {pctl_risk}")
        reasons.append(", ".join(why))
    out["flag_reason"] = reasons

    out["flagged"] = out["flag_high"] | out["flag_risk"]
    return out
