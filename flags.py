# flags.py
import pandas as pd

def apply_flags(df: pd.DataFrame, pctl_risk=25, pctl_high=10) -> pd.DataFrame:
    """
    MAP-aligned labels:
    - Percentile colours indicate achievement percentile rank (relative standing). :contentReference[oaicite:1]{index=1}
    """
    out = df.copy()

    out["high_risk_achievement"] = out["percentile"].notna() & (out["percentile"] < pctl_high)
    out["at_risk_achievement"] = out["percentile"].notna() & (out["percentile"] < pctl_risk)

    # Intervention Priority (MAP-aligned wording)
    out["intervention_priority"] = out["high_risk_achievement"] | out["at_risk_achievement"]

    def reason(row):
        if row.get("high_risk_achievement"):
            return f"Achievement percentile < {pctl_high}"
        if row.get("at_risk_achievement"):
            return f"Achievement percentile < {pctl_risk}"
        return ""

    out["priority_reason"] = out.apply(reason, axis=1)

    # Optional: helpful categorical field for sorting
    out["priority_level"] = out.apply(
        lambda r: "High Risk" if r["high_risk_achievement"]
        else ("At Risk" if r["at_risk_achievement"] else "On Track"),
        axis=1
    )

    return out
