import streamlit as st
import pandas as pd

from data_loader import load_map_excel
from flags import apply_flags

st.set_page_config(page_title="MAP Dashboard", layout="wide")

st.title("MAP Dashboard (Sep 2025 → Mar 2026 Growth)")

# -------- Sidebar controls --------
st.sidebar.header("Data")
fall_file = st.sidebar.file_uploader("Upload Fall (Sep 2025) MAP Excel", type=["xlsx"])
spring_file = st.sidebar.file_uploader("Upload Spring (Mar 2026) MAP Excel (optional)", type=["xlsx"])

st.sidebar.header("View")
view_mode = st.sidebar.radio("Select View", ["Teacher View", "Admin View"], index=0)

st.sidebar.header("Flag Rules")
pctl_risk = st.sidebar.slider("Risk threshold (percentile)", 5, 50, 25, 1)
pctl_high = st.sidebar.slider("High-risk threshold (percentile)", 1, 25, 10, 1)

@st.cache_data(show_spinner=False)
def load_all(fall_bytes, spring_bytes):
    frames = []
    if fall_bytes:
        frames.append(load_map_excel(fall_bytes, term_label="Sep 2025"))
    if spring_bytes:
        frames.append(load_map_excel(spring_bytes, term_label="Mar 2026"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

data = load_all(fall_file, spring_file)

if data.empty:
    st.info("Upload at least the Fall (Sep 2025) MAP file to begin.")
    st.stop()

data = apply_flags(data, pctl_risk=pctl_risk, pctl_high=pctl_high)

# Common filters
subjects = ["All"] + sorted(data["subject"].dropna().unique().tolist())
terms = ["All"] + sorted(data["term"].dropna().unique().tolist())
grades = ["All"] + sorted([int(x) for x in data["grade"].dropna().unique().tolist()])

# -------- Teacher view --------
if view_mode == "Teacher View":
    st.subheader("Teacher View")

    teacher_list = sorted(data["teacher"].dropna().unique().tolist())
    teacher = st.selectbox("Select Teacher", teacher_list)

    c1, c2, c3 = st.columns(3)
    term_sel = c1.selectbox("Term", terms, index=0)
    grade_sel = c2.selectbox("Grade", grades, index=0)
    subject_sel = c3.selectbox("Subject", subjects, index=0)

    df = data[data["teacher"] == teacher].copy()
    if term_sel != "All":
        df = df[df["term"] == term_sel]
    if grade_sel != "All":
        df = df[df["grade"] == grade_sel]
    if subject_sel != "All":
        df = df[df["subject"] == subject_sel]

    # KPIs
    total = len(df)
    flagged = int(df["flagged"].sum())
    st.metric("Students (rows)", total)
    st.metric("Flagged (rows)", flagged)

    st.divider()
    st.markdown("### Flagged Students")
    flagged_df = df[df["flagged"]].sort_values(["subject", "percentile", "rit"], ascending=[True, True, True])

    show_cols = ["student_id", "student_name", "grade", "section", "subject", "rit", "percentile", "flag_reason"]
    st.dataframe(flagged_df[show_cols], use_container_width=True)

    st.download_button(
        "Download flagged list (CSV)",
        data=flagged_df[show_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"flagged_{teacher.replace(' ', '_')}.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("### All Students (filtered)")
    st.dataframe(df[show_cols], use_container_width=True)

# -------- Admin view --------
else:
    st.subheader("Admin View")

    c1, c2, c3, c4 = st.columns(4)
    term_sel = c1.selectbox("Term", terms, index=0)
    grade_sel = c2.selectbox("Grade", grades, index=0)
    subject_sel = c3.selectbox("Subject", subjects, index=0)
    teacher_sel = c4.selectbox("Teacher", ["All"] + sorted(data["teacher"].dropna().unique().tolist()), index=0)

    df = data.copy()
    if term_sel != "All":
        df = df[df["term"] == term_sel]
    if grade_sel != "All":
        df = df[df["grade"] == grade_sel]
    if subject_sel != "All":
        df = df[df["subject"] == subject_sel]
    if teacher_sel != "All":
        df = df[df["teacher"] == teacher_sel]

    # KPIs
    total = len(df)
    flagged = int(df["flagged"].sum())
    pct_flagged = (flagged / total * 100) if total else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("Rows", total)
    k2.metric("Flagged rows", flagged)
    k3.metric("% flagged", f"{pct_flagged:.1f}%")

    st.divider()
    st.markdown("### Flagged Students (Schoolwide)")
    flagged_df = df[df["flagged"]].sort_values(["grade", "subject", "percentile"], ascending=[True, True, True])

    show_cols = ["teacher", "student_id", "student_name", "grade", "section", "subject", "rit", "percentile", "flag_reason", "term"]
    st.dataframe(flagged_df[show_cols], use_container_width=True)

    st.download_button(
        "Download flagged list (CSV)",
        data=flagged_df[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="flagged_schoolwide.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("### Summary by Grade + Subject")
    summary = (
        df.groupby(["grade", "subject"], dropna=False)
          .agg(
              rows=("student_id", "count"),
              flagged=("flagged", "sum"),
              avg_rit=("rit", "mean"),
              avg_percentile=("percentile", "mean"),
          )
          .reset_index()
    )
    summary["pct_flagged"] = (summary["flagged"] / summary["rows"] * 100).round(1)
    st.dataframe(summary, use_container_width=True)
