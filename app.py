# app.py
import streamlit as st
import pandas as pd

from data_loader import load_map_excel
from flags import apply_flags
from ui_theme import inject_map_css, percentile_band

# ---------------- Page config + MAP styling ----------------
st.set_page_config(page_title="MAP Dashboard", layout="wide")
inject_map_css()

st.title("MAP Dashboard")
st.caption(
    "MAP notes: Percentile indicates relative achievement vs norms; growth is evaluated using RIT change "
    "(Observed Growth) once Spring data is available."
)

# ---------------- Sidebar controls ----------------
st.sidebar.header("Data Uploads")
fall_file = st.sidebar.file_uploader("Upload Fall (Sep 2025) MAP Excel", type=["xlsx"])
spring_file = st.sidebar.file_uploader("Upload Spring (Mar 2026) MAP Excel (optional)", type=["xlsx"])

st.sidebar.header("View")
view_mode = st.sidebar.radio(
    "Select View",
    ["Teacher View", "Admin View", "Student Profile (Leader)"],
    index=0,
)

st.sidebar.header("Intervention Priority Rules (Achievement)")
pctl_risk = st.sidebar.slider("At Risk threshold (percentile)", 5, 50, 25, 1)
pctl_high = st.sidebar.slider("High Risk threshold (percentile)", 1, 25, 10, 1)

# MAP-style legend pills
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <span class="map-pill map-blue">On Track</span><br>
    <span class="map-pill map-yellow">At Risk (below {pctl_risk}th)</span><br>
    <span class="map-pill map-red">High Risk (below {pctl_high}th)</span>
    """,
    unsafe_allow_html=True,
)


# ---------------- Data loading ----------------
@st.cache_data(show_spinner=False)
def load_all(fall_upload, spring_upload) -> pd.DataFrame:
    frames = []
    if fall_upload is not None:
        frames.append(load_map_excel(fall_upload.getvalue(), term_label="Sep 2025"))
    if spring_upload is not None:
        frames.append(load_map_excel(spring_upload.getvalue(), term_label="Mar 2026"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


data = load_all(fall_file, spring_file)

if data.empty:
    st.info("Upload at least the Fall (Sep 2025) MAP file to begin.")
    st.stop()

# Apply MAP-aligned intervention priority
data = apply_flags(data, pctl_risk=pctl_risk, pctl_high=pctl_high)

# Add percentile status (MAP-style interpretation)
data["percentile_status"] = data["percentile"].apply(
    lambda x: percentile_band(x, pctl_risk=pctl_risk, pctl_high=pctl_high)[0]
)

# ---------------- Common filter options ----------------
subjects = ["All"] + sorted(data["subject"].dropna().unique().tolist())
terms = ["All"] + sorted(data["term"].dropna().unique().tolist())

grade_vals = sorted([int(x) for x in data["grade"].dropna().unique().tolist()]) if "grade" in data.columns else []
grades = ["All"] + grade_vals

teacher_vals = sorted(data["teacher"].dropna().unique().tolist()) if "teacher" in data.columns else []
teachers = ["All"] + teacher_vals

# Student list labels for the profile view
student_vals = []
if "student_id" in data.columns and "student_name" in data.columns:
    tmp = data[["student_id", "student_name"]].dropna().drop_duplicates()
    tmp["label"] = tmp["student_id"].astype(str).str.strip() + " — " + tmp["student_name"].astype(str).str.strip()
    student_vals = tmp.sort_values("label")["label"].tolist()


def map_badge_row(pctl_risk_val: int, pctl_high_val: int):
    st.markdown(
        f"""
        <span class="map-pill map-blue">On Track</span>
        <span class="map-pill map-yellow">At Risk (below {pctl_risk_val}th percentile)</span>
        <span class="map-pill map-red">High Risk (below {pctl_high_val}th percentile)</span>
        """,
        unsafe_allow_html=True,
    )


def metric_row(df: pd.DataFrame):
    total_rows = len(df)
    priority_rows = int(df["intervention_priority"].sum()) if "intervention_priority" in df.columns else 0
    pct_priority = (priority_rows / total_rows * 100) if total_rows else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Records (rows)", f"{total_rows}")
    c2.metric("Intervention Priority (rows)", f"{priority_rows}")
    c3.metric("% Intervention Priority", f"{pct_priority:.1f}%")


# ----------------- Table styling (color-coded) -----------------
def _row_bg_color(row: pd.Series):
    status = row.get("percentile_status", "")
    if status == "High Risk (Achievement)":
        bg = "rgba(192,57,43,0.12)"   # red tint
    elif status == "At Risk (Achievement)":
        bg = "rgba(242,201,76,0.20)"  # yellow tint
    elif status == "On Track (Achievement)":
        bg = "rgba(47,111,178,0.12)"  # blue tint
    else:
        bg = "transparent"
    return [f"background-color: {bg}"] * len(row)


def style_map_table(df: pd.DataFrame, highlight_status_col: bool = True):
    styler = (
        df.style
        .apply(_row_bg_color, axis=1)
        .format({"rit": "{:.0f}", "percentile": "{:.0f}"}, na_rep="")
    )

    if highlight_status_col and "percentile_status" in df.columns:
        def _status_cell(val):
            if val == "High Risk (Achievement)":
                return "font-weight: 700; color: #C0392B;"
            if val == "At Risk (Achievement)":
                return "font-weight: 700; color: #8a6a00;"
            if val == "On Track (Achievement)":
                return "font-weight: 700; color: #2F6FB2;"
            return ""
        styler = styler.map(_status_cell, subset=["percentile_status"])

    styler = styler.set_table_styles(
        [
            {"selector": "th", "props": [("background-color", "white"), ("color", "#111827")]},
            {"selector": "td", "props": [("border-color", "rgba(0,0,0,0.06)")]},
        ]
    )
    return styler


def show_styled_table(df: pd.DataFrame, use_container_width: bool = True):
    if df.empty:
        st.info("No records match the current filters.")
        return
    st.dataframe(style_map_table(df), use_container_width=use_container_width)


# ---------------- Teacher View ----------------
if view_mode == "Teacher View":
    st.subheader("Teacher View")

    if not teacher_vals:
        st.warning("No teacher column detected in the data.")
        st.stop()

    teacher = st.selectbox("Select Teacher", teacher_vals)

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

    metric_row(df)
    map_badge_row(pctl_risk, pctl_high)
    st.divider()

    st.markdown("### Intervention Priority Students")
    priority_sort_key = {"High Risk": 0, "At Risk": 1, "On Track": 2}
    df["_priority_sort"] = df["priority_level"].map(priority_sort_key).fillna(9)

    priority_df = df[df["intervention_priority"]].sort_values(
        ["_priority_sort", "subject", "percentile", "rit"],
        ascending=[True, True, True, True]
    )

    show_cols = [
        "student_id", "student_name", "grade", "section", "subject",
        "rit", "percentile", "percentile_status", "priority_reason"
    ]
    show_styled_table(priority_df[show_cols])

    st.download_button(
        "Download intervention list (CSV)",
        data=priority_df[show_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"intervention_priority_{teacher.replace(' ', '_')}.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("### All Students (filtered)")
    show_styled_table(df[show_cols])

    df.drop(columns=["_priority_sort"], inplace=True, errors="ignore")


# ---------------- Admin View ----------------
elif view_mode == "Admin View":
    st.subheader("Admin View")

    c1, c2, c3, c4 = st.columns(4)
    term_sel = c1.selectbox("Term", terms, index=0)
    grade_sel = c2.selectbox("Grade", grades, index=0)
    subject_sel = c3.selectbox("Subject", subjects, index=0)
    teacher_sel = c4.selectbox("Teacher", teachers, index=0)

    df = data.copy()
    if term_sel != "All":
        df = df[df["term"] == term_sel]
    if grade_sel != "All":
        df = df[df["grade"] == grade_sel]
    if subject_sel != "All":
        df = df[df["subject"] == subject_sel]
    if teacher_sel != "All":
        df = df[df["teacher"] == teacher_sel]

    metric_row(df)
    map_badge_row(pctl_risk, pctl_high)
    st.divider()

    st.markdown("### Intervention Priority (Schoolwide)")
    priority_sort_key = {"High Risk": 0, "At Risk": 1, "On Track": 2}
    df["_priority_sort"] = df["priority_level"].map(priority_sort_key).fillna(9)

    priority_df = df[df["intervention_priority"]].sort_values(
        ["_priority_sort", "grade", "subject", "percentile", "rit"],
        ascending=[True, True, True, True, True],
    )

    show_cols_admin = [
        "teacher", "student_id", "student_name", "grade", "section", "subject",
        "rit", "percentile", "percentile_status", "priority_reason", "term"
    ]
    show_styled_table(priority_df[show_cols_admin])

    st.download_button(
        "Download intervention list (CSV)",
        data=priority_df[show_cols_admin].to_csv(index=False).encode("utf-8"),
        file_name="intervention_priority_schoolwide.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("### Summary by Grade × Subject")

    summary = (
        df.groupby(["grade", "subject"], dropna=False)
          .agg(
              records=("student_id", "count"),
              intervention_priority=("intervention_priority", "sum"),
              avg_rit=("rit", "mean"),
              avg_percentile=("percentile", "mean"),
          )
          .reset_index()
    )
    summary["pct_intervention_priority"] = (
        (summary["intervention_priority"] / summary["records"] * 100).round(1)
    )
    summary = summary.sort_values(["grade", "subject"])
    st.dataframe(summary, use_container_width=True)

    df.drop(columns=["_priority_sort"], inplace=True, errors="ignore")


# ---------------- Student Profile (Leader) ----------------
else:
    st.subheader("Student Profile (Leader)")

    if not student_vals:
        st.warning("Student ID / Student Name columns not detected.")
        st.stop()

    # Leader filters
    f1, f2 = st.columns(2)
    grade_sel = f1.selectbox("Grade (optional)", grades, index=0)
    teacher_sel = f2.selectbox("Teacher (optional)", teachers, index=0)

    df0 = data.copy()
    if grade_sel != "All":
        df0 = df0[df0["grade"] == grade_sel]
    if teacher_sel != "All":
        df0 = df0[df0["teacher"] == teacher_sel]

    tmp = df0[["student_id", "student_name"]].dropna().drop_duplicates()
    tmp["label"] = tmp["student_id"].astype(str) + " — " + tmp["student_name"].astype(str)
    student_list = tmp.sort_values("label")["label"].tolist()

    student_label = st.selectbox("Select Student", student_list)
    student_id = student_label.split("—")[0].strip()

    s_df = data[data["student_id"].astype(str) == str(student_id)].copy()
    if s_df.empty:
        st.info("No records found for this student.")
        st.stop()

    # Student header
    student_name = s_df["student_name"].iloc[0]
    grade = s_df["grade"].iloc[0]
    section = s_df["section"].iloc[0]
    teacher = s_df["teacher"].iloc[0]

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Student", student_name)
    h2.metric("Grade / Section", f"{int(grade)} / {section}")
    h3.metric("Teacher", teacher)

    latest_term = "Mar 2026" if "Mar 2026" in s_df["term"].unique() else "Sep 2025"
    latest = s_df[s_df["term"] == latest_term]
    overall_status = "Intervention Priority" if latest["intervention_priority"].any() else "On Track"
    h4.metric("Current Status", f"{overall_status} ({latest_term})")

    map_badge_row(pctl_risk, pctl_high)
    st.divider()

    # Achievement Summary
    st.markdown("### Achievement Summary")

    core = s_df[["subject", "term", "rit", "percentile", "percentile_status"]].copy()

    wide = core.pivot_table(
        index="subject",
        columns="term",
        values=["rit", "percentile"],
        aggfunc="first"
    )

    if isinstance(wide.columns, pd.MultiIndex):
        wide.columns = [f"{c[1]} {c[0].upper()}" for c in wide.columns]

    wide = wide.reset_index()

    # Observed Growth (RIT)
    if "Sep 2025 RIT" in wide.columns and "Mar 2026 RIT" in wide.columns:
        wide["Observed Growth (RIT)"] = wide["Mar 2026 RIT"] - wide["Sep 2025 RIT"]

    # Latest percentile status per subject
    latest_status = (
        latest[["subject", "percentile_status"]]
        .drop_duplicates()
        .set_index("subject")
    )
    wide["Achievement Status"] = wide["subject"].map(latest_status["percentile_status"])

    st.dataframe(wide.sort_values("subject"), use_container_width=True)

    st.caption(
        "Percentile indicates relative standing compared to national norms for the same grade and term. "
        "Observed Growth reflects the change in RIT score across test events."
    )

    st.divider()

    # Instructional Areas
    st.markdown("### Instructional Areas")

    subj_list = sorted(s_df["subject"].unique().tolist())
    selected_subject = st.selectbox("Select Subject", subj_list)

    subj_df = s_df[s_df["subject"] == selected_subject].sort_values("term")
    ia_cols = [c for c in subj_df.columns if c.startswith("ia_")]

    if not ia_cols:
        st.info("No instructional area data available for this subject in the export.")
    else:
        for term in subj_df["term"].unique():
            row = subj_df[subj_df["term"] == term].iloc[0]

            st.markdown(f"#### {selected_subject} — {term}")
            st.markdown(
                f"**RIT:** {int(row['rit']) if pd.notna(row['rit']) else ''} &nbsp;&nbsp; "
                f"**Percentile:** {int(row['percentile']) if pd.notna(row['percentile']) else ''} &nbsp;&nbsp; "
                f"**Status:** {row['percentile_status']}"
            )

            ia_table = pd.DataFrame({
                "Instructional Area": [
                    c.replace("ia_", "").replace("_", " ").title()
                    for c in ia_cols
                ],
                "RIT Score": [row[c] for c in ia_cols],
            })

            st.dataframe(ia_table, use_container_width=True)
            st.divider()

    # Raw records (audit-safe)
    with st.expander("Show raw MAP records for this student"):
        audit_cols = [
            "term", "subject", "rit", "percentile",
            "percentile_status", "priority_reason",
            "grade", "section", "teacher"
        ]
        audit_cols = [c for c in audit_cols if c in s_df.columns]
        show_styled_table(s_df[audit_cols].sort_values(["subject", "term"]))
