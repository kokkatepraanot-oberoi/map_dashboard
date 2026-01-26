# app.py
import re
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from gdrive_store import list_files_in_folder, download_file_bytes


# ---------------- Page config ----------------
st.set_page_config(page_title="MAP Dashboard", layout="wide")


# ---------------- MAP-ish CSS ----------------
MAP_BLUE = "#2F6FB2"
MAP_YELLOW = "#F2C94C"
MAP_RED = "#C0392B"
MAP_BG = "#F9FAFB"


def inject_map_css():
    st.markdown(
        f"""
        <style>
          .stApp {{ background: {MAP_BG}; }}
          h1,h2,h3 {{ letter-spacing: -0.2px; }}

          .map-pill {{
            display:inline-block; padding:0.18rem 0.55rem; border-radius:999px;
            font-size:0.85rem; font-weight:600; margin-right:0.35rem;
            border:1px solid rgba(0,0,0,0.08);
          }}
          .map-blue {{ background: rgba(47,111,178,0.12); color: {MAP_BLUE}; }}
          .map-yellow {{ background: rgba(242,201,76,0.18); color: #8a6a00; }}
          .map-red {{ background: rgba(192,57,43,0.12); color: {MAP_RED}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_map_css()


# ---------------- Helpers: status, ranges ----------------
def percentile_band(pctl_mid: float | None, pctl_risk=25, pctl_high=10) -> Tuple[str, str]:
    if pctl_mid is None or (isinstance(pctl_mid, float) and np.isnan(pctl_mid)):
        return ("No Data", "map-yellow")
    p = float(pctl_mid)
    if p < pctl_high:
        return ("High Risk (Achievement)", "map-red")
    if p < pctl_risk:
        return ("At Risk (Achievement)", "map-yellow")
    return ("On Track (Achievement)", "map-blue")


def map_badge_row(pctl_risk_val: int, pctl_high_val: int):
    st.markdown(
        f"""
        <span class="map-pill map-blue">On Track</span>
        <span class="map-pill map-yellow">At Risk (below {pctl_risk_val}th percentile)</span>
        <span class="map-pill map-red">High Risk (below {pctl_high_val}th percentile)</span>
        """,
        unsafe_allow_html=True,
    )


def _format_range(low: Optional[float], high: Optional[float], mid: Optional[float]) -> str:
    low_ok = low is not None and not (isinstance(low, float) and np.isnan(low))
    high_ok = high is not None and not (isinstance(high, float) and np.isnan(high))
    mid_ok = mid is not None and not (isinstance(mid, float) and np.isnan(mid))
    if low_ok and high_ok:
        return f"{int(low)}–{int(high)}"
    if mid_ok:
        return f"{int(mid)}"
    if low_ok:
        return f"{int(low)}"
    if high_ok:
        return f"{int(high)}"
    return ""


def add_display_ranges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rit_range"] = out.apply(lambda r: _format_range(r.get("rit_low"), r.get("rit_high"), r.get("rit")), axis=1)
    out["percentile_range"] = out.apply(
        lambda r: _format_range(r.get("percentile_low"), r.get("percentile_high"), r.get("percentile")),
        axis=1
    )
    return out


def apply_flags(df: pd.DataFrame, pctl_risk=25, pctl_high=10) -> pd.DataFrame:
    out = df.copy()
    out["high_risk_achievement"] = out["percentile"].notna() & (out["percentile"] < pctl_high)
    out["at_risk_achievement"] = out["percentile"].notna() & (out["percentile"] < pctl_risk)
    out["intervention_priority"] = out["high_risk_achievement"] | out["at_risk_achievement"]
    out["priority_level"] = np.select(
        [out["high_risk_achievement"], out["at_risk_achievement"]],
        ["High Risk", "At Risk"],
        default="On Track"
    )
    out["priority_reason"] = np.select(
        [out["high_risk_achievement"], out["at_risk_achievement"]],
        [f"Achievement percentile < {pctl_high}", f"Achievement percentile < {pctl_risk}"],
        default=""
    )
    return out

def style_ia_table(df: pd.DataFrame, percentile_status: str):
    """
    Color-code Instructional Areas table based on the student's percentile_status.
    Uses the same MAP palette as the rest of the app.
    """
    if df.empty:
        return df.style

    if percentile_status == "High Risk (Achievement)":
        bg = "rgba(192,57,43,0.12)"
        fg = "#C0392B"
    elif percentile_status == "At Risk (Achievement)":
        bg = "rgba(242,201,76,0.20)"
        fg = "#8a6a00"
    elif percentile_status == "On Track (Achievement)":
        bg = "rgba(47,111,178,0.12)"
        fg = "#2F6FB2"
    else:
        bg = "transparent"
        fg = "inherit"

    def _row(_):
        return [f"background-color: {bg}"] * len(df.columns)

    styler = df.style.apply(_row, axis=1)

    if "Instructional Area" in df.columns:
        styler = styler.map(lambda _: f"font-weight:700; color:{fg};", subset=["Instructional Area"])

    return styler



# ---------------- Styling: color-coded rows ----------------
def _row_bg_color(row: pd.Series):
    status = row.get("percentile_status", "")
    if status == "High Risk (Achievement)":
        bg = "rgba(192,57,43,0.12)"
    elif status == "At Risk (Achievement)":
        bg = "rgba(242,201,76,0.20)"
    elif status == "On Track (Achievement)":
        bg = "rgba(47,111,178,0.12)"
    else:
        bg = "transparent"
    return [f"background-color: {bg}"] * len(row)


def style_map_table(df: pd.DataFrame):
    styler = df.style.apply(_row_bg_color, axis=1)
    if "percentile_status" in df.columns:
        def _status_cell(val):
            if val == "High Risk (Achievement)":
                return "font-weight:700; color:#C0392B;"
            if val == "At Risk (Achievement)":
                return "font-weight:700; color:#8a6a00;"
            if val == "On Track (Achievement)":
                return "font-weight:700; color:#2F6FB2;"
            return ""
        styler = styler.map(_status_cell, subset=["percentile_status"])
    return styler


def show_styled_table(df: pd.DataFrame):
    if df.empty:
        st.info("No records match the current filters.")
        return
    st.dataframe(style_map_table(df), use_container_width=True)


# ---------------- Parser for your "Grade Report" exports ----------------
def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def _find_header_rows(df: pd.DataFrame) -> list[int]:
    hits = []
    for i in range(df.shape[0]):
        if _safe_str(df.iat[i, 0]).lower() == "name (student id)":
            hits.append(i)
    return hits


def _infer_subject(df: pd.DataFrame, header_row: int, fallback: str) -> str:
    lookback = max(0, header_row - 80)
    text_block = df.iloc[lookback:header_row, 0].dropna().astype(str).tolist()
    joined = "\n".join(text_block).lower()
    if "math:" in joined or "math k-12" in joined:
        return "Math"
    if "language usage" in joined:
        return "Language Usage"
    if "reading" in joined:
        return "Reading"
    return fallback


def _extract_name_id(cell: str) -> Tuple[str, str]:
    s = _safe_str(cell)
    m = re.match(r"^(.*)\(([^()]*)\)\s*$", s)
    if not m:
        return (s, "")
    return (m.group(1).strip(), m.group(2).strip())


def _numeric_cols_in_window(row: pd.Series, start: int, end: int) -> list[int]:
    cols = []
    for c in range(start, end + 1):
        v = row.iloc[c]
        if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
            cols.append(c)
    return cols


def _get_range_triplet(row: pd.Series, anchor: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    start = max(0, anchor - 2)
    end = min(len(row) - 1, anchor + 10)
    cols = _numeric_cols_in_window(row, start, end)
    if len(cols) == 0:
        return (None, None, None)
    if len(cols) == 1:
        v = row.iloc[cols[0]]
        return (None, float(v), None)
    low = float(row.iloc[cols[0]])
    mid = float(row.iloc[cols[len(cols) // 2]])
    high = float(row.iloc[cols[-1]])
    return (low, mid, high)


def _find_positions(row: pd.Series, labels: list[str]) -> dict[str, int]:
    pos = {}
    for i, v in enumerate(row):
        s = _safe_str(v)
        for lab in labels:
            if s == lab:
                pos[lab] = i
    return pos


def parse_grade_report_excel(excel_bytes: bytes, grade: int, term_label: str) -> pd.DataFrame:
    xls = pd.ExcelFile(BytesIO(excel_bytes))
    sheet = "Grade Report" if "Grade Report" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(BytesIO(excel_bytes), sheet_name=sheet, header=None)

    header_rows = _find_header_rows(df)
    if not header_rows:
        return pd.DataFrame()

    frames = []
    fallback_order = ["Math", "Reading", "Language Usage"]

    for block_idx, h in enumerate(header_rows):
        fallback = fallback_order[block_idx] if block_idx < len(fallback_order) else f"Subject {block_idx+1}"
        subject = _infer_subject(df, h, fallback=fallback)

        header = df.iloc[h]
        header2 = df.iloc[h + 1] if h + 1 < df.shape[0] else pd.Series([None] * df.shape[1])

        positions = _find_positions(header, ["RIT Score", "Percentile", "Lexile", "Test", "A", "B", "C", "D"])

        test_anchor = positions.get("Test", None)
        date_col = test_anchor + 1 if test_anchor is not None and (test_anchor + 1) < df.shape[1] else None

        duration_col = None
        test_positions = [i for i, v in enumerate(header) if _safe_str(v) == "Test"]
        for tp in test_positions:
            if tp < len(header2) and _safe_str(header2.iloc[tp]).lower() == "duration":
                duration_col = tp + 1 if (tp + 1) < df.shape[1] else None

        rit_anchor = positions.get("RIT Score", None)
        pct_anchor = positions.get("Percentile", None)

        lex_anchor = positions.get("Lexile", None)
        lex_col = lex_anchor + 1 if lex_anchor is not None and (lex_anchor + 1) < df.shape[1] else None

        ia_cols = {lab: positions[lab] for lab in ["A", "B", "C", "D"] if lab in positions}

        name_col = 2 if df.shape[1] > 2 else 0
        start_row = h + 2

        blank_streak = 0
        for r in range(start_row, df.shape[0]):
            name_raw = _safe_str(df.iat[r, name_col] if name_col < df.shape[1] else None)
            if not name_raw:
                blank_streak += 1
                if blank_streak >= 5:
                    break
                continue
            blank_streak = 0

            student_name, student_id = _extract_name_id(name_raw)
            row = df.iloc[r]

            test_date = row.iloc[date_col] if date_col is not None else None

            rit_low, rit_mid, rit_high = (None, None, None)
            if rit_anchor is not None:
                rit_low, rit_mid, rit_high = _get_range_triplet(row, rit_anchor)

            p_low, p_mid, p_high = (None, None, None)
            if pct_anchor is not None:
                p_low, p_mid, p_high = _get_range_triplet(row, pct_anchor)

            duration = _safe_str(row.iloc[duration_col]) if duration_col is not None else ""
            lexile = _safe_str(row.iloc[lex_col]) if (subject.lower() == "reading" and lex_col is not None) else ""

            ia_vals = {f"ia_{k.lower()}": _safe_str(row.iloc[c]) for k, c in ia_cols.items()}

            frames.append(
                {
                    "term": term_label,
                    "grade": grade,
                    "subject": subject,
                    "student_id": student_id,
                    "student_name": student_name,
                    # mid numeric internally for risk + growth
                    "rit": rit_mid,
                    "rit_low": rit_low,
                    "rit_high": rit_high,
                    "percentile": p_mid,
                    "percentile_low": p_low,
                    "percentile_high": p_high,
                    "test_date": test_date,
                    "lexile": lexile if lexile else None,
                    "duration": duration if duration else None,
                    **ia_vals,
                }
            )

    out = pd.DataFrame(frames)
    for c in ["rit", "rit_low", "rit_high", "percentile", "percentile_low", "percentile_high"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[~(out["student_id"].fillna("") == "")]
    return out.reset_index(drop=True)


# ---------------- Drive file naming + term ordering ----------------
FNAME_RE = re.compile(r"^MAP_(\d{4}-\d{2})_(Sep|Mar)_Grade(6|7|8)\.xlsx$", re.IGNORECASE)


def term_date_from_ay_season(academic_year: str, season: str) -> pd.Timestamp:
    """Model A: '2025-26' => Sep 2025, Mar 2026"""
    start_year = int(academic_year.split("-")[0])
    if season.lower() == "sep":
        return pd.Timestamp(year=start_year, month=9, day=1)
    return pd.Timestamp(year=start_year + 1, month=3, day=1)


def parse_drive_filename(name: str):
    m = FNAME_RE.match(name)
    if not m:
        return None
    academic_year = m.group(1)
    season = m.group(2).title()
    grade = int(m.group(3))
    term_label = f"{season} {academic_year}"
    term_date = term_date_from_ay_season(academic_year, season)
    return academic_year, season, grade, term_label, term_date


# ---------------- Sidebar: Drive + ingestion (READ ONLY) ----------------
st.title("MAP Dashboard")
st.caption("Reads MAP Excel files from Google Drive (read-only). Upload files manually to the Drive folder.")

folder_id = st.secrets.get("GDRIVE_FOLDER_ID")
if not folder_id:
    st.error("Missing secret: GDRIVE_FOLDER_ID")
    st.stop()

st.sidebar.header("Data Source (Read-only)")
st.sidebar.caption("Upload files manually to the Drive folder. The app will detect them automatically on refresh.")

st.sidebar.info(
    "✅ **Naming format** (required):\n\n"
    "`MAP_2025-26_Sep_Grade6.xlsx`\n"
    "`MAP_2025-26_Sep_Grade7.xlsx`\n"
    "`MAP_2025-26_Sep_Grade8.xlsx`\n\n"
    "Later:\n"
    "`MAP_2025-26_Mar_Grade6.xlsx` etc."
)

# Load Drive file list
@st.cache_data(show_spinner=False)
def get_drive_map_files(folder_id: str):
    files = list_files_in_folder(folder_id)
    parsed = []
    for f in files:
        meta = parse_drive_filename(f["name"])
        if meta is None:
            continue
        academic_year, season, grade, term_label, term_date = meta
        parsed.append(
            {
                "id": f["id"],
                "name": f["name"],
                "modifiedTime": f.get("modifiedTime"),
                "academic_year": academic_year,
                "season": season,
                "grade": grade,
                "term_label": term_label,
                "term_date": term_date,
            }
        )
    return pd.DataFrame(parsed)

drive_df = get_drive_map_files(folder_id)

st.sidebar.markdown("---")
st.sidebar.subheader("Files found in Drive")
if drive_df.empty:
    st.sidebar.warning("No correctly named MAP_*.xlsx files found yet.")
    st.sidebar.caption("Check naming + that the files are in the configured folder.")
else:
    st.sidebar.success(f"{len(drive_df)} file(s) detected.")
    with st.sidebar.expander("Show detected files", expanded=False):
        st.sidebar.dataframe(drive_df[["name", "modifiedTime"]], use_container_width=True)

if st.sidebar.button("Refresh file list"):
    st.cache_data.clear()
    st.rerun()

# ---------------- Controls ----------------
st.sidebar.header("View")
view_mode = st.sidebar.radio(
    "Select View",
    ["Teacher View", "Admin View", "Student Profile (Leader)", "Growth Trends"],
    index=1,
)

st.sidebar.header("Intervention Priority Rules (Achievement)")
pctl_risk = st.sidebar.slider("At Risk threshold (percentile mid)", 5, 50, 25, 1)
pctl_high = st.sidebar.slider("High Risk threshold (percentile mid)", 1, 25, 10, 1)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <span class="map-pill map-blue">On Track</span><br>
    <span class="map-pill map-yellow">At Risk (below {pctl_risk}th)</span><br>
    <span class="map-pill map-red">High Risk (below {pctl_high}th)</span>
    """,
    unsafe_allow_html=True,
)

# ---------------- Build dataset from Drive ----------------
@st.cache_data(show_spinner=False)
def load_all_data_from_drive(drive_meta: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in drive_meta.iterrows():
        content = download_file_bytes(row["id"])
        df_term = parse_grade_report_excel(content, grade=int(row["grade"]), term_label=row["term_label"])
        if df_term.empty:
            continue
        df_term["academic_year"] = row["academic_year"]
        df_term["season"] = row["season"]
        df_term["term_date"] = row["term_date"]
        frames.append(df_term)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

data = load_all_data_from_drive(drive_df)

if data.empty:
    st.info("No usable MAP data loaded yet. Ensure at least one correctly named file exists in Drive.")
    st.stop()

# Risk status based on MID percentile (as requested)
data = apply_flags(data, pctl_risk=pctl_risk, pctl_high=pctl_high)
data["percentile_status"] = data["percentile"].apply(lambda x: percentile_band(x, pctl_risk, pctl_high)[0])
data = add_display_ranges(data)

# Common filters
grades = ["All"] + sorted(data["grade"].dropna().unique().tolist())
subjects = ["All"] + sorted(data["subject"].dropna().unique().tolist())
terms = ["All"] + [t for t in data.sort_values("term_date")["term"].dropna().unique().tolist()]
academic_years = ["All"] + sorted(data["academic_year"].dropna().unique().tolist())

# Student labels
tmp_students = data[["student_id", "student_name"]].dropna().drop_duplicates()
tmp_students["label"] = tmp_students["student_id"].astype(str) + " — " + tmp_students["student_name"].astype(str)
student_labels = tmp_students.sort_values("label")["label"].tolist()


def metric_row(df: pd.DataFrame):
    total_rows = len(df)
    priority_rows = int(df["intervention_priority"].sum()) if "intervention_priority" in df.columns else 0
    pct_priority = (priority_rows / total_rows * 100) if total_rows else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{total_rows}")
    c2.metric("Intervention Priority", f"{priority_rows}")
    c3.metric("% Priority", f"{pct_priority:.1f}%")


# ---------------- Teacher View ----------------
if view_mode == "Teacher View":
    st.subheader("Teacher View")

    c1, c2, c3, c4 = st.columns(4)
    ay_sel = c1.selectbox("Academic Year", academic_years, index=0)
    term_sel = c2.selectbox("Term", terms, index=0)
    grade_sel = c3.selectbox("Grade", grades, index=0)
    subject_sel = c4.selectbox("Subject", subjects, index=0)

    df = data.copy()
    if ay_sel != "All":
        df = df[df["academic_year"] == ay_sel]
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
        ["_priority_sort", "grade", "subject", "percentile", "rit"],
        ascending=[True, True, True, True, True]
    )

    show_cols = [
        "academic_year", "term", "grade", "subject",
        "student_id", "student_name",
        "rit_range", "percentile_range",
        "percentile_status", "priority_reason",
        "lexile", "duration"
    ]
    show_cols = [c for c in show_cols if c in priority_df.columns]
    show_styled_table(priority_df[show_cols])

    st.download_button(
        "Download intervention list (CSV)",
        data=priority_df[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="intervention_priority_teacher_view.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("### All Students (filtered)")
    show_styled_table(df[show_cols])


# ---------------- Admin View ----------------
elif view_mode == "Admin View":
    st.subheader("Admin View")

    c1, c2, c3, c4 = st.columns(4)
    ay_sel = c1.selectbox("Academic Year", academic_years, index=0)
    term_sel = c2.selectbox("Term", terms, index=0)
    grade_sel = c3.selectbox("Grade", grades, index=0)
    subject_sel = c4.selectbox("Subject", subjects, index=0)

    df = data.copy()
    if ay_sel != "All":
        df = df[df["academic_year"] == ay_sel]
    if term_sel != "All":
        df = df[df["term"] == term_sel]
    if grade_sel != "All":
        df = df[df["grade"] == grade_sel]
    if subject_sel != "All":
        df = df[df["subject"] == subject_sel]

    metric_row(df)
    map_badge_row(pctl_risk, pctl_high)
    st.divider()

    st.markdown("### Intervention Priority (Schoolwide / Filtered)")
    priority_sort_key = {"High Risk": 0, "At Risk": 1, "On Track": 2}
    df["_priority_sort"] = df["priority_level"].map(priority_sort_key).fillna(9)
    priority_df = df[df["intervention_priority"]].sort_values(
        ["_priority_sort", "grade", "subject", "percentile", "rit"],
        ascending=[True, True, True, True, True],
    )

    show_cols_admin = [
        "academic_year", "term", "grade", "subject",
        "student_id", "student_name",
        "rit_range", "percentile_range",
        "percentile_status", "priority_reason",
        "lexile", "duration"
    ]
    show_cols_admin = [c for c in show_cols_admin if c in priority_df.columns]
    show_styled_table(priority_df[show_cols_admin])

    st.download_button(
        "Download intervention list (CSV)",
        data=priority_df[show_cols_admin].to_csv(index=False).encode("utf-8"),
        file_name="intervention_priority_admin_view.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown("### Summary by Academic Year × Term × Grade × Subject (mid values used internally)")
    summary = (
        df.groupby(["academic_year", "term", "grade", "subject"], dropna=False)
        .agg(
            records=("student_id", "count"),
            intervention_priority=("intervention_priority", "sum"),
            avg_rit_mid=("rit", "mean"),
            avg_percentile_mid=("percentile", "mean"),
        )
        .reset_index()
    )
    summary["pct_intervention_priority"] = (summary["intervention_priority"] / summary["records"] * 100).round(1)
    summary = summary.sort_values(["academic_year", "term", "grade", "subject"])
    st.dataframe(summary, use_container_width=True)

# ---------------- Student Profile (Leader) ----------------
elif view_mode == "Student Profile (Leader)":
    st.subheader("Student Profile (Leader)")

    f1, f2, f3 = st.columns(3)

    # Stable keys so Streamlit state behaves correctly
    ay_sel = f1.selectbox("Academic Year (optional)", academic_years, index=0, key="sp_ay")
    grade_sel = f2.selectbox("Grade (optional)", grades, index=0, key="sp_grade")

    # Build the student dropdown AFTER filters so it refreshes properly
    student_pool = data.copy()
    if ay_sel != "All":
        student_pool = student_pool[student_pool["academic_year"] == ay_sel]
    if grade_sel != "All":
        student_pool = student_pool[student_pool["grade"] == grade_sel]

    tmp_students = student_pool[["student_id", "student_name"]].dropna().drop_duplicates().copy()
    tmp_students["label"] = tmp_students["student_id"].astype(str) + " — " + tmp_students["student_name"].astype(str)
    student_options = tmp_students.sort_values("label")["label"].tolist()

    if not student_options:
        st.warning("No students found for the selected filters.")
        st.stop()

    # Reset selection if the previously selected value is no longer valid
    key_student = "sp_student"
    current = st.session_state.get(key_student)
    if current not in student_options:
        st.session_state[key_student] = student_options[0]

    student_label = f3.selectbox("Select Student", student_options, key=key_student)

    # Robust split
    student_id = student_label.split("—", 1)[0].strip()
    if not student_id:
        st.info("Student ID not detected.")
        st.stop()

    # Student records (then apply same filters)
    s_df = data[data["student_id"].astype(str) == str(student_id)].copy()
    if ay_sel != "All":
        s_df = s_df[s_df["academic_year"] == ay_sel]
    if grade_sel != "All":
        s_df = s_df[s_df["grade"] == grade_sel]

    if s_df.empty:
        st.info("No records found for this selection.")
        st.stop()

    # Safe sort
    if "term_date" in s_df.columns:
        s_df = s_df.sort_values("term_date")
    else:
        s_df = s_df.sort_values("term")

    student_name = s_df["student_name"].dropna().iloc[0]
    latest = s_df.iloc[-1]
    overall_status = "Intervention Priority" if bool(latest.get("intervention_priority", False)) else "On Track"

    h1, h2, h3 = st.columns(3)
    h1.metric("Student", student_name)
    h2.metric("Latest Term", str(latest.get("term", "")))
    h3.metric("Current Status", overall_status)

    map_badge_row(pctl_risk, pctl_high)
    st.divider()

    # ---------- Achievement Summary ----------
    st.markdown("### Achievement Summary (Full Ranges)")

    needed = ["subject", "term", "rit_range", "percentile_range", "percentile_status"]
    if "term_date" in s_df.columns:
        needed.insert(2, "term_date")

    core = s_df[[c for c in needed if c in s_df.columns]].copy()
    if "term_date" not in core.columns:
        core["term_date"] = pd.NaT

    wide_rit = core.pivot_table(index="subject", columns="term", values="rit_range", aggfunc="first")
    wide_pct = core.pivot_table(index="subject", columns="term", values="percentile_range", aggfunc="first")

    out = pd.DataFrame(index=sorted(core["subject"].dropna().unique()))
    term_order = core.sort_values("term_date")["term"].dropna().unique().tolist()

    for t in term_order:
        if t in wide_rit.columns:
            out[f"{t} RIT Range"] = wide_rit[t]
        if t in wide_pct.columns:
            out[f"{t} Percentile Range"] = wide_pct[t]

    if "percentile_status" in core.columns:
        latest_status = (
            core.sort_values("term_date")
            .groupby("subject")
            .tail(1)
            .set_index("subject")["percentile_status"]
        )
        out["Achievement Status (Latest)"] = out.index.map(latest_status)
    else:
        out["Achievement Status (Latest)"] = ""

    st.dataframe(out.reset_index().rename(columns={"index": "Subject"}), width="stretch")

    st.divider()

    # ---------- Instructional Areas (Diagnostic Detail) ----------
    st.markdown("### Instructional Areas (Diagnostic Detail)")

    subj_list = sorted(s_df["subject"].dropna().unique().tolist())
    if not subj_list:
        st.info("No subjects found for this student.")
        st.stop()

    selected_subject = st.selectbox("Select Subject", subj_list, key="sp_subject")

    subj_df = s_df[s_df["subject"] == selected_subject].copy()
    if "term_date" in subj_df.columns:
        subj_df = subj_df.sort_values("term_date")
    else:
        subj_df = subj_df.sort_values("term")

    # Only A/B/C/D value columns (exclude *_name helper columns)
    ia_val_cols = [c for c in subj_df.columns if c.startswith("ia_") and not c.endswith("_name")]

    if not ia_val_cols:
        st.info("No instructional area columns detected in this export for this subject.")
    else:
        for _, row in subj_df.iterrows():
            st.markdown(f"#### {selected_subject} — {row.get('term','')}")
            st.markdown(
                f"**RIT Range:** {row.get('rit_range','')} &nbsp;&nbsp; "
                f"**Percentile Range:** {row.get('percentile_range','')} &nbsp;&nbsp; "
                f"**Status:** {row.get('percentile_status','')}"
            )

            # Build A/B/C/D table with names if present (ia_a_name, etc.)
            ia_rows = []
            for c in sorted(ia_val_cols):
                letter = c.replace("ia_", "").upper()  # A/B/C/D
                name = row.get(f"{c}_name", "")
                band = row.get(c, "")

                name_str = "" if (name is None or str(name).strip().lower() == "nan") else str(name).strip()
                band_str = "" if (band is None or str(band).strip().lower() == "nan") else str(band).strip()

                label = f"{letter} — {name_str}" if name_str else f"{letter}"
                ia_rows.append({"Instructional Area": label, "Band / Range": band_str})

            ia_table = pd.DataFrame(ia_rows)

            # MAP colour-coded table (uses student's status colour)
            st.dataframe(style_ia_table(ia_table, row.get("percentile_status", "")), width="stretch")
            st.divider()

    # ---------- Raw Records ----------
    with st.expander("Show raw records (audit-safe)"):
        cols = [
            "academic_year", "term", "grade", "subject", "student_id", "student_name",
            "rit_range", "percentile_range", "percentile_status", "priority_reason",
            "lexile", "duration",
        ]
        cols += [c for c in s_df.columns if c.startswith("ia_")]
        cols = [c for c in cols if c in s_df.columns]

        # sort BEFORE selecting cols (prevents KeyError on term_date)
        if "term_date" in s_df.columns:
            sorted_s = s_df.sort_values(["subject", "term_date"], ascending=[True, True])
        else:
            sorted_s = s_df.sort_values(["subject", "term"], ascending=[True, True])

        show_styled_table(sorted_s[cols])



# ---------------- Growth Trends ----------------
else:
    st.subheader("Growth Trends")

    if not student_labels:
        st.warning("No students detected in the stored Drive files.")
        st.stop()

    mode = st.radio("View mode", ["Term sequence (timeline)", "Academic year (Sep→Mar)"], horizontal=True)

    c1, c2 = st.columns(2)
    student_label = c1.selectbox("Student", student_labels)
    subject_sel = c2.selectbox("Subject", sorted(data["subject"].dropna().unique().tolist()))

    student_id = student_label.split("—", 1)[0].strip()
    s = data[(data["student_id"].astype(str) == str(student_id)) & (data["subject"] == subject_sel)].copy()
    s = s.sort_values("term_date")

    if s.empty:
        st.info("No records found for this student/subject yet.")
        st.stop()

    st.markdown("### RIT (mid) over time")
    chart_df = s[["term_date", "rit"]].dropna().rename(columns={"term_date": "Term", "rit": "RIT (mid)"})
    chart_df = chart_df.set_index("Term")
    st.line_chart(chart_df)

    st.markdown("### Observed Growth (RIT mid)")
    s["Observed Growth (RIT mid)"] = s["rit"].diff()
    growth_table = s[["academic_year", "term", "grade", "rit_range", "Observed Growth (RIT mid)", "percentile_status"]].copy()
    show_styled_table(growth_table)

    if mode.startswith("Academic year"):
        st.divider()
        st.markdown("### Academic year growth (Sep → Mar)")
        pivot = s.pivot_table(index="academic_year", columns="season", values="rit", aggfunc="first")
        pivot = pivot.rename(columns={"Sep": "Sep RIT (mid)", "Mar": "Mar RIT (mid)"})
        pivot["Year Growth (RIT mid)"] = pivot.get("Mar RIT (mid)") - pivot.get("Sep RIT (mid)")
        st.dataframe(pivot.reset_index(), use_container_width=True)

        st.caption("Year growth uses RIT mid internally; tables elsewhere show full ranges (low–high).")

