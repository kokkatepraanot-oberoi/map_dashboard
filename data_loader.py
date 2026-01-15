# data_loader.py
import pandas as pd
import numpy as np
import re

def _slug(s: str) -> str:
    s = str(s).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")

def _extract_block(df, start_row, base_cols, subject_name, rit_col, perc_col, band_col, ia_cols, ia_label_row):
    cols = list(base_cols) + [rit_col, perc_col] + ([band_col] if band_col else []) + list(ia_cols)
    block = df.loc[start_row:, cols].copy()

    rename = {
        base_cols[0]: "student_id",
        base_cols[1]: "last_name",
        base_cols[2]: "first_name",
        base_cols[3]: "section",
        base_cols[4]: "teacher",
        rit_col: "rit",
        perc_col: "percentile",
    }
    if band_col:
        rename[band_col] = "band"

    block = block.rename(columns=rename)

    # Rename IA columns based on the second header row (inside the sheet)
    ia_cols = list(ia_cols)
    ia_names = [str(ia_label_row[c]).strip() for c in ia_cols]
    ia_rename = {c: f"ia_{_slug(name)}" for c, name in zip(ia_cols, ia_names)}
    block = block.rename(columns=ia_rename)

    block["subject"] = subject_name

    # Clean types
    block["student_id"] = block["student_id"].astype(str).str.strip()
    block.loc[block["student_id"].isin(["nan", "None", ""]), "student_id"] = np.nan

    block["rit"] = pd.to_numeric(block["rit"], errors="coerce")
    block["percentile"] = pd.to_numeric(block["percentile"], errors="coerce")

    for c in ia_rename.values():
        block[c] = pd.to_numeric(block[c], errors="coerce")

    # Drop empty rows
    block = block.dropna(subset=["student_id", "rit"], how="all")
    return block

def load_map_excel(filepath: str, term_label: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name="MAP Data MasterSheet")

    # In your file: row 0 = labels like "RIT/Percentile", row 1 = IA names, data starts row 2
    ia_label_row = df.iloc[1]

    # Math block
    math = _extract_block(
        df=df,
        start_row=2,
        base_cols=["Student ID", "Last Name", "First Name", "Section", "Teacher"],
        subject_name="Math",
        rit_col="Mathematics",
        perc_col="Unnamed: 6",
        band_col="Unnamed: 7",          # Quantile
        ia_cols=df.columns[8:12],
        ia_label_row=ia_label_row,
    )

    # Reading block
    reading = _extract_block(
        df=df,
        start_row=2,
        base_cols=["Student ID.1", "Last Name.1", "First Name.1", "Section.1", "Teacher\n"],
        subject_name="Reading",
        rit_col="Reading",
        perc_col="Unnamed: 18",
        band_col="Unnamed: 19",         # Lexile
        ia_cols=df.columns[20:23],
        ia_label_row=ia_label_row,
    )

    # Language Usage block (no extra band column in your export; just percentile + IA)
    lang = _extract_block(
        df=df,
        start_row=2,
        base_cols=["Student ID.2", "Last Name.2", "First Name.2", "Section.2", "Teacher.1"],
        subject_name="Language Usage",
        rit_col="Language Usage",
        perc_col="Unnamed: 29",
        band_col=None,
        ia_cols=df.columns[30:33],
        ia_label_row=ia_label_row,
    )

    tidy = pd.concat([math, reading, lang], ignore_index=True)

    # Add helper fields
    tidy["student_name"] = (tidy["first_name"].astype(str).str.strip() + " " + tidy["last_name"].astype(str).str.strip()).str.strip()
    tidy["grade"] = pd.to_numeric(tidy["section"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    tidy["term"] = term_label
    return tidy
