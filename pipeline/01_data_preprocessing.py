"""
CancerGPT — Step 1: Data Preprocessing
Handles cBioPortal multi-modal cancer data ingestion and normalization.
Compatible with: data_clinical_patient, data_clinical_sample,
                 data_mutations, data_cna, data_sv, data_timeline

FIXED VERSION — Handles:
  - Windows paths
  - Duplicate column names in mutations file
  - Uppercase column names after cleaning
  - List columns in natural language serializer
  - Files in same folder as script (DATA_DIR = ".")
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cancergpt.preprocess")

# ── Point to current folder where your .txt files are ──────────────────────
DATA_DIR = Path(".")
OUT_DIR  = Path("processed")
OUT_DIR.mkdir(exist_ok=True)


# ─── Loaders ────────────────────────────────────────────────────────────────

def load_cbio_txt(filepath: Path) -> pd.DataFrame:
    """Load a cBioPortal tab-delimited .txt file, skipping comment lines."""
    comment_lines = 0
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                comment_lines += 1
            else:
                break
    df = pd.read_csv(filepath, sep="\t", skiprows=comment_lines,
                 low_memory=False, encoding="utf-8", encoding_errors="replace")
    log.info(f"Loaded {filepath.name}: {df.shape}")
    return df


def load_all_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    files = {
        "clinical_patient": "data_clinical_patient.txt",
        "clinical_sample":  "data_clinical_sample.txt",
        "mutations":        "data_mutations.txt",
        "cna":              "data_cna.txt",
        "sv":               "data_sv.txt",
        "timeline":         "data_timeline.txt",
    }
    datasets = {}
    for key, fname in files.items():
        fpath = data_dir / fname
        if fpath.exists():
            datasets[key] = load_cbio_txt(fpath)
        else:
            log.warning(f"Missing file: {fname}")
    return datasets


# ─── Helpers ────────────────────────────────────────────────────────────────

def dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate columns, keeping the first occurrence."""
    return df.loc[:, ~df.columns.duplicated()]


def safe_notna(val) -> bool:
    """Safe pd.notna that works for scalars, lists, and arrays."""
    if val is None:
        return False
    if isinstance(val, (list, np.ndarray)):
        return len(val) > 0
    try:
        return bool(pd.notna(val))
    except (ValueError, TypeError):
        return False


def safe_flatten_genes(series) -> list:
    """Flatten a series of gene lists into a unique flat list."""
    result = []
    for item in series:
        if isinstance(item, list):
            result.extend(item)
    return list(set(result))


# ─── Cleaning ───────────────────────────────────────────────────────────────

def clean_clinical_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize clinical patient data."""
    df = df.copy()
    df.columns = [str(c).upper().strip() for c in df.columns]
    df = dedup_columns(df)

    # Drop cBioPortal display-name meta rows if present
    if len(df) > 0 and df.iloc[0].astype(str).str.startswith("#").any():
        df = df.iloc[4:].reset_index(drop=True)

    # Numeric coercion
    for col in ["AGE", "OVERALL_SURVIVAL_MONTHS", "DISEASE_FREE_MONTHS",
                "OS_MONTHS", "DFS_MONTHS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binary survival status
    for col in ["OS_STATUS", "DFS_STATUS"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: 1 if str(x).upper() in ["1:DECEASED", "1:RECURRED", "1"] else 0
            )

    if "PATIENT_ID" in df.columns:
        df.dropna(subset=["PATIENT_ID"], inplace=True)

    log.info(f"Clinical patient cleaned: {df.shape}")
    return df


def clean_clinical_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize clinical sample data."""
    df = df.copy()
    df.columns = [str(c).upper().strip() for c in df.columns]
    df = dedup_columns(df)
    log.info(f"Clinical sample cleaned: {df.shape}")
    return df


def clean_mutations(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and annotate somatic mutations."""
    df = df.copy()
    df.columns = [str(c).upper().strip() for c in df.columns]

    # Remove duplicate columns FIRST — this was causing the parquet error
    df = dedup_columns(df)

    # Keep only somatic/germline calls
    if "MUTATION_STATUS" in df.columns:
        df = df[df["MUTATION_STATUS"].str.upper().isin(["SOMATIC", "GERMLINE"])].copy()

    # Pathogenicity flag
    if "ONCOGENIC" in df.columns:
        keywords = ["oncogenic", "likely oncogenic", "gain-of-function"]
        df["IS_ONCOGENIC"] = df["ONCOGENIC"].apply(
            lambda x: 1 if any(k in str(x).lower() for k in keywords) else 0
        )

    # Loss-of-function flag
    if "VARIANT_CLASSIFICATION" in df.columns:
        lof = ["Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
               "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation"]
        df["IS_LOF"] = df["VARIANT_CLASSIFICATION"].isin(lof).astype(int)

    log.info(f"Mutations cleaned: {df.shape}")
    return df


def clean_cna(df: pd.DataFrame) -> pd.DataFrame:
    """
    CNA matrix: rows=genes, cols=samples.
    Melts to long format for downstream use.
    """
    df = df.copy()
    df = dedup_columns(df)

    gene_col = df.columns[0]
    df = df.set_index(gene_col)

    df_long = df.reset_index().melt(
        id_vars=df.index.name if df.index.name else gene_col,
        var_name="SAMPLE_ID",
        value_name="CNA_VALUE"
    )
    df_long.columns = ["GENE", "SAMPLE_ID", "CNA_VALUE"]
    df_long["CNA_VALUE"] = pd.to_numeric(df_long["CNA_VALUE"], errors="coerce")
    df_long["CNA_LABEL"] = df_long["CNA_VALUE"].map({
        2: "Amplification", 1: "Gain",
        0: "Diploid", -1: "Shallow Deletion", -2: "Deep Deletion"
    })

    log.info(f"CNA (long) cleaned: {df_long.shape}")
    return df_long


def clean_sv(df: pd.DataFrame) -> pd.DataFrame:
    """Structural variant data cleaning."""
    df = df.copy()
    df.columns = [str(c).upper().strip() for c in df.columns]
    df = dedup_columns(df)

    if "SV_STATUS" in df.columns:
        df = df[df["SV_STATUS"].str.upper() == "SOMATIC"].copy()

    log.info(f"SV cleaned: {df.shape}")
    return df


def clean_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Treatment timeline normalization."""
    df = df.copy()
    df.columns = [str(c).upper().strip() for c in df.columns]
    df = dedup_columns(df)

    if "START_DATE" in df.columns and "STOP_DATE" in df.columns:
        df["DURATION_DAYS"] = (
            pd.to_numeric(df["STOP_DATE"], errors="coerce") -
            pd.to_numeric(df["START_DATE"], errors="coerce")
        )

    log.info(f"Timeline cleaned: {df.shape}")
    return df


# ─── Patient-Level Aggregation ───────────────────────────────────────────────

def build_patient_profile(
    clinical_patient: pd.DataFrame,
    clinical_sample:  Optional[pd.DataFrame],
    mutations:        Optional[pd.DataFrame],
    cna:              Optional[pd.DataFrame],
    sv:               Optional[pd.DataFrame],
    timeline:         Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge all modalities into a single patient-level summary DataFrame.
    """
    if clinical_patient is None or len(clinical_patient) == 0:
        log.warning("No clinical patient data available")
        return pd.DataFrame()

    profile = clinical_patient.copy()
    pid_col = "PATIENT_ID"

    if pid_col not in profile.columns:
        log.warning(f"PATIENT_ID not found. Available columns: {list(profile.columns[:10])}")
        return profile

    # ── Sample aggregation ──────────────────────────────────────────────────
    if clinical_sample is not None and len(clinical_sample) > 0:
        cs = clinical_sample.copy()
        cs.columns = [str(c).upper().strip() for c in cs.columns]
        cs = dedup_columns(cs)

        if pid_col in cs.columns:
            agg_dict = {}
            if "SAMPLE_ID" in cs.columns:
                agg_dict["N_SAMPLES"] = ("SAMPLE_ID", "count")
            if "CANCER_TYPE" in cs.columns:
                agg_dict["CANCER_TYPE"] = (
                    "CANCER_TYPE",
                    lambda x: x.mode()[0] if len(x) > 0 else np.nan
                )
            if "CANCER_TYPE_DETAILED" in cs.columns:
                agg_dict["CANCER_TYPE_DETAILED"] = (
                    "CANCER_TYPE_DETAILED",
                    lambda x: x.mode()[0] if len(x) > 0 else np.nan
                )
            if agg_dict:
                sample_agg = cs.groupby(pid_col).agg(**agg_dict).reset_index()
                profile = profile.merge(sample_agg, on=pid_col, how="left")

    # ── Mutation aggregation ────────────────────────────────────────────────
    if mutations is not None and len(mutations) > 0:
        mut = mutations.copy()

        # Find gene column
        hugo_col = None
        for candidate in ["HUGO_SYMBOL", "GENE", "SYMBOL"]:
            if candidate in mut.columns:
                hugo_col = candidate
                break

        # Find sample column
        sample_col = None
        for candidate in ["TUMOR_SAMPLE_BARCODE", "SAMPLE_ID"]:
            if candidate in mut.columns:
                sample_col = candidate
                break

        if hugo_col and sample_col:
            agg_dict = {
                "TMB": (hugo_col, "count"),
                "TOP_GENES": (hugo_col, lambda x: list(x.value_counts().head(5).index))
            }
            if "IS_ONCOGENIC" in mut.columns:
                agg_dict["N_ONCOGENIC"] = ("IS_ONCOGENIC", "sum")

            mut_agg = mut.groupby(sample_col).agg(**agg_dict).reset_index()
            mut_agg = mut_agg.rename(columns={sample_col: "SAMPLE_ID"})

            # Map samples → patients
            if clinical_sample is not None and len(clinical_sample) > 0:
                cs = clinical_sample.copy()
                cs.columns = [str(c).upper().strip() for c in cs.columns]
                cs = dedup_columns(cs)

                if "SAMPLE_ID" in cs.columns and pid_col in cs.columns:
                    s2p = cs[["SAMPLE_ID", pid_col]].drop_duplicates()
                    mut_agg = mut_agg.merge(s2p, on="SAMPLE_ID", how="left")

                    patient_mut = mut_agg.groupby(pid_col).agg(
                        TMB=("TMB", "sum"),
                        TOP_MUTATED_GENES=("TOP_GENES", safe_flatten_genes),
                    ).reset_index()

                    if "N_ONCOGENIC" in mut_agg.columns:
                        onco = mut_agg.groupby(pid_col)["N_ONCOGENIC"].sum().reset_index()
                        patient_mut = patient_mut.merge(onco, on=pid_col, how="left")

                    profile = profile.merge(patient_mut, on=pid_col, how="left")
        else:
            log.warning(f"Gene/sample columns not found in mutations. Found: {list(mut.columns[:10])}")

    log.info(f"Patient profile built: {profile.shape}")
    return profile


# ─── Natural Language Serializer ─────────────────────────────────────────────

def profile_to_natural_language(row: pd.Series) -> str:
    """Convert a patient profile row to a natural-language clinical summary for RAG."""
    parts = []

    parts.append(f"Patient ID: {row.get('PATIENT_ID', 'Unknown')}.")

    age = row.get("AGE")
    if safe_notna(age):
        try:
            parts.append(f"Age: {int(float(age))} years.")
        except (ValueError, TypeError):
            pass

    sex = row.get("SEX")
    if safe_notna(sex):
        parts.append(f"Sex: {sex}.")

    cancer = row.get("CANCER_TYPE")
    if not safe_notna(cancer):
        cancer = row.get("CANCER_TYPE_DETAILED")
    if safe_notna(cancer):
        parts.append(f"Cancer type: {cancer}.")

    cancer_detail = row.get("CANCER_TYPE_DETAILED")
    if safe_notna(cancer_detail) and str(cancer_detail) != str(cancer):
        parts.append(f"Detailed subtype: {cancer_detail}.")

    tmb = row.get("TMB")
    if safe_notna(tmb):
        try:
            tmb_int = int(float(tmb))
            label = "High" if tmb_int > 10 else "Low"
            parts.append(f"Tumor mutational burden (TMB): {tmb_int} mutations ({label}).")
        except (ValueError, TypeError):
            pass

    # Safe list handling for genes
    genes = row.get("TOP_MUTATED_GENES")
    if isinstance(genes, list) and len(genes) > 0:
        gene_strs = [str(g) for g in genes[:8] if g is not None]
        if gene_strs:
            parts.append(f"Key mutated genes: {', '.join(gene_strs)}.")
    elif isinstance(genes, str) and genes.strip() and genes.strip() not in ("nan", "None"):
        parts.append(f"Key mutated genes: {genes}.")

    os_months = row.get("OS_MONTHS")
    if safe_notna(os_months):
        try:
            parts.append(f"Overall survival: {float(os_months):.1f} months.")
            status = "Deceased" if str(row.get("OS_STATUS")) == "1" else "Alive"
            parts.append(f"Survival status: {status}.")
        except (ValueError, TypeError):
            pass

    stage = row.get("STAGE")
    if safe_notna(stage):
        parts.append(f"Stage: {stage}.")

    subtype = row.get("SUBTYPE")
    if safe_notna(subtype):
        parts.append(f"Molecular subtype: {subtype}.")

    return " ".join(parts)


# ─── Save helper ─────────────────────────────────────────────────────────────

def save_dataframe(df: pd.DataFrame, path: Path):
    """Save a dataframe as parquet, converting list columns to strings first."""
    df = df.copy()
    df = dedup_columns(df)

    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(
                lambda x: ", ".join(str(i) for i in x) if isinstance(x, list) else x
            )

    try:
        df.to_parquet(path, index=False)
        log.info(f"Saved {path.name} {df.shape}")
    except Exception as e:
        csv_path = path.with_suffix(".csv")
        log.warning(f"Parquet failed ({e}), saving as CSV: {csv_path.name}")
        df.to_csv(csv_path, index=False)
        log.info(f"Saved {csv_path.name} {df.shape}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("=== CancerGPT Data Preprocessing ===")

    raw = load_all_data(DATA_DIR)

    if not raw:
        log.error("No data files found! Make sure .txt files are in the same folder as this script.")
        return

    # Clean each dataset
    cleaned = {}

    if "clinical_patient" in raw:
        cleaned["clinical_patient"] = clean_clinical_patient(raw["clinical_patient"])

    if "clinical_sample" in raw:
        cleaned["clinical_sample"] = clean_clinical_sample(raw["clinical_sample"])

    if "mutations" in raw:
        cleaned["mutations"] = clean_mutations(raw["mutations"])

    if "cna" in raw:
        cleaned["cna"] = clean_cna(raw["cna"])

    if "sv" in raw:
        cleaned["sv"] = clean_sv(raw["sv"])

    if "timeline" in raw:
        cleaned["timeline"] = clean_timeline(raw["timeline"])

    # Build unified patient profiles
    profile = build_patient_profile(
        clinical_patient=cleaned.get("clinical_patient", pd.DataFrame()),
        clinical_sample=cleaned.get("clinical_sample"),
        mutations=cleaned.get("mutations"),
        cna=cleaned.get("cna"),
        sv=cleaned.get("sv"),
        timeline=cleaned.get("timeline"),
    )

    if len(profile) == 0:
        log.error("Patient profile is empty — check your input files.")
        return

    # Save patient profiles
    save_dataframe(profile, OUT_DIR / "patient_profiles.parquet")

    # Generate natural language summaries for RAG
    profile["nl_summary"] = profile.apply(profile_to_natural_language, axis=1)
    nl_records = profile[["PATIENT_ID", "nl_summary"]].to_dict(orient="records")

    with open(OUT_DIR / "patient_nl_summaries.json", "w", encoding="utf-8") as f:
        json.dump(nl_records, f, indent=2, ensure_ascii=False)
    log.info(f"Saved patient_nl_summaries.json ({len(nl_records)} records)")

    # Save each cleaned file
    for name, df in cleaned.items():
        save_dataframe(df, OUT_DIR / f"{name}_cleaned.parquet")

    log.info("=" * 50)
    log.info("Preprocessing complete!")
    log.info(f"Patients processed : {len(profile)}")
    log.info(f"Output folder      : {OUT_DIR.resolve()}")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
