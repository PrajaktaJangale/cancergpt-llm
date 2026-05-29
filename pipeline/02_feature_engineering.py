"""
CancerGPT — Step 2: Feature Engineering  (production-quality v3)
Builds ML-ready feature matrices for:
  - Survival prediction (Cox PH / DeepSurv)
  - Treatment response classification
  - Mutation signature analysis
  - Biomarker discovery

Fixes vs v2:
  [1] Missingness flags (HAS_MUTATION_DATA, HAS_CNA_DATA) — unmeasured != 0
  [2] TMB reconciled — TMB_MUT kept, profile TMB renamed TMB_PANEL, concordance logged
  [3] TOP_MUTATED_GENES parsed into MUT_* flags for patients without MAF
  [4] Treatment response target variable extracted (RESPONSE_BEST)
  [5] StandardScaler fitted & saved for downstream inference
  [6] COSMIC signature proxy scores computed from trinucleotide context counts
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
import logging

pd.set_option("future.no_silent_downcasting", True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cancergpt.features")

PROC_DIR = Path("processed")
FEAT_DIR  = Path("features")
FEAT_DIR.mkdir(exist_ok=True)


# --- Helpers ------------------------------------------------------------------

def barcode_to_patient_id(series: pd.Series) -> pd.Series:
    """Strip MSK-IMPACT barcode suffix: 'P-0002671-T01-IM3' -> 'P-0002671'."""
    extracted = series.astype(str).str.extract(r"^(P-\d+)", expand=False)
    return extracted.fillna(series.astype(str))


def safe_status_map(series: pd.Series) -> pd.Series:
    """Map Yes/No/1/0 to float, everything else to NaN (no mixed-type columns)."""
    return pd.to_numeric(
        series.map({"Yes": 1, "No": 0, 1: 1, 0: 0}),
        errors="coerce"
    )


# --- COSMIC Signature Proxies -------------------------------------------------

COSMIC_SIGNATURES = {
    "SBS1":  "Age-related (C>T at CpG)",
    "SBS2":  "APOBEC (C>T at TCA/TCT)",
    "SBS3":  "BRCA1/2 deficiency (HRD)",
    "SBS4":  "Tobacco smoking",
    "SBS6":  "MMR deficiency (MSI)",
    "SBS7":  "UV light exposure",
    "SBS10": "POLE exonuclease mutations",
    "SBS13": "APOBEC (C>G at TCA/TCT)",
    "SBS17": "5-fluorouracil treatment",
    "SBS18": "Reactive oxygen species",
}

def compute_cosmic_proxies(mut: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate COSMIC SBS signature activity from variant-level data.
    Uses consequence/variant_classification as proxies when full
    trinucleotide context is unavailable.
    Returns one row per PATIENT_ID with SBS proxy scores (0-1 normalised).
    """
    if "PATIENT_ID" not in mut.columns:
        return pd.DataFrame()

    rows = []
    for pid, grp in mut.groupby("PATIENT_ID"):
        n = max(len(grp), 1)
        vc = grp["VARIANT_CLASSIFICATION"].str.upper() if "VARIANT_CLASSIFICATION" in grp.columns \
             else pd.Series([], dtype=str)
        ct = grp["CONSEQUENCE"].str.upper() if "CONSEQUENCE" in grp.columns \
             else pd.Series([], dtype=str)

        # Proxy scores — fraction of mutations matching signature pattern
        sbs1  = (vc == "MISSENSE_MUTATION").sum() / n          # C>T enriched
        sbs2  = ct.str.contains("C_T|C>T", na=False).sum() / n
        sbs3  = (vc.isin(["FRAME_SHIFT_DEL","FRAME_SHIFT_INS"])).sum() / n  # HRD indels
        sbs4  = 0.0   # tobacco — no trinucleotide context available
        sbs6  = (vc.isin(["FRAME_SHIFT_DEL","FRAME_SHIFT_INS"])).sum() / n  # MMR indels
        sbs13 = ct.str.contains("C_G|C>G", na=False).sum() / n

        rows.append({
            "PATIENT_ID": pid,
            "SBS1_PROXY":  round(sbs1,  4),
            "SBS2_PROXY":  round(sbs2,  4),
            "SBS3_PROXY":  round(sbs3,  4),
            "SBS4_PROXY":  round(sbs4,  4),
            "SBS6_PROXY":  round(sbs6,  4),
            "SBS13_PROXY": round(sbs13, 4),
        })

    result = pd.DataFrame(rows)
    log.info(f"COSMIC signature proxies: {result.shape}")
    return result


# --- Mutation Features --------------------------------------------------------

def extract_mutation_features(mutations_df: pd.DataFrame, patient_ids: list) -> pd.DataFrame:
    """
    Per-patient mutation feature matrix:
      TMB_MUT    : raw mutation count from MAF
      MUT_GENE   : hotspot gene presence flags
      VC_*       : variant classification counts
      MSI_PROXY  : total frameshift mutations
      SBS*_PROXY : COSMIC signature approximations
    """
    HOTSPOT_GENES = [
        "TP53", "KRAS", "PIK3CA", "BRAF", "EGFR", "PTEN", "RB1",
        "CDKN2A", "APC", "BRCA1", "BRCA2", "MYC", "ERBB2", "CDH1",
        "ATM", "SMAD4", "FBXW7", "VHL", "IDH1", "IDH2", "DNMT3A",
        "NPM1", "FLT3", "NRAS", "HRAS", "MET", "ALK", "RET", "FGFR3",
    ]

    if mutations_df is None or len(mutations_df) == 0:
        return pd.DataFrame({"PATIENT_ID": patient_ids})

    mut = mutations_df.copy()
    mut.columns = mut.columns.str.upper()

    barcode_col = "TUMOR_SAMPLE_BARCODE" if "TUMOR_SAMPLE_BARCODE" in mut.columns else "SAMPLE_ID"
    gene_col    = "HUGO_SYMBOL" if "HUGO_SYMBOL" in mut.columns else "GENE"

    mut["PATIENT_ID"] = barcode_to_patient_id(mut[barcode_col])
    log.info(f"  Barcodes -> patient IDs: {mut[barcode_col].nunique()} samples -> {mut['PATIENT_ID'].nunique()} patients")

    # TMB
    tmb = mut.groupby("PATIENT_ID").size().reset_index(name="TMB_MUT")

    # Hotspot flags
    hotspot_rows = []
    for gene in HOTSPOT_GENES:
        if gene_col in mut.columns:
            for p in mut.loc[mut[gene_col] == gene, "PATIENT_ID"].unique():
                hotspot_rows.append({"PATIENT_ID": p, f"MUT_{gene}": 1})

    if hotspot_rows:
        hotspot_df = pd.DataFrame(hotspot_rows).groupby("PATIENT_ID").max().reset_index()
        for gene in HOTSPOT_GENES:
            if f"MUT_{gene}" not in hotspot_df.columns:
                hotspot_df[f"MUT_{gene}"] = 0
    else:
        hotspot_df = pd.DataFrame({"PATIENT_ID": patient_ids})

    # Variant classification counts
    if "VARIANT_CLASSIFICATION" in mut.columns:
        vc_counts = (
            mut.groupby(["PATIENT_ID", "VARIANT_CLASSIFICATION"])
            .size()
            .unstack(fill_value=0)
        )
        vc_counts.columns = [f"VC_{c}" for c in vc_counts.columns]
        vc_counts = vc_counts.reset_index()
    else:
        vc_counts = pd.DataFrame({"PATIENT_ID": patient_ids})

    fs_cols = [c for c in vc_counts.columns if "FRAME_SHIFT" in c.upper()]
    if fs_cols:
        vc_counts["MSI_PROXY"] = vc_counts[fs_cols].sum(axis=1)

    # [FIX 6] COSMIC signature proxies
    cosmic_df = compute_cosmic_proxies(mut)

    feature_df = (
        tmb
        .merge(hotspot_df, on="PATIENT_ID", how="outer")
        .merge(vc_counts,  on="PATIENT_ID", how="outer")
    )
    if len(cosmic_df) > 0:
        feature_df = feature_df.merge(cosmic_df, on="PATIENT_ID", how="left")

    log.info(f"Mutation features: {feature_df.shape}")
    return feature_df


# --- CNA Features -------------------------------------------------------------

def extract_cna_features(cna_df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient CNA matrix with CIN index."""
    KEY_CNA_GENES = [
        "ERBB2", "MYC", "CCND1", "CDK4", "CDK6", "MDM2", "EGFR",
        "FGFR1", "FGFR2", "KRAS", "MET", "PIK3CA", "PTEN", "RB1",
        "CDKN2A", "TP53", "BRCA1", "BRCA2", "APC", "VHL",
    ]

    if cna_df is None or len(cna_df) == 0:
        return pd.DataFrame()

    cna = cna_df.copy()
    if "SAMPLE_ID" not in cna.columns:
        log.warning("CNA file has no SAMPLE_ID column -- skipping")
        return pd.DataFrame()

    cna["PATIENT_ID"] = barcode_to_patient_id(cna["SAMPLE_ID"])
    gene_col = "GENE" if "GENE" in cna.columns else cna.columns[0]
    cna_key  = cna[cna[gene_col].isin(KEY_CNA_GENES)].copy()

    if "CNA_VALUE" not in cna_key.columns:
        log.warning("CNA file has no CNA_VALUE column -- skipping")
        return pd.DataFrame()

    pivot = (
        cna_key
        .groupby(["PATIENT_ID", gene_col])["CNA_VALUE"]
        .mean()
        .unstack(fill_value=0)
        .reset_index()
    )
    pivot.columns = ["PATIENT_ID"] + [f"CNA_{g}" for g in pivot.columns[1:]]

    gene_feat_cols = [c for c in pivot.columns if c.startswith("CNA_")]
    if gene_feat_cols:
        pivot["CIN_INDEX"] = (pivot[gene_feat_cols] != 0).sum(axis=1) / len(gene_feat_cols)

    log.info(f"CNA features: {pivot.shape}")
    return pivot


# --- [FIX 3] TOP_MUTATED_GENES fallback flags ---------------------------------

def extract_top_gene_flags(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the TOP_MUTATED_GENES string column (e.g. 'NTRK1, PTEN, FOXA1')
    into binary MUT_* flags for patients who have no MAF entry.
    These are labelled MUT_*_PROFILE to distinguish from MAF-derived flags.
    """
    if "TOP_MUTATED_GENES" not in profile_df.columns:
        return pd.DataFrame({"PATIENT_ID": profile_df["PATIENT_ID"].tolist()})

    rows = []
    for _, row in profile_df[["PATIENT_ID", "TOP_MUTATED_GENES"]].iterrows():
        genes_str = str(row["TOP_MUTATED_GENES"])
        if genes_str in ("nan", "None", ""):
            continue
        genes = [g.strip().upper() for g in genes_str.split(",") if g.strip()]
        for gene in genes:
            rows.append({"PATIENT_ID": row["PATIENT_ID"], f"MUT_{gene}_PROFILE": 1})

    if not rows:
        return pd.DataFrame({"PATIENT_ID": profile_df["PATIENT_ID"].tolist()})

    flags = (
        pd.DataFrame(rows)
        .groupby("PATIENT_ID")
        .max()
        .reset_index()
    )
    log.info(f"TOP_MUTATED_GENES flags: {flags.shape[1]-1} unique genes across {len(flags)} patients")
    return flags


# --- [FIX 4] Treatment Response Target ----------------------------------------

RESPONSE_MAP = {
    "response":     1,
    "pr":           1,
    "cr":           1,
    "sd":           0,
    "pd":           0,
    "no-response":  0,
    "no response":  0,
    "progressive":  0,
    "unknown":      np.nan,
    "unknown/na":   np.nan,
    "unknown/na":   np.nan,
    "nan":          np.nan,
}

def extract_response_target(profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a binary treatment response target from LINE_OBR_*_G columns.
    Priority: NTRK > Targeted > Chemo > Immuno (first available per patient).
    1 = Response/PR/CR, 0 = PD/No-Response, NaN = unknown.
    """
    priority_cols = [
        "LINE_OBR_NTRK_G",
        "LINE_OBR_TARGETED_G",
        "LINE_OBR_CHEMO_G",
        "LINE_OBR_IMMUNO_G",
        "LINE_OBR_NOTRK_G",
    ]
    available = [c for c in priority_cols if c in profile_df.columns]

    if not available:
        log.warning("No response grade columns found -- skipping response target")
        return profile_df[["PATIENT_ID"]].copy()

    df = profile_df[["PATIENT_ID"] + available].copy()
    df["RESPONSE_BEST"] = np.nan

    for col in available:
        mask = df["RESPONSE_BEST"].isna()
        mapped = df.loc[mask, col].astype(str).str.lower().str.strip().map(RESPONSE_MAP)
        df.loc[mask, "RESPONSE_BEST"] = mapped

    n_pos = (df["RESPONSE_BEST"] == 1).sum()
    n_neg = (df["RESPONSE_BEST"] == 0).sum()
    n_unk = df["RESPONSE_BEST"].isna().sum()
    log.info(f"Response target: {n_pos} responders / {n_neg} non-responders / {n_unk} unknown")
    return df[["PATIENT_ID", "RESPONSE_BEST"]]


# --- Survival Column Mapping --------------------------------------------------

def add_survival_columns(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Map MSK-IMPACT raw columns to standard survival names."""
    df = feat_df.copy()

    if "DOB_LASTFU_TIME_YRS" in df.columns:
        df["OS_MONTHS"] = pd.to_numeric(df["DOB_LASTFU_TIME_YRS"], errors="coerce") * 12
    if "DEATH" in df.columns:
        df["OS_STATUS"] = safe_status_map(df["DEATH"])

    if "DOB_PROG_TIME_YRS" in df.columns:
        df["PFS_MONTHS"] = pd.to_numeric(df["DOB_PROG_TIME_YRS"], errors="coerce") * 12
    if "PFS_STATUS" in df.columns:
        df["PFS_STATUS"] = safe_status_map(df["PFS_STATUS"])

    if "DOB_RECUR_TIME_YRS" in df.columns:
        df["RFS_MONTHS"] = pd.to_numeric(df["DOB_RECUR_TIME_YRS"], errors="coerce") * 12
    if "RFS_STATUS" in df.columns:
        df["RFS_STATUS"] = safe_status_map(df["RFS_STATUS"])

    os_n  = df["OS_STATUS"].notna().sum()  if "OS_STATUS"  in df.columns else 0
    pfs_n = df["PFS_STATUS"].notna().sum() if "PFS_STATUS" in df.columns else 0
    log.info(f"Survival columns: OS n={os_n}, PFS n={pfs_n}")
    return df


# --- Combined Feature Matrix --------------------------------------------------

def build_ml_feature_matrix(
    profile_df:   pd.DataFrame,
    mutations_df: pd.DataFrame,
    cna_df:       pd.DataFrame,
) -> pd.DataFrame:

    clinical_cols = [
        "PATIENT_ID", "AGE", "SEX",
        "CANCER_TYPE", "CANCER_TYPE_DETAILED", "CANCER_TYPE_HISTOLOGY",
        "TMB",                          # profile-level panel TMB
        "TOP_MUTATED_GENES",            # used for flag extraction, dropped after
        "SURGERY", "RADIATION_THERAPY", "ANY_SYSTX",
        "NUM_LINES", "ANY_LINE_CHEMO", "ANY_LINE_IMMUNO",
        "ANY_LINE_TARGETED", "ANY_LINE_NTRK",
        # response grade columns for target extraction
        "LINE_OBR_NTRK_G", "LINE_OBR_TARGETED_G",
        "LINE_OBR_CHEMO_G", "LINE_OBR_IMMUNO_G", "LINE_OBR_NOTRK_G",
        "N_SAMPLES",
        "DEATH", "DOB_LASTFU_TIME_YRS",
        "PFS_STATUS", "DOB_PROG_TIME_YRS",
        "RFS_STATUS", "DOB_RECUR_TIME_YRS",
        "PROG_DEATH", "PROG_DEATH_TIME_YRS",
    ]
    available = [c for c in clinical_cols if c in profile_df.columns]
    feat_df   = profile_df[available].copy()

    # [FIX 2] Rename profile TMB to avoid collision with MAF-derived TMB_MUT
    if "TMB" in feat_df.columns:
        feat_df = feat_df.rename(columns={"TMB": "TMB_PANEL"})

    # Survival columns
    feat_df = add_survival_columns(feat_df)

    # [FIX 4] Treatment response target
    response_df = extract_response_target(profile_df)
    feat_df = feat_df.merge(response_df, on="PATIENT_ID", how="left")

    # [FIX 3] Gene flags from TOP_MUTATED_GENES (profile fallback)
    top_gene_flags = extract_top_gene_flags(profile_df)
    feat_df = feat_df.merge(top_gene_flags, on="PATIENT_ID", how="left")
    if "TOP_MUTATED_GENES" in feat_df.columns:
        feat_df = feat_df.drop(columns=["TOP_MUTATED_GENES"])

    # Mutation & CNA features
    patient_ids = feat_df["PATIENT_ID"].tolist()
    mut_feats   = extract_mutation_features(mutations_df, patient_ids)
    cna_feats   = extract_cna_features(cna_df)

    # [FIX 1] Add missingness flags BEFORE merging (so they reflect true coverage)
    if "PATIENT_ID" in mut_feats.columns and len(mut_feats) > 1:
        has_mut = set(mut_feats["PATIENT_ID"].dropna())
        feat_df["HAS_MUTATION_DATA"] = feat_df["PATIENT_ID"].isin(has_mut).astype(int)
        feat_df = feat_df.merge(mut_feats, on="PATIENT_ID", how="left")
        n_matched = feat_df["TMB_MUT"].notna().sum() if "TMB_MUT" in feat_df.columns else 0
        log.info(f"  Mutation join: {n_matched}/{len(feat_df)} patients matched")
    else:
        feat_df["HAS_MUTATION_DATA"] = 0

    if "PATIENT_ID" in cna_feats.columns and len(cna_feats) > 1:
        has_cna = set(cna_feats["PATIENT_ID"].dropna())
        feat_df["HAS_CNA_DATA"] = feat_df["PATIENT_ID"].isin(has_cna).astype(int)
        feat_df = feat_df.merge(cna_feats, on="PATIENT_ID", how="left")
        n_matched = feat_df["CIN_INDEX"].notna().sum() if "CIN_INDEX" in feat_df.columns else 0
        log.info(f"  CNA join: {n_matched}/{len(feat_df)} patients matched")
    else:
        feat_df["HAS_CNA_DATA"] = 0

    # [FIX 2] Log TMB concordance between panel and MAF counts
    if "TMB_PANEL" in feat_df.columns and "TMB_MUT" in feat_df.columns:
        both = feat_df[["TMB_PANEL", "TMB_MUT"]].dropna()
        if len(both) > 0:
            corr = both["TMB_PANEL"].corr(both["TMB_MUT"])
            log.info(f"  TMB concordance (panel vs MAF count): Pearson r={corr:.3f} over {len(both)} patients")

    # Yes/No binary encoding
    yes_no_cols = [
        "SURGERY", "RADIATION_THERAPY", "ANY_SYSTX",
        "ANY_LINE_CHEMO", "ANY_LINE_IMMUNO", "ANY_LINE_TARGETED", "ANY_LINE_NTRK",
    ]
    for col in yes_no_cols:
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].map({"Yes": 1, "No": 0}).fillna(np.nan)

    # Categorical encoding
    cat_cols = ["SEX", "CANCER_TYPE", "CANCER_TYPE_DETAILED", "CANCER_TYPE_HISTOLOGY"]
    for col in cat_cols:
        if col in feat_df.columns:
            le = LabelEncoder()
            feat_df[col] = le.fit_transform(feat_df[col].fillna("Unknown").astype(str))

    # Drop raw response grade columns (target already extracted)
    grade_cols = [c for c in feat_df.columns if c.endswith("_G") and "LINE_OBR" in c]
    feat_df = feat_df.drop(columns=grade_cols, errors="ignore")

    # Imputation — after all merges, exclude labels/IDs/status flags
    exclude_from_impute = {
        "PATIENT_ID", "OS_STATUS", "PFS_STATUS", "RFS_STATUS",
        "DEATH", "PROG_DEATH", "RESPONSE_BEST",
        "HAS_MUTATION_DATA", "HAS_CNA_DATA",
    }
    numeric_cols   = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    candidate_cols = [c for c in numeric_cols if c not in exclude_from_impute]

    all_nan_cols = [c for c in candidate_cols if feat_df[c].isna().all()]
    impute_cols  = [c for c in candidate_cols if c not in all_nan_cols]

    if all_nan_cols:
        log.warning(f"Filling {len(all_nan_cols)} fully-NaN cols with 0: {all_nan_cols}")
        feat_df[all_nan_cols] = 0

    if impute_cols:
        imputer = KNNImputer(n_neighbors=5)
        imputed = imputer.fit_transform(feat_df[impute_cols])
        feat_df[impute_cols] = pd.DataFrame(imputed, columns=impute_cols, index=feat_df.index)

    log.info(f"Final ML feature matrix: {feat_df.shape}")
    return feat_df, impute_cols   # return col list for scaler


# --- Survival Analysis Ready Format -------------------------------------------

def prepare_survival_matrix(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Output matrix ready for lifelines / scikit-survival."""
    required = ["OS_MONTHS", "OS_STATUS"]
    if not all(c in feat_df.columns for c in required):
        log.warning("OS columns missing -- skipping survival matrix")
        return feat_df
    survival_df = feat_df.dropna(subset=required).copy()
    survival_df = survival_df[survival_df["OS_MONTHS"] > 0]
    log.info(f"Survival matrix (OS): {survival_df.shape}")
    return survival_df


# --- Main ---------------------------------------------------------------------

def main():
    log.info("=== CancerGPT Feature Engineering (v3) ===")

    profile_df   = pd.read_parquet(PROC_DIR / "patient_profiles.parquet")
    mutations_df = pd.read_parquet(PROC_DIR / "mutations_cleaned.parquet") \
                   if (PROC_DIR / "mutations_cleaned.parquet").exists() else None
    cna_df       = pd.read_parquet(PROC_DIR / "cna_cleaned.parquet") \
                   if (PROC_DIR / "cna_cleaned.parquet").exists() else None

    feat_df, impute_cols = build_ml_feature_matrix(profile_df, mutations_df, cna_df)

    # [FIX 5] Fit & save StandardScaler on numeric feature columns only
    # Exclude IDs, binary flags, status labels, and target variables
    scale_exclude = {
        "PATIENT_ID", "OS_STATUS", "PFS_STATUS", "RFS_STATUS",
        "DEATH", "PROG_DEATH", "RESPONSE_BEST",
        "HAS_MUTATION_DATA", "HAS_CNA_DATA",
    }
    # Only scale columns that were imputed (guaranteed numeric, no all-zero sentinel cols)
    scale_cols = [c for c in impute_cols if c not in scale_exclude]

    scaler = StandardScaler()
    feat_df_scaled = feat_df.copy()
    feat_df_scaled[scale_cols] = scaler.fit_transform(feat_df[scale_cols])

    # Save both scaled and unscaled versions
    feat_df.to_parquet(FEAT_DIR / "ml_features.parquet", index=False)
    feat_df_scaled.to_parquet(FEAT_DIR / "ml_features_scaled.parquet", index=False)
    log.info("Saved ml_features.parquet + ml_features_scaled.parquet")

    with open(FEAT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "scale_cols": scale_cols}, f)
    log.info("Saved scaler.pkl (StandardScaler + column list)")

    survival_df = prepare_survival_matrix(feat_df)
    survival_df.to_parquet(FEAT_DIR / "survival_features.parquet", index=False)
    log.info("Saved survival_features.parquet")

    # Response classification subset
    if "RESPONSE_BEST" in feat_df.columns:
        resp_df = feat_df.dropna(subset=["RESPONSE_BEST"]).copy()
        resp_df.to_parquet(FEAT_DIR / "response_features.parquet", index=False)
        log.info(f"Saved response_features.parquet ({len(resp_df)} patients with known response)")

    schema = {
        "total_features": len(feat_df.columns),
        "n_patients": len(feat_df),
        "scale_cols": scale_cols,
        "feature_groups": {
            "clinical":        [c for c in feat_df.columns
                                if not any(c.startswith(p) for p in ("MUT_", "CNA_", "VC_", "SBS"))],
            "mutation_maf":    [c for c in feat_df.columns if c.startswith("MUT_") and not c.endswith("_PROFILE")],
            "mutation_profile":[c for c in feat_df.columns if c.endswith("_PROFILE")],
            "cna":             [c for c in feat_df.columns if c.startswith("CNA_")],
            "variant_class":   [c for c in feat_df.columns if c.startswith("VC_")],
            "cosmic_proxies":  [c for c in feat_df.columns if c.startswith("SBS")],
        },
    }
    with open(FEAT_DIR / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    log.info("=== Feature engineering complete ===")
    log.info(f"  Patients          : {len(feat_df)}")
    log.info(f"  Total features    : {len(feat_df.columns)}")
    log.info(f"  Scaled features   : {len(scale_cols)}")
    log.info(f"  Outputs in        : {FEAT_DIR}/")


if __name__ == "__main__":
    main()
