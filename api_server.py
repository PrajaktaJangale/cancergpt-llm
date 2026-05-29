"""
CancerGPT — FastAPI Server (Fixed Version)
Changes:
  - /patients returns decoded cancer type names (not numeric codes)
  - /consult/load returns structured patient profile for web UI
  - Paths work both locally and in Docker
  - GROQ_API_KEY always read from environment
  - CORS enabled for all origins
"""

import json, logging
from pathlib import Path
from typing import Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cancergpt.api")

BASE_DIR    = Path(__file__).parent
INDEX_DIR   = BASE_DIR / "rag_index"
FEAT_DIR    = BASE_DIR / "features"
PROC_DIR    = BASE_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(
    title="CancerGPT API",
    description="AI Clinical Decision Support for Oncology",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

web_dir = BASE_DIR / "web"
if web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

# ── Cancer type decode map — matches your actual cBioPortal data ─────────────
CANCER_TYPE_MAP = {
    0:  "Breast Cancer",
    1:  "Colorectal Cancer",
    2:  "Salivary Gland Cancer",
    3:  "Hepatobiliary Cancer",
    4:  "Pancreatic Cancer",
    6:  "Non-Small Cell Lung Cancer",
    9:  "Thyroid Cancer",
    10: "Melanoma",
    11: "Soft Tissue Sarcoma",
    15: "Cancer of Unknown Primary",
    17: "Glioma",
    42: "Miscellaneous Brain Tumor",
    55: "Uterine Sarcoma",
    62: "Breast Sarcoma",
    75: "Appendiceal Cancer",
}

def decode_cancer_type(val):
    """Convert numeric cancer type code to human-readable name."""
    if val is None or val == "":
        return "Unknown"
    try:
        key = int(round(float(val)))
        return CANCER_TYPE_MAP.get(key, f"Cancer Type {key}")
    except (ValueError, TypeError):
        return str(val)  # already a string name — return as-is

# ── Schemas ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    patient_id: Optional[str] = None

class PatientLoadRequest(BaseModel):
    patient_id: str

class QueryRequest(BaseModel):
    query: str

# ── Agent lazy load ───────────────────────────────────────────────────────────
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        try:
            import importlib.util

            # 04_llm_agent.py starts with a digit — cannot use normal import
            # Use importlib to load it by file path instead
            agent_path = BASE_DIR / "04_llm_agent.py"
            if not agent_path.exists():
                raise FileNotFoundError(f"Agent file not found: {agent_path}")

            spec = importlib.util.spec_from_file_location("llm_agent_04", str(agent_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            CancerGPTAgent = module.CancerGPTAgent
            _agent = CancerGPTAgent()
            log.info("CancerGPT agent initialized")
        except Exception as e:
            log.error(f"Agent init failed: {e}")
    return _agent

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "CancerGPT",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "web_interface": "/web/index.html",
        "patient_browser": "/web/patient_browser.html"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/patients")
async def list_patients(limit: int = 500):
    """
    Returns patient list with decoded cancer type names.
    Used by patient_browser.html to populate the table and filter dropdown.
    """
    try:
        import pandas as pd
        for path in [FEAT_DIR / "ml_features.parquet",
                     PROC_DIR / "patient_profiles.parquet"]:
            if path.exists():
                df = pd.read_parquet(path)

                # Decode cancer type to text name
                if "CANCER_TYPE" in df.columns:
                    df["CANCER_TYPE"] = df["CANCER_TYPE"].apply(decode_cancer_type)

                cols = [c for c in ["PATIENT_ID", "AGE", "SEX", "CANCER_TYPE",
                                    "STAGE", "TMB", "OS_MONTHS"] if c in df.columns]
                records = df[cols].head(limit).fillna("").to_dict(orient="records")
                return {
                    "patients": records,
                    "total": len(df)
                }
        return {"patients": [], "total": 0, "message": "Run pipeline first"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{patient_id}")
async def get_patient(patient_id: str):
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")
    patient = agent.patient_db.get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    profile = agent.patient_db.get_web_profile(patient)
    profile["cancer_type"] = decode_cancer_type(profile.get("cancer_type"))
    return {"patient_id": patient_id, "profile": profile}

@app.get("/patients/{patient_id}/profile")
async def get_patient_profile(patient_id: str):
    """Structured profile for web UI panels."""
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")
    patient = agent.patient_db.get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    profile = agent.patient_db.get_web_profile(patient)
    profile["cancer_type"] = decode_cancer_type(profile.get("cancer_type"))
    return profile

@app.post("/consult/load")
async def load_patient(req: PatientLoadRequest):
    """
    Load patient and return:
      - initial_assessment: LLM clinical text
      - profile: structured dict with decoded cancer type for web UI sidebar
    """
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")
    try:
        patient_raw = agent.patient_db.get_patient(req.patient_id)
        if patient_raw is None:
            available = agent.patient_db.list_patients(5)
            raise HTTPException(
                status_code=404,
                detail=f"Patient '{req.patient_id}' not found. Try: {', '.join(available)}"
            )
        profile = agent.patient_db.get_web_profile(patient_raw)
        profile["cancer_type"] = decode_cancer_type(profile.get("cancer_type"))

        response = agent.load_patient(req.patient_id)

        return {
            "patient_id":         req.patient_id,
            "initial_assessment": response,
            "profile":            profile,
            "status":             "consultation_active"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/consult/chat")
async def chat(req: ChatRequest):
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")
    try:
        if req.patient_id and (
            not agent.current_patient or
            str(agent.current_patient.get("PATIENT_ID", "")) != req.patient_id
        ):
            agent.load_patient(req.patient_id)
        response = agent.chat(req.message)
        return {"response": response, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/consult/query")
async def general_query(req: QueryRequest):
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")
    try:
        response = agent.ask(req.query)
        return {"response": response, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/consult/report")
async def generate_report(req: PatientLoadRequest):
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")
    try:
        if req.patient_id:
            agent.load_patient(req.patient_id)
        result = agent.generate_report()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/cohort")
async def cohort_analytics():
    try:
        import pandas as pd
        for path in [FEAT_DIR / "ml_features.parquet",
                     PROC_DIR / "patient_profiles.parquet"]:
            if path.exists():
                df = pd.read_parquet(path)
                stats = {"total_patients": len(df)}
                if "CANCER_TYPE" in df.columns:
                    decoded = df["CANCER_TYPE"].apply(decode_cancer_type)
                    stats["cancer_types"] = decoded.value_counts().head(10).to_dict()
                if "AGE" in df.columns:
                    stats["age_mean"] = round(float(df["AGE"].mean()), 1)
                if "TMB" in df.columns:
                    stats["tmb_mean"] = round(float(df["TMB"].mean()), 1)
                    stats["tmb_high_pct"] = round(float((df["TMB"] >= 10).mean() * 100), 1)
                if "OS_MONTHS" in df.columns:
                    stats["median_os_months"] = round(float(df["OS_MONTHS"].median()), 1)
                return stats
        return {"message": "Run pipeline first"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/top_genes")
async def top_genes(n: int = 15):
    try:
        import pandas as pd
        for path in [FEAT_DIR / "ml_features.parquet",
                     PROC_DIR / "patient_profiles.parquet"]:
            if path.exists():
                df = pd.read_parquet(path)
                mut_cols = [c for c in df.columns if c.startswith("MUT_")]
                if not mut_cols:
                    return {"genes": [], "message": "No mutation features found"}
                gene_freq = {
                    col.replace("MUT_", ""): int(df[col].sum())
                    for col in mut_cols
                }
                sorted_genes = sorted(gene_freq.items(), key=lambda x: x[1], reverse=True)[:n]
                return {
                    "genes": [
                        {"gene": g, "count": c,
                         "frequency_pct": round(c / len(df) * 100, 1)}
                        for g, c in sorted_genes
                    ]
                }
        return {"genes": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*55)
    print("  CancerGPT API Server starting...")
    print("  API docs:        http://localhost:8000/docs")
    print("  Health check:    http://localhost:8000/health")
    print("  Patient list:    http://localhost:8000/patients")
    print("  Patient browser: http://localhost:8000/web/patient_browser.html")
    print("  Consultation:    http://localhost:8000/web/index.html")
    print("="*55 + "\n")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
