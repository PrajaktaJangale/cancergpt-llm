"""
CancerGPT — FastAPI Backend
RESTful API for the CancerGPT web interface.
Endpoints for patient lookup, chat, report generation,
and cohort-level analytics.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cancergpt.api")

app = FastAPI(
    title="CancerGPT API",
    description="Clinical Decision Support AI for Oncology",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Lazy-load agent ────────────────────────────────────────────────────────
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        try:
            from pipeline.llm_agent import CancerGPTAgent
            _agent = CancerGPTAgent()
            log.info("CancerGPT agent initialized")
        except Exception as e:
            log.error(f"Agent init failed: {e}")
    return _agent


# ─── Schemas ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    patient_id: Optional[str] = None
    session_id: Optional[str] = None

class PatientLoadRequest(BaseModel):
    patient_id: str

class ReportRequest(BaseModel):
    patient_id: Optional[str] = None

class QueryRequest(BaseModel):
    query: str


# ─── In-memory session store (use Redis in production) ───────────────────────
sessions = {}


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "CancerGPT",
        "version": "1.0.0",
        "status":  "operational",
        "description": "Clinical Decision Support AI for Oncology"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/patients")
async def list_patients(limit: int = 50, cancer_type: str = None):
    """List patients in the database."""
    try:
        import pandas as pd
        feat_path = Path("features/ml_features.parquet")
        if not feat_path.exists():
            return {"patients": [], "total": 0, "message": "Run preprocessing pipeline first"}

        df = pd.read_parquet(feat_path)

        if cancer_type:
            df = df[df["CANCER_TYPE"].str.contains(cancer_type, case=False, na=False)]

        cols = [c for c in ["PATIENT_ID", "AGE", "SEX", "CANCER_TYPE",
                             "CANCER_TYPE_DETAILED", "STAGE", "TMB", "OS_MONTHS"] if c in df.columns]
        subset = df[cols].head(limit)
        return {
            "patients": subset.to_dict(orient="records"),
            "total":    len(df),
            "returned": len(subset)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/patients/{patient_id}")
async def get_patient(patient_id: str):
    """Get full patient profile."""
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    patient = agent.patient_db.get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    return {"patient_id": patient_id, "profile": patient}


@app.post("/consult/load")
async def load_patient(req: PatientLoadRequest):
    """Load a patient and get initial clinical assessment."""
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    try:
        response = agent.load_patient(req.patient_id)
        session_id = f"session_{req.patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        sessions[session_id] = {
            "patient_id": req.patient_id,
            "created_at": datetime.now().isoformat(),
            "agent_snapshot": agent  # In production: serialize conversation
        }
        return {
            "session_id":     session_id,
            "patient_id":     req.patient_id,
            "initial_assessment": response,
            "status":         "consultation_active"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/consult/chat")
async def chat(req: ChatRequest):
    """Send a message in an active consultation."""
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    try:
        if req.patient_id and (not agent.current_patient or
           str(agent.current_patient.get("PATIENT_ID")) != req.patient_id):
            agent.load_patient(req.patient_id)

        response = agent.chat(req.message)
        return {
            "response":    response,
            "patient_id":  req.patient_id,
            "timestamp":   datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/consult/query")
async def general_query(req: QueryRequest):
    """Answer a general oncology question (no specific patient)."""
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    try:
        response = agent.query_without_patient(req.query)
        return {"response": response, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/consult/report")
async def generate_report(req: ReportRequest):
    """Generate a formal clinical consultation report."""
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not available")

    try:
        result = agent.generate_clinical_report(req.patient_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/cohort")
async def cohort_analytics():
    """Cohort-level statistics for the dashboard."""
    try:
        import pandas as pd
        import numpy as np
        feat_path = Path("features/ml_features.parquet")
        if not feat_path.exists():
            return {"message": "Run preprocessing pipeline first"}

        df = pd.read_parquet(feat_path)
        stats = {}

        if "CANCER_TYPE" in df.columns:
            stats["cancer_type_distribution"] = (
                df["CANCER_TYPE"].value_counts().head(10).to_dict()
            )

        if "AGE" in df.columns:
            stats["age_stats"] = {
                "mean":   round(df["AGE"].mean(), 1),
                "median": round(df["AGE"].median(), 1),
                "min":    int(df["AGE"].min()),
                "max":    int(df["AGE"].max())
            }

        if "TMB" in df.columns:
            stats["tmb_stats"] = {
                "mean":        round(df["TMB"].mean(), 1),
                "high_tmb_pct": round((df["TMB"] >= 10).mean() * 100, 1)
            }

        if "OS_MONTHS" in df.columns:
            stats["survival_stats"] = {
                "median_os_months": round(df["OS_MONTHS"].median(), 1),
                "n_deceased":       int(df.get("OS_STATUS", pd.Series()).sum()) if "OS_STATUS" in df else None
            }

        stats["total_patients"] = len(df)
        stats["total_features"] = len(df.columns)

        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/top_genes")
async def top_mutated_genes(n: int = 20):
    """Most commonly mutated genes in the cohort."""
    try:
        import pandas as pd
        feat_path = Path("features/ml_features.parquet")
        if not feat_path.exists():
            return {"genes": []}

        df = pd.read_parquet(feat_path)
        mut_cols = [c for c in df.columns if c.startswith("MUT_")]

        if not mut_cols:
            return {"genes": [], "message": "No mutation feature columns found"}

        gene_freq = {col.replace("MUT_", ""): int(df[col].sum())
                     for col in mut_cols}
        sorted_genes = sorted(gene_freq.items(), key=lambda x: x[1], reverse=True)[:n]

        return {
            "genes": [{"gene": g, "count": c, "frequency": round(c/len(df)*100, 1)}
                      for g, c in sorted_genes]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
