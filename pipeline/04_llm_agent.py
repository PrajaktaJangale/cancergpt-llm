"""
CancerGPT — Step 4: LLM Agent (Groq Version)
Uses Groq API (free, very fast) as backbone LLM.
Model: llama-3.3-70b-versatile (best free model on Groq)

SETUP:
  1. pip install groq
  2. Paste your Groq key below (starts with gsk_...)
  3. python 04_llm_agent.py
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cancergpt.agent")

# ── Paths ────────────────────────────────────────────────────────────────────
INDEX_DIR   = Path("rag_index")
FEAT_DIR    = Path("features")
PROC_DIR    = Path("processed")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ── PASTE YOUR GROQ API KEY HERE ─────────────────────────────────────────────
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
# Get free key from: https://console.groq.com/keys
# Key starts with: gsk_...
# ─────────────────────────────────────────────────────────────────────────────

# Groq models to try in order (all free)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


# ─── System Prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are CancerGPT, an advanced clinical decision support AI assistant
designed to help oncologists during patient consultations and treatment planning.

Your Role:
You assist board-certified oncologists by:
1. Interpreting multi-omic cancer genomics data (mutations, CNAs, structural variants)
2. Matching patient profiles to evidence-based treatment guidelines (NCCN, ESMO, ASCO)
3. Identifying predictive biomarkers and targeted therapy opportunities
4. Assessing prognosis based on clinical and molecular features
5. Generating structured clinical consultation reports

Clinical Reasoning Framework:
For each patient, reason through:
- Genomic Alterations: Identify actionable mutations, amplifications, deletions
- Biomarker Status: TMB, MSI, HRD, PD-L1, BRCA, EGFR, ALK, KRAS, BRAF etc.
- Treatment Tiers:
  * Tier 1A: FDA-approved for this cancer type + biomarker (strongest evidence)
  * Tier 1B: FDA-approved for different cancer type (pan-tumor approval)
  * Tier 2: Clinical trial or off-label with strong evidence
  * Tier 3: Pre-clinical or mechanistic rationale only
- Prognosis: Integrate molecular and clinical features
- Clinical Trials: Suggest relevant NCT numbers when applicable

Output Format - always use this structure:
CLINICAL SUMMARY: Brief patient overview
KEY GENOMIC FINDINGS: Actionable alterations with significance
TREATMENT RECOMMENDATIONS: Ranked by Tier (1A, 1B, 2, 3)
PROGNOSTIC ASSESSMENT: Estimated outcomes
MONITORING PLAN: Follow-up biomarkers and schedule
SUPPORTING EVIDENCE: Guideline references (NCCN, ESMO, FDA)

Safety: ALWAYS end every response with:
[DISCLAIMER: This AI output requires review and validation by a licensed oncologist
before clinical application. CancerGPT supports but does not replace physician judgment.]
"""


# ─── TF-IDF RAG Retriever ────────────────────────────────────────────────────

class RAGRetriever:
    """Loads the TF-IDF index built by 03_rag_indexing.py."""

    def __init__(self):
        self._vectorizer  = None
        self._matrix      = None
        self._docs        = []
        self._metas       = []
        self._load()

    def _load(self):
        index_path = INDEX_DIR / "tfidf_index.pkl"
        if not index_path.exists():
            log.warning(f"RAG index not found at {index_path}. Run 03_rag_indexing.py first.")
            return
        try:
            with open(index_path, "rb") as f:
                bundle = pickle.load(f)
            self._vectorizer = bundle["vectorizer"]
            self._matrix     = bundle["tfidf_matrix"]
            raw_docs         = bundle["documents"]

            if raw_docs and isinstance(raw_docs[0], dict):
                self._docs  = [d.get("text", str(d)) for d in raw_docs]
                self._metas = [d.get("metadata", {}) for d in raw_docs]
            else:
                self._docs  = [str(d) for d in raw_docs]
                self._metas = [{} for _ in raw_docs]

            log.info(f"TF-IDF RAG index loaded: {len(self._docs)} docs, "
                     f"{self._matrix.shape[1]} terms")
        except Exception as e:
            log.warning(f"RAG index load failed: {e}")

    def retrieve(self, query: str, n_results: int = 5) -> str:
        """Return formatted context string for the LLM."""
        if self._vectorizer is None:
            return ""
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            q_vec   = self._vectorizer.transform([query])
            scores  = cosine_similarity(q_vec, self._matrix).flatten()
            top_idx = np.argsort(scores)[::-1][:n_results]

            parts = []
            for i in top_idx:
                if scores[i] < 0.01:
                    continue
                meta     = self._metas[i] if self._metas else {}
                source   = meta.get("source",   "Unknown")
                doc_type = meta.get("doc_type", "")
                sim      = round(float(scores[i]), 3)
                parts.append(
                    f"[{source} | {doc_type} | score:{sim}]\n{self._docs[i]}"
                )
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            log.error(f"RAG retrieval error: {e}")
            return ""


# ─── Patient Database ─────────────────────────────────────────────────────────

class PatientDatabase:
    def __init__(self):
        self.profiles = None
        for path in [
            FEAT_DIR / "ml_features.parquet",
            PROC_DIR / "patient_profiles.parquet",
        ]:
            if path.exists():
                try:
                    import pandas as pd
                    self.profiles = pd.read_parquet(path)
                    log.info(f"Patient database loaded: {len(self.profiles)} patients "
                             f"from {path.name}")
                    break
                except Exception as e:
                    log.warning(f"Could not load {path.name}: {e}")

        if self.profiles is None:
            log.warning("No patient database found. Run pipeline steps 01 and 02 first.")

    def list_patients(self, n: int = 20) -> List[str]:
        if self.profiles is None or "PATIENT_ID" not in self.profiles.columns:
            return []
        return self.profiles["PATIENT_ID"].astype(str).tolist()[:n]

    def get_patient(self, patient_id: str) -> Optional[Dict]:
        if self.profiles is None or "PATIENT_ID" not in self.profiles.columns:
            return None
        pid_clean = patient_id.strip()
        matches = self.profiles[
            self.profiles["PATIENT_ID"].astype(str).str.strip() == pid_clean
        ]
        if len(matches) == 0:
            matches = self.profiles[
                self.profiles["PATIENT_ID"].astype(str).str.upper() == pid_clean.upper()
            ]
        if len(matches) == 0:
            return None

        row    = matches.iloc[0]
        result = {}
        import pandas as pd
        import numpy as np
        for k, v in row.items():
            try:
                if isinstance(v, list):
                    result[k] = v
                elif pd.isna(v):
                    continue
                elif isinstance(v, np.integer):
                    result[k] = int(v)
                elif isinstance(v, np.floating):
                    result[k] = float(v)
                else:
                    result[k] = v
            except (TypeError, ValueError):
                result[k] = str(v)
        return result

    def format_for_llm(self, patient: Dict) -> str:
        lines = ["== PATIENT CLINICAL & GENOMIC PROFILE ==\n"]
        priority = [
            "PATIENT_ID", "AGE", "SEX", "CANCER_TYPE", "CANCER_TYPE_DETAILED",
            "STAGE", "STAGE_NUM", "SUBTYPE", "TMB", "MSI_STATUS",
            "OS_MONTHS", "OS_STATUS", "DFS_MONTHS", "DFS_STATUS",
            "TOP_MUTATED_GENES", "N_ONCOGENIC", "MSI_PROXY", "CIN_INDEX", "N_SAMPLES",
        ]
        for field in priority:
            if field in patient:
                val = patient[field]
                if isinstance(val, list):
                    val = ", ".join(str(g) for g in val[:10])
                lines.append(f"  {field}: {val}")

        mut_genes = [
            k.replace("MUT_", "") for k, v in patient.items()
            if k.startswith("MUT_") and str(v) in ("1", "1.0")
        ]
        if mut_genes:
            lines.append(f"  MUTATED_GENES: {', '.join(mut_genes)}")

        cna_altered = []
        for k, v in patient.items():
            if k.startswith("CNA_"):
                try:
                    fv = float(v)
                    if fv != 0:
                        label = "Amp" if fv > 0 else "Del"
                        cna_altered.append(f"{k.replace('CNA_', '')}({label}{int(fv):+d})")
                except (TypeError, ValueError):
                    pass
        if cna_altered:
            lines.append(f"  CNA_ALTERATIONS: {', '.join(cna_altered)}")

        return "\n".join(lines)


# ─── Groq LLM Call ───────────────────────────────────────────────────────────

def call_groq(messages: List[Dict], max_tokens: int = 2000) -> str:
    """Call Groq API — fast, free LLM inference."""
    key = GROQ_API_KEY.strip()
    if not key or key == "PASTE_YOUR_GROQ_KEY_HERE":
        return ("ERROR: Groq API key not set.\n"
                "Open 04_llm_agent.py and replace PASTE_YOUR_GROQ_KEY_HERE "
                "with your real key from https://console.groq.com/keys")
    try:
        from groq import Groq
    except ImportError:
        return ("ERROR: groq package not installed.\n"
                "Run: pip install groq")

    client = Groq(api_key=key)

    # Build messages with system prompt
    groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages:
        groq_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    last_error = None
    for model in GROQ_MODELS:
        try:
            log.info(f"Trying Groq model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=groq_messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            result = response.choices[0].message.content
            log.info(f"Success with model: {model}")
            return result
        except Exception as e:
            err_str = str(e)
            log.warning(f"Model {model} failed: {err_str[:100]}")
            last_error = e
            # If rate limited wait and retry same model
            if "rate_limit" in err_str.lower() or "429" in err_str:
                log.info("Rate limited — trying next model")
            continue

    return f"ERROR: All Groq models failed. Last error: {str(last_error)}"


# ─── CancerGPT Agent ─────────────────────────────────────────────────────────

class CancerGPTAgent:

    def __init__(self):
        self.retriever        = RAGRetriever()
        self.patient_db       = PatientDatabase()
        self.history:         List[Dict] = []
        self.current_patient: Optional[Dict] = None

    def load_patient(self, patient_id: str) -> str:
        patient = self.patient_db.get_patient(patient_id)
        if patient is None:
            available = self.patient_db.list_patients(5)
            return (f"Patient '{patient_id}' not found.\n"
                    f"Try one of these: {', '.join(available)}")

        self.current_patient = patient
        self.history         = []

        patient_context = self.patient_db.format_for_llm(patient)
        cancer    = str(patient.get("CANCER_TYPE", "cancer"))
        mut_genes = [
            k.replace("MUT_", "") for k, v in patient.items()
            if k.startswith("MUT_") and str(v) in ("1", "1.0")
        ][:6]
        rag_query   = f"{cancer} {' '.join(mut_genes)} treatment biomarkers guidelines"
        rag_context = self.retriever.retrieve(rag_query, n_results=5)

        prompt = f"""{patient_context}

== RETRIEVED CLINICAL GUIDELINES (RAG) ==
{rag_context if rag_context else '[General knowledge mode]'}

Please provide a complete initial clinical assessment:
1. Key genomic findings and clinical significance
2. Actionable biomarkers identified
3. Treatment recommendations with evidence tiers (1A / 1B / 2 / 3)
4. Prognostic assessment
5. MDT discussion priorities
"""
        self.history.append({"role": "user", "content": prompt})
        response = call_groq(self.history)
        self.history.append({"role": "assistant", "content": response})
        log.info(f"Patient {patient_id} loaded")
        return response

    def chat(self, message: str) -> str:
        cancer      = str(self.current_patient.get("CANCER_TYPE", "")) if self.current_patient else ""
        rag_context = self.retriever.retrieve(f"{cancer} {message}", n_results=3)
        content     = f"{message}\n\n[Retrieved context]\n{rag_context}" if rag_context else message
        self.history.append({"role": "user", "content": content})
        response = call_groq(self.history)
        self.history.append({"role": "assistant", "content": response})
        return response

    def generate_report(self) -> Dict:
        if self.current_patient is None:
            return {"report_text": "ERROR: No patient loaded. Use 'load <PATIENT_ID>' first.",
                    "report_path": ""}

        report_prompt = f"""Generate a formal structured clinical consultation report.

# CancerGPT Clinical Consultation Report
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Patient ID: {self.current_patient.get('PATIENT_ID', 'Unknown')}
System: CancerGPT v1.0 (Groq-powered)

## 1. CLINICAL SUMMARY
## 2. GENOMIC PROFILE
   ### 2a. Somatic Mutations (Actionable)
   ### 2b. Copy Number Alterations
   ### 2c. Tumor Microenvironment Biomarkers (TMB, MSI, CIN)
## 3. TREATMENT RECOMMENDATIONS
   ### Tier 1A (FDA-Approved, On-Label)
   ### Tier 1B (FDA-Approved, Pan-Tumor)
   ### Tier 2 (Clinical Trials / Strong Evidence)
   ### Standard of Care
## 4. PROGNOSTIC ASSESSMENT
## 5. MONITORING & FOLLOW-UP
## 6. MDT DISCUSSION POINTS
## 7. CLINICAL TRIAL ELIGIBILITY

[DISCLAIMER: Generated by CancerGPT AI. Requires oncologist validation before clinical use.]
"""
        self.history.append({"role": "user", "content": report_prompt})
        report_text = call_groq(self.history, max_tokens=3000)
        self.history.append({"role": "assistant", "content": report_text})

        pid       = str(self.current_patient.get("PATIENT_ID", "unknown"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path      = REPORTS_DIR / f"report_{pid}_{timestamp}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)
        log.info(f"Report saved: {path}")

        return {
            "report_text": report_text,
            "report_path": str(path),
            "patient_id":  pid,
        }

    def ask(self, question: str) -> str:
        rag_context = self.retriever.retrieve(question, n_results=5)
        content     = (f"{question}\n\n[Retrieved Guidelines]\n{rag_context}"
                       if rag_context else question)
        messages = [{"role": "user", "content": content}]
        return call_groq(messages)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def cli():
    print("\n" + "=" * 62)
    print("  CancerGPT — Clinical Decision Support System")
    print("  Powered by Groq LLM (free) + TF-IDF RAG")
    print("  Type 'help' for commands, 'quit' to exit")
    print("=" * 62 + "\n")

    agent = CancerGPTAgent()

    patients = agent.patient_db.list_patients(8)
    if patients:
        print(f"Sample patient IDs: {', '.join(patients)}\n")

    while True:
        try:
            user_input = input("Doctor > ").strip()
            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Session ended.")
                break

            elif user_input.lower() == "help":
                print("""
Commands:
  load <PATIENT_ID>    Load a patient  (e.g. load P-0005935)
  report               Generate formal clinical report
  ask <question>       General oncology question
  patients             Show all patient IDs
  history              Show conversation length
  clear                Clear conversation history
  quit                 Exit CancerGPT
                """)

            elif user_input.lower() == "patients":
                ids = agent.patient_db.list_patients(50)
                if ids:
                    print("\nAvailable patient IDs:")
                    for i, pid in enumerate(ids, 1):
                        print(f"  {i:>3}. {pid}")
                    print()
                else:
                    print("No patients found.\n")

            elif user_input.lower().startswith("load "):
                pid = user_input[5:].strip()
                if not pid:
                    print("Usage: load <PATIENT_ID>\n")
                    continue
                print(f"\nLoading patient {pid} — please wait...\n")
                response = agent.load_patient(pid)
                print(f"CancerGPT:\n{response}\n")

            elif user_input.lower() == "report":
                if agent.current_patient is None:
                    print("Load a patient first: load <PATIENT_ID>\n")
                    continue
                print("\nGenerating clinical report...\n")
                result = agent.generate_report()
                print(result["report_text"])
                print(f"\n[Saved to: {result['report_path']}]\n")

            elif user_input.lower().startswith("ask "):
                question = user_input[4:].strip()
                if not question:
                    print("Usage: ask <question>\n")
                    continue
                print("\nCancerGPT:\n")
                print(agent.ask(question))
                print()

            elif user_input.lower() == "history":
                print(f"Conversation turns: {len(agent.history) // 2}\n")

            elif user_input.lower() == "clear":
                agent.history = []
                print("Conversation cleared.\n")

            else:
                if agent.current_patient is None:
                    print("Tip: use 'load <PATIENT_ID>' for patient-specific answers\n")
                print("CancerGPT:\n")
                print(agent.chat(user_input))
                print()

        except KeyboardInterrupt:
            print("\nType 'quit' to exit.\n")


if __name__ == "__main__":
    cli()
