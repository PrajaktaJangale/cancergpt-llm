"""
CancerGPT — Step 3: RAG Indexing (TF-IDF Version)
NO torch / sentence-transformers dependency.
Uses TF-IDF + cosine similarity for retrieval.
Works on any Windows machine without GPU or DLL issues.

Indexes:
  - 76 patient natural-language summaries
  - NCCN/ESMO clinical guidelines
  - Drug-gene interaction knowledge (DGIdb)
"""

import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict

# Pure sklearn — no torch needed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cancergpt.rag")

PROC_DIR  = Path("processed")
INDEX_DIR = Path("rag_index")
INDEX_DIR.mkdir(exist_ok=True)


# ─── Document Builders ───────────────────────────────────────────────────────

def build_patient_documents(nl_summaries_path: Path) -> List[Dict]:
    """Load patient NL summaries as RAG documents."""
    with open(nl_summaries_path, encoding="utf-8") as f:
        records = json.load(f)
    docs = []
    for r in records:
        docs.append({
            "id":       f"patient_{r['PATIENT_ID']}",
            "text":     r["nl_summary"],
            "metadata": {
                "source":     "patient_data",
                "patient_id": r["PATIENT_ID"],
                "doc_type":   "patient_profile"
            }
        })
    log.info(f"Built {len(docs)} patient documents")
    return docs


def build_guideline_documents() -> List[Dict]:
    """NCCN/ESMO clinical guideline knowledge base."""
    guidelines = [
        {
            "id": "nccn_brca_her2",
            "text": (
                "For HER2-positive breast cancer, NCCN guidelines recommend trastuzumab-based "
                "regimens as the standard of care. Pertuzumab is added for metastatic or neoadjuvant settings. "
                "ERBB2 amplification (CNA value +2) is the primary predictive biomarker. "
                "Patients with BRCA1/2 mutations may benefit from PARP inhibitors (olaparib, talazoparib). "
                "CDK4/6 inhibitors (palbociclib, ribociclib) plus endocrine therapy for HR+ HER2- disease."
            ),
            "metadata": {"source": "NCCN", "cancer_type": "Breast Cancer", "doc_type": "guideline"}
        },
        {
            "id": "nccn_nsclc_egfr",
            "text": (
                "Non-small cell lung cancer NSCLC with EGFR exon 19 deletions or L858R mutations: "
                "osimertinib third-generation EGFR TKI is first-line standard of care per NCCN and ESMO. "
                "EGFR T790M resistance mutation detected via liquid biopsy indicates osimertinib benefit "
                "in second-line after erlotinib or gefitinib. ALK ROS1 rearrangements: alectinib first-line. "
                "KRAS G12C mutation: sotorasib or adagrasib indicated. MET exon 14 skipping: capmatinib tepotinib."
            ),
            "metadata": {"source": "NCCN/ESMO", "cancer_type": "NSCLC", "doc_type": "guideline"}
        },
        {
            "id": "nccn_crc_msi",
            "text": (
                "Colorectal cancer with microsatellite instability-high MSI-H or mismatch repair "
                "deficiency dMMR: pembrolizumab is first-line standard per FDA approval and NCCN. "
                "KRAS NRAS BRAF mutations predict non-response to EGFR inhibitors cetuximab panitumumab. "
                "BRAF V600E mutation: encorafenib plus cetuximab combination is indicated. "
                "HER2 amplification in CRC: trastuzumab plus pertuzumab or lapatinib combinations."
            ),
            "metadata": {"source": "NCCN", "cancer_type": "Colorectal Cancer", "doc_type": "guideline"}
        },
        {
            "id": "nccn_melanoma_braf",
            "text": (
                "Melanoma with BRAF V600E or V600K mutation: combined BRAF plus MEK inhibition "
                "dabrafenib plus trametinib, vemurafenib plus cobimetinib, or encorafenib plus binimetinib "
                "is standard of care. PD-1 inhibitors pembrolizumab nivolumab are first-line for "
                "BRAF wild-type or as alternative to targeted therapy. "
                "ipilimumab plus nivolumab combination for high-risk advanced disease."
            ),
            "metadata": {"source": "NCCN", "cancer_type": "Melanoma", "doc_type": "guideline"}
        },
        {
            "id": "nccn_ovarian_brca",
            "text": (
                "Ovarian cancer with BRCA1 BRCA2 pathogenic variants: PARP inhibitors olaparib niraparib "
                "rucaparib as maintenance after platinum-based chemotherapy per NCCN ESMO. "
                "Homologous recombination deficiency HRD score predicts broader PARP inhibitor benefit. "
                "Bevacizumab anti-VEGF added for advanced high-risk disease. "
                "Platinum-sensitive recurrence: PARP inhibitor maintenance strongly recommended."
            ),
            "metadata": {"source": "NCCN/ESMO", "cancer_type": "Ovarian Cancer", "doc_type": "guideline"}
        },
        {
            "id": "nccn_prostate_ar",
            "text": (
                "Prostate cancer androgen deprivation therapy ADT backbone for metastatic disease. "
                "AR pathway inhibitors enzalutamide abiraterone apalutamide darolutamide added for "
                "metastatic hormone-sensitive or castration-resistant disease mCRPC. "
                "BRCA1 BRCA2 ATM mutations: olaparib or rucaparib indicated for post-chemotherapy mCRPC. "
                "PTEN loss associated with PI3K pathway activation and poor prognosis."
            ),
            "metadata": {"source": "NCCN", "cancer_type": "Prostate Cancer", "doc_type": "guideline"}
        },
        {
            "id": "tmb_immunotherapy",
            "text": (
                "Tumor mutational burden TMB-High 10 or more mutations per megabase is FDA-approved "
                "pan-tumor biomarker for pembrolizumab Keytruda per KEYNOTE-158 trial. "
                "High TMB correlates with response to immune checkpoint inhibitors immunotherapy. "
                "MSI-H dMMR status is separate but related biomarker for immunotherapy response. "
                "PD-L1 expression TPS CPS score also guides immunotherapy eligibility across cancers. "
                "Low TMB tumors may still respond if MSI-H or if PD-L1 expression is high."
            ),
            "metadata": {"source": "FDA/NCCN", "cancer_type": "Pan-tumor", "doc_type": "biomarker_guideline"}
        },
        {
            "id": "cin_genomic_instability",
            "text": (
                "Chromosomal instability CIN is associated with poor prognosis across cancer types. "
                "High CIN index greater than 25 percent genome altered correlates with increased "
                "metastatic potential and resistance to DNA-damaging chemotherapy. "
                "WNT pathway activation often co-occurs with CIN. "
                "ATR CHK1 inhibitors are in clinical trials for CIN-high tumors. "
                "TP53 mutations strongly associated with chromosomal instability."
            ),
            "metadata": {"source": "Literature", "cancer_type": "Pan-tumor", "doc_type": "biomarker_guideline"}
        },
    ]
    log.info(f"Built {len(guidelines)} guideline documents")
    return guidelines


def build_drug_gene_documents() -> List[Dict]:
    """Drug-gene interaction knowledge base."""
    interactions = [
        {"gene": "EGFR",   "drugs": "Erlotinib, Gefitinib, Afatinib, Osimertinib, Dacomitinib",    "type": "EGFR TKI inhibitor"},
        {"gene": "BRAF",   "drugs": "Vemurafenib, Dabrafenib, Encorafenib",                         "type": "BRAF inhibitor"},
        {"gene": "ERBB2",  "drugs": "Trastuzumab, Pertuzumab, Lapatinib, T-DM1, Tucatinib",        "type": "HER2 inhibitor"},
        {"gene": "BRCA1",  "drugs": "Olaparib, Niraparib, Rucaparib, Talazoparib",                 "type": "PARP inhibitor"},
        {"gene": "BRCA2",  "drugs": "Olaparib, Niraparib, Rucaparib, Talazoparib",                 "type": "PARP inhibitor"},
        {"gene": "ALK",    "drugs": "Crizotinib, Alectinib, Brigatinib, Lorlatinib",               "type": "ALK inhibitor"},
        {"gene": "MET",    "drugs": "Capmatinib, Tepotinib, Crizotinib",                            "type": "MET inhibitor"},
        {"gene": "KRAS",   "drugs": "Sotorasib, Adagrasib (KRAS G12C specific)",                   "type": "KRAS inhibitor"},
        {"gene": "IDH1",   "drugs": "Ivosidenib",                                                   "type": "IDH1 inhibitor"},
        {"gene": "IDH2",   "drugs": "Enasidenib",                                                   "type": "IDH2 inhibitor"},
        {"gene": "FLT3",   "drugs": "Midostaurin, Quizartinib, Gilteritinib",                       "type": "FLT3 inhibitor"},
        {"gene": "CDK4",   "drugs": "Palbociclib, Ribociclib, Abemaciclib",                        "type": "CDK4/6 inhibitor"},
        {"gene": "CDK6",   "drugs": "Palbociclib, Ribociclib, Abemaciclib",                        "type": "CDK4/6 inhibitor"},
        {"gene": "PIK3CA", "drugs": "Alpelisib with fulvestrant for HR+ breast cancer",             "type": "PI3K inhibitor"},
        {"gene": "RET",    "drugs": "Selpercatinib, Pralsetinib",                                   "type": "RET inhibitor"},
        {"gene": "NTRK1",  "drugs": "Larotrectinib, Entrectinib",                                  "type": "TRK inhibitor"},
        {"gene": "FGFR2",  "drugs": "Pemigatinib, Infigratinib for cholangiocarcinoma",            "type": "FGFR inhibitor"},
        {"gene": "TP53",   "drugs": "APR-246 eprenetapopt in clinical trials",                     "type": "TP53 reactivator"},
        {"gene": "PTEN",   "drugs": "mTOR inhibitors everolimus temsirolimus",                     "type": "pathway inhibitor"},
    ]
    docs = []
    for item in interactions:
        text = (
            f"Gene {item['gene']} mutation or alteration. "
            f"Targeted therapy drug interactions {item['type']}: {item['drugs']}. "
            f"Patients with {item['gene']} alterations may benefit from: {item['drugs']}. "
            f"Mechanism of action: {item['type']}. "
            f"Biomarker: {item['gene']} is predictive for {item['drugs']}."
        )
        docs.append({
            "id":       f"dgi_{item['gene'].lower()}",
            "text":     text,
            "metadata": {
                "source":   "DGIdb/Literature",
                "gene":     item["gene"],
                "doc_type": "drug_gene"
            }
        })
    log.info(f"Built {len(docs)} drug-gene documents")
    return docs


# ─── TF-IDF RAG Indexer ──────────────────────────────────────────────────────

class CancerRAGIndexer:
    """
    TF-IDF based RAG indexer.
    No torch, no GPU, no DLL issues.
    Uses sklearn TfidfVectorizer + cosine similarity.
    """

    def __init__(self, persist_dir: str = str(INDEX_DIR)):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        self.vectorizer  = None
        self.tfidf_matrix = None
        self.documents   = []  # list of {id, text, metadata}
        log.info(f"TF-IDF RAG indexer ready at {persist_dir}")

    def index_documents(self, docs: List[Dict]):
        """Build TF-IDF index from all documents."""
        self.documents = docs
        texts = [d["text"] for d in docs]

        # Medical-domain optimized TF-IDF settings
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),        # unigrams + bigrams
            min_df=1,
            max_df=0.95,
            max_features=10000,
            sublinear_tf=True,         # log normalization
            strip_accents="unicode",
            token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9\-]{1,}\b"
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        log.info(f"TF-IDF matrix built: {self.tfidf_matrix.shape} "
                 f"({len(docs)} docs x {self.tfidf_matrix.shape[1]} terms)")

    def query(self, query_text: str, n_results: int = 5,
              filter_source: str = None) -> List[Dict]:
        """Retrieve top-k relevant documents for a clinical query."""
        if self.vectorizer is None or self.tfidf_matrix is None:
            log.error("Index not built yet — call index_documents() first")
            return []

        query_vec = self.vectorizer.transform([query_text])
        scores    = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Apply source filter if requested
        if filter_source:
            for i, doc in enumerate(self.documents):
                if doc.get("metadata", {}).get("source") != filter_source:
                    scores[i] = 0.0

        top_indices = np.argsort(scores)[::-1][:n_results]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "text":       self.documents[idx]["text"],
                    "metadata":   self.documents[idx].get("metadata", {}),
                    "similarity": round(float(scores[idx]), 4)
                })
        return results

    def save(self):
        """Persist index to disk."""
        index_data = {
            "vectorizer":   self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "documents":    self.documents,
        }
        with open(self.persist_dir / "tfidf_index.pkl", "wb") as f:
            pickle.dump(index_data, f)
        log.info(f"Index saved to {self.persist_dir / 'tfidf_index.pkl'}")

    def load(self):
        """Load persisted index from disk."""
        index_path = self.persist_dir / "tfidf_index.pkl"
        if not index_path.exists():
            log.error(f"Index not found at {index_path}")
            return False
        with open(index_path, "rb") as f:
            index_data = pickle.load(f)
        self.vectorizer   = index_data["vectorizer"]
        self.tfidf_matrix = index_data["tfidf_matrix"]
        self.documents    = index_data["documents"]
        log.info(f"Index loaded: {len(self.documents)} documents")
        return True

    def save_manifest(self):
        """Save human-readable index manifest."""
        manifest = {
            "index_type":   "TF-IDF (sklearn)",
            "total_docs":   len(self.documents),
            "index_dir":    str(self.persist_dir),
            "doc_types":    {},
        }
        for doc in self.documents:
            dt = doc.get("metadata", {}).get("doc_type", "unknown")
            manifest["doc_types"][dt] = manifest["doc_types"].get(dt, 0) + 1

        with open(self.persist_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log.info(f"Manifest saved: {manifest}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("=== CancerGPT RAG Indexing ===")

    # Build document corpus
    all_docs = []

    nl_path = PROC_DIR / "patient_nl_summaries.json"
    if nl_path.exists():
        all_docs += build_patient_documents(nl_path)
    else:
        log.warning("patient_nl_summaries.json not found — run 01_data_preprocessing.py first")

    all_docs += build_guideline_documents()
    all_docs += build_drug_gene_documents()

    log.info(f"Total documents to index: {len(all_docs)}")

    # Build and save index
    indexer = CancerRAGIndexer()
    indexer.index_documents(all_docs)
    indexer.save()
    indexer.save_manifest()

    # Validation queries
    test_queries = [
        "EGFR mutated lung cancer treatment options",
        "BRCA1 ovarian cancer PARP inhibitor",
        "TMB high immunotherapy pembrolizumab",
        "TP53 mutation prognosis",
    ]

    log.info("\n--- Validation Queries ---")
    for query in test_queries:
        results = indexer.query(query, n_results=2)
        log.info(f"\nQuery: '{query}'")
        for r in results:
            source = r["metadata"].get("source", "?")
            log.info(f"  [{r['similarity']:.3f}] [{source}] {r['text'][:80]}...")

    log.info("\n" + "=" * 50)
    log.info("RAG indexing complete!")
    log.info(f"Documents indexed : {len(all_docs)}")
    log.info(f"Index saved to    : {INDEX_DIR.resolve()}")
    log.info("No torch/GPU required — TF-IDF index")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
