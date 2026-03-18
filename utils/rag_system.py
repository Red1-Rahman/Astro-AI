"""Lightweight RAG System for Astro-AI

A retrieval-augmented generation system using only existing dependencies.
Stores astronomical knowledge and retrieves relevant context for LLM queries.
No external vector databases — uses numpy and basic TF-IDF similarity.

Bug fixes vs original:
  1. add_scientific_result() no longer touches st.session_state — timestamps
     are taken from the stdlib datetime module instead.
  2. save_knowledge_base() no longer uses json.dump(default=str), which was
     silently converting SessionStateProxy / numpy objects to strings and
     producing a JSON that could not be round-tripped correctly.  All data
     is now sanitised through _to_serializable() before serialisation.
  3. _load_knowledge_base() no longer calls st.warning() — that crashes when
     the RAG is initialised outside a Streamlit render context (e.g. during
     @st.cache_resource construction).  Errors are logged instead.
  4. get_stats() no longer returns live sets — it returns sorted lists so the
     output is deterministic and JSON-serialisable.
"""

import json
import logging
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialisation helper (same contract as the one in openai_integration.py)
# ---------------------------------------------------------------------------

def _to_serializable(obj: Any, _depth: int = 0) -> Any:
    """
    Recursively convert *obj* to a JSON-serialisable form.

    Never raises — worst case returns a descriptive string.
    Hard depth cap at 8 to prevent runaway recursion.
    """
    if _depth > 8:
        return "<truncated>"

    # numpy
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return obj.tolist() if obj.size <= 200 else f"<array shape={obj.shape}>"
        if isinstance(obj, (_np.integer, _np.floating, _np.bool_)):
            return obj.item()
    except ImportError:
        pass

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v, _depth + 1) for v in obj]

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                out[str(k)] = _to_serializable(v, _depth + 1)
            except Exception:
                out[str(k)] = "<unserializable>"
        return out

    # Streamlit SessionStateProxy — never store in the knowledge base
    type_name = type(obj).__name__
    if "SessionState" in type_name or "Proxy" in type_name:
        return "<session_state — excluded>"

    # pandas
    try:
        import pandas as _pd
        if isinstance(obj, _pd.DataFrame):
            return f"<DataFrame rows={len(obj)} cols={list(obj.columns)}>"
    except ImportError:
        pass

    # bytes / matplotlib figures
    if isinstance(obj, (bytes, bytearray)) or hasattr(obj, "savefig"):
        return "<binary — excluded>"

    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


class AstroRAGSystem:
    """
    Lightweight Retrieval-Augmented Generation system for astronomical data.

    Uses TF-IDF with cosine similarity for document retrieval without
    external dependencies.  Designed specifically for astrophysical contexts
    and scientific explanations.
    """

    def __init__(
        self, knowledge_base_path: str = "data/astro_knowledge_base.json"
    ):
        self.knowledge_base_path = knowledge_base_path
        self.documents:        List[Dict[str, Any]] = []
        self.document_vectors: List[np.ndarray]     = []
        self.vocabulary:       Dict[str, int]        = {}
        self.idf_scores:       Dict[str, float]      = {}

        self._ensure_data_directory()
        self._load_knowledge_base()

    # -----------------------------------------------------------------------
    # Internal utilities
    # -----------------------------------------------------------------------

    def _ensure_data_directory(self):
        os.makedirs(os.path.dirname(self.knowledge_base_path), exist_ok=True)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenise scientific text, preserving numeric and technical terms."""
        tokens = re.findall(r'\b(?:\d+\.?\d*(?:e[+-]?\d+)?|\w+)\b', text.lower())
        return [t for t in tokens if len(t) > 1]

    def _compute_tf_idf(self, documents: List[str]) -> np.ndarray:
        """Compute TF-IDF matrix for a list of document strings."""
        if not documents:
            return np.zeros((0, 0))

        tokenized = [self._tokenize(d) for d in documents]

        # Vocabulary
        all_words = set(w for doc in tokenized for w in doc)
        self.vocabulary = {w: i for i, w in enumerate(sorted(all_words))}
        vocab_size = len(self.vocabulary)
        n_docs     = len(documents)

        # IDF
        word_doc_count: Dict[str, int] = defaultdict(int)
        for doc_tokens in tokenized:
            for w in set(doc_tokens):
                word_doc_count[w] += 1

        self.idf_scores = {
            w: math.log(n_docs / cnt)
            for w, cnt in word_doc_count.items()
        }

        # TF-IDF matrix
        matrix = np.zeros((n_docs, vocab_size), dtype=np.float32)
        for di, doc_tokens in enumerate(tokenized):
            tf_counts  = Counter(doc_tokens)
            doc_length = max(len(doc_tokens), 1)
            for w, cnt in tf_counts.items():
                if w in self.vocabulary:
                    wi = self.vocabulary[w]
                    matrix[di, wi] = (cnt / doc_length) * self.idf_scores[w]

        return matrix

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _rebuild_vectors(self):
        if not self.documents:
            self.document_vectors = []
            return
        matrix = self._compute_tf_idf([d["content"] for d in self.documents])
        self.document_vectors = [matrix[i] for i in range(len(self.documents))]

    # -----------------------------------------------------------------------
    # Public document management
    # -----------------------------------------------------------------------

    def add_document(self, content: str, metadata: Dict[str, Any]):
        """
        Add a document to the knowledge base.

        *metadata* is sanitised through _to_serializable() before storage so
        that no SessionStateProxy, numpy array, or other non-serialisable
        object is ever held in self.documents.
        """
        safe_meta = _to_serializable(metadata)
        doc = {
            "content":  str(content),
            "metadata": safe_meta,
            "id":       len(self.documents),
        }
        self.documents.append(doc)
        self._rebuild_vectors()

    def add_scientific_result(
        self,
        analysis_type: str,
        results: Dict[str, Any],
        interpretation: str,
        source_module: str,
    ):
        """
        Add scientific results with structured metadata.

        BUG FIX: timestamp now comes from datetime.now() — NOT from
        st.session_state, which is a SessionStateProxy and is unhashable /
        non-serialisable.
        """
        # Build content string
        content_parts = [str(interpretation)]

        safe_results = _to_serializable(results)
        if isinstance(safe_results, dict):
            if "key_metrics" in safe_results:
                km = safe_results["key_metrics"]
                if isinstance(km, dict):
                    content_parts.append(
                        "Key metrics: "
                        + ", ".join(f"{k}={v}" for k, v in km.items())
                    )
            if "method" in safe_results:
                content_parts.append(f"Analysis method: {safe_results['method']}")

        content = " ".join(content_parts)

        # BUG FIX: use stdlib datetime — no st.session_state access here
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        metadata = {
            "type":          "scientific_result",
            "analysis_type": analysis_type,
            "source_module": source_module,
            "results":       safe_results,   # already sanitised
            "timestamp":     timestamp,       # plain ISO string, always serialisable
        }

        # add_document sanitises metadata again as a safety net
        self.add_document(content, metadata)

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Return the top_k most relevant documents for *query*."""
        if not self.documents or not self.vocabulary:
            return []

        # Query vector
        q_tokens  = self._tokenize(query)
        q_vector  = np.zeros(len(self.vocabulary), dtype=np.float32)
        if q_tokens:
            q_tf     = Counter(q_tokens)
            q_length = len(q_tokens)
            for w, cnt in q_tf.items():
                if w in self.vocabulary:
                    wi = self.vocabulary[w]
                    q_vector[wi] = (cnt / q_length) * self.idf_scores.get(w, 0.0)

        # Cosine similarities
        sims = [
            (self.documents[i], self._cosine_similarity(q_vector, dv))
            for i, dv in enumerate(self.document_vectors)
        ]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def generate_context(self, query: str, top_k: int = 3) -> str:
        """Generate a context block suitable for inclusion in an LLM prompt."""
        retrieved = self.retrieve(query, top_k)
        if not retrieved:
            return "No relevant context found in knowledge base."

        parts = ["=== RETRIEVED CONTEXT ==="]
        included = 0
        for i, (doc, score) in enumerate(retrieved, 1):
            if score < 0.01:
                continue
            parts.append(f"\n[Context {i}] (Relevance: {score:.3f})")
            parts.append(f"Type: {doc['metadata'].get('analysis_type', 'general')}")
            parts.append(f"Source: {doc['metadata'].get('source_module', 'unknown')}")
            parts.append(f"Content: {doc['content']}")
            included += 1

        if included == 0:
            return "No sufficiently relevant context found."

        parts.append("\n=== END CONTEXT ===")
        return "\n".join(parts)

    # -----------------------------------------------------------------------
    # Default knowledge base
    # -----------------------------------------------------------------------

    def populate_with_defaults(self):
        """Populate the knowledge base with core astrophysical background."""
        defaults = [
            {
                "content": (
                    "21cm intensity mapping probes the cosmic dark ages and reionization. "
                    "The 21cm hyperfine line of neutral hydrogen traces large-scale structure "
                    "formation. The global signal shows an absorption trough near z~17-20 "
                    "(Pritchard & Loeb 2012) and evolves through heating and reionization "
                    "to reach zero signal by z~6 (Fan+2006)."
                ),
                "metadata": {
                    "type": "explanation",
                    "analysis_type": "cosmic_evolution",
                    "source_module": "default",
                },
            },
            {
                "content": (
                    "Galaxy clusters are the largest gravitationally bound structures. "
                    "Environmental quenching in clusters is parameterised by the Peng+2010 "
                    "mass- and environment-quenching model. Ram-pressure stripping, "
                    "strangulation, and tidal interactions suppress star formation in "
                    "cluster satellites. The red fraction in clusters exceeds the field "
                    "by ~0.15 at fixed stellar mass."
                ),
                "metadata": {
                    "type": "explanation",
                    "analysis_type": "cluster_analysis",
                    "source_module": "default",
                },
            },
            {
                "content": (
                    "JWST NIRSpec spectroscopy covers 0.6–5.3 μm at R~1000 (PRISM/CLEAR). "
                    "Optimal 1D extraction follows Horne (1986). Spectral fitting with "
                    "Bagpipes (Carnall+2018) constrains stellar mass to ±0.15 dex, "
                    "redshift to σ_z~0.01, and dust attenuation via Calzetti+2000."
                ),
                "metadata": {
                    "type": "explanation",
                    "analysis_type": "jwst_spectroscopy",
                    "source_module": "default",
                },
            },
            {
                "content": (
                    "Bagpipes is a Bayesian SED fitting code (Carnall+2018, MNRAS 480). "
                    "It models stellar populations with flexible star formation histories, "
                    "nebular emission (Ferland+2013 CLOUDY), and dust attenuation. "
                    "Posterior sampling uses MultiNest nested sampling."
                ),
                "metadata": {
                    "type": "explanation",
                    "analysis_type": "sed_fitting",
                    "source_module": "default",
                },
            },
            {
                "content": (
                    "The stellar mass function of field galaxies is described by a "
                    "double-Schechter function (Baldry+2012, MNRAS 421). Cluster galaxies "
                    "show an enhanced characteristic mass log(M*)~10.81 and steeper "
                    "faint-end slope (Vulcani+2013, A&A 550). The Speagle+2014 (ApJS 214) "
                    "star-forming main sequence gives SFR as a function of mass and "
                    "cosmic time with 0.2 dex scatter."
                ),
                "metadata": {
                    "type": "explanation",
                    "analysis_type": "cluster_analysis",
                    "source_module": "default",
                },
            },
            {
                "content": (
                    "Emission line diagnostics in galaxy spectra: [OIII]5007/Hβ traces "
                    "ionization parameter and metallicity (Pettini & Pagel 2004, O3N2). "
                    "Hα/Hβ = 2.86 intrinsic (Case B, T=10^4 K, Osterbrock & Ferland 2006). "
                    "Calzetti+2000 dust law: E(B-V)_stars = 0.44 × E(B-V)_gas."
                ),
                "metadata": {
                    "type": "explanation",
                    "analysis_type": "jwst_spectroscopy",
                    "source_module": "default",
                },
            },
            {
                "content": (
                    "21cm power spectrum Δ²(k) peaks near k~0.1 Mpc⁻¹ during reionization "
                    "midpoint (Mesinger+2011, MNRAS 411). Amplitude is sensitive to the "
                    "ionized fraction x_HI and spin temperature T_S. Furlanetto+2006 "
                    "(Phys. Rep. 433) provides the standard brightness temperature formula: "
                    "δTb ≈ 27 x_HI (1 - T_CMB/T_S) sqrt((1+z)/10 · 0.15/Ω_m h²) mK."
                ),
                "metadata": {
                    "type": "explanation",
                    "analysis_type": "cosmic_evolution",
                    "source_module": "default",
                },
            },
        ]

        for entry in defaults:
            self.add_document(entry["content"], entry["metadata"])

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save_knowledge_base(self):
        """
        Persist the knowledge base to disk.

        BUG FIX: all data is passed through _to_serializable() before
        json.dump so there is no silent str() coercion of live Python objects
        (the original code used default=str which masked SessionStateProxy
        objects rather than preventing them from entering the store).
        """
        data = {
            "documents":   _to_serializable(self.documents),
            "vocabulary":  self.vocabulary,           # dict[str, int] — always safe
            "idf_scores":  self.idf_scores,           # dict[str, float] — always safe
        }

        try:
            with open(self.knowledge_base_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Could not save knowledge base JSON: %s", e)
            return

        vector_path = self.knowledge_base_path.replace(".json", "_vectors.pkl")
        try:
            with open(vector_path, "wb") as f:
                pickle.dump(self.document_vectors, f)
        except Exception as e:
            logger.error("Could not save knowledge base vectors: %s", e)

    def _load_knowledge_base(self):
        """
        Load the knowledge base from disk.

        BUG FIX: errors are logged rather than shown via st.warning() so this
        method is safe to call during @st.cache_resource construction (which
        happens outside a Streamlit render context).
        """
        if not os.path.exists(self.knowledge_base_path):
            self.populate_with_defaults()
            self.save_knowledge_base()
            return

        try:
            with open(self.knowledge_base_path, "r") as f:
                data = json.load(f)

            self.documents   = data.get("documents",   [])
            self.vocabulary  = data.get("vocabulary",  {})
            self.idf_scores  = data.get("idf_scores",  {})

            vector_path = self.knowledge_base_path.replace(".json", "_vectors.pkl")
            if os.path.exists(vector_path):
                with open(vector_path, "rb") as f:
                    self.document_vectors = pickle.load(f)
            else:
                self._rebuild_vectors()

        except Exception as e:
            logger.warning(
                "Could not load knowledge base (%s) — using defaults.", e
            )
            self.documents   = []
            self.vocabulary  = {}
            self.idf_scores  = {}
            self.document_vectors = []
            self.populate_with_defaults()

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the knowledge base.

        BUG FIX: returns sorted lists instead of sets so the output is
        deterministic and directly JSON-serialisable.
        """
        return {
            "total_documents": len(self.documents),
            "vocabulary_size": len(self.vocabulary),
            "analysis_types":  sorted(set(
                d["metadata"].get("analysis_type", "unknown")
                for d in self.documents
            )),
            "source_modules":  sorted(set(
                d["metadata"].get("source_module", "unknown")
                for d in self.documents
            )),
        }


# ---------------------------------------------------------------------------
# Streamlit-cached singleton
# ---------------------------------------------------------------------------

@st.cache_resource
def get_rag_system() -> AstroRAGSystem:
    """Get or create the global RAG system instance (one per server process)."""
    return AstroRAGSystem()