"""AI Integration for Astro-AI (OpenAI/OpenRouter Compatible)

Supports multiple AI providers:
- OpenAI (gpt-4o, gpt-3.5-turbo)
- OpenRouter (deepseek/deepseek-r1:free, anthropic/claude-3.5-sonnet, etc.)

Key features:
1. Uses the OpenAI-compatible client interface
2. Flexible API key and base URL configuration
3. Support for OpenRouter's free models like DeepSeek R1
4. Graceful fallback to simulation mode when no API key is available

Configuration options:
1. OpenAI: Set OPENAI_API_KEY
2. OpenRouter: Set OPENROUTER_API_KEY and optionally OPENROUTER_MODEL

Environment setup for OpenRouter:
    set OPENROUTER_API_KEY=sk-or-v1-...   (Windows PowerShell)
    set OPENROUTER_MODEL=deepseek/deepseek-r1:free   (Optional, defaults to deepseek-r1)

Example:
    # OpenRouter with DeepSeek R1 (free)
    assistant = OpenAIAssistant(provider="openrouter", model="deepseek/deepseek-r1:free")

    # OpenAI
    assistant = OpenAIAssistant(provider="openai", model="gpt-4o")

Security: ensure you NEVER commit real API keys to source control.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

import streamlit as st

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI, AsyncOpenAI, APIError
    _NEW_OPENAI_SDK = True
except Exception:
    import openai  # type: ignore
    _NEW_OPENAI_SDK = False

try:
    from utils.rag_system import get_rag_system
except ImportError:
    get_rag_system = None


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _to_serializable(obj: Any, _depth: int = 0) -> Any:
    """
    Recursively convert an object to a JSON-serialisable form.

    Rules:
    - Plain scalars (int, float, bool, str, None) → pass through
    - list / tuple → recurse each element
    - dict → recurse each value (skip keys whose values can't be serialised)
    - numpy scalar / array → convert via .tolist()
    - Everything else → str(obj)  (last resort; never raises)

    A depth cap of 6 prevents runaway recursion on deeply nested structures.
    """
    if _depth > 6:
        return "<truncated>"

    # numpy types — check before generic int/float because np.float64 IS a float
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            if obj.size > 100:          # don't embed large arrays in prompts
                return f"<array shape={obj.shape}>"
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
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

    # Streamlit SessionStateProxy and similar objects — never pass their
    # internals to the AI or RAG; return an empty sentinel instead.
    type_name = type(obj).__name__
    if "SessionState" in type_name or "Proxy" in type_name:
        return "<session_state — excluded>"

    # pandas DataFrame
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return f"<DataFrame rows={len(obj)} cols={list(obj.columns)}>"
    except ImportError:
        pass

    # matplotlib Figure / bytes blobs
    if hasattr(obj, "savefig") or isinstance(obj, (bytes, bytearray)):
        return "<binary — excluded>"

    # Generic fallback
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


def _safe_json(obj: Any, indent: int = 2) -> str:
    """Serialise *obj* to a JSON string, never raising."""
    try:
        return json.dumps(_to_serializable(obj), indent=indent)
    except Exception as e:
        return f'{{"error": "serialisation failed: {e}"}}'


class OpenAIAssistant:
    """
    AI-powered assistant for astronomical data analysis and scientific reporting.

    Supports OpenAI and OpenRouter providers for flexible AI model access.
    Provides natural language insights, scientific interpretation, and
    automated report generation for galaxy evolution studies.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: str = "auto",
        base_url: Optional[str] = None,
        use_responses_api: bool = False,
    ):
        self.provider = provider
        self.use_responses_api = use_responses_api

        # Initialise RAG system
        self.rag_system = None
        if get_rag_system:
            try:
                self.rag_system = get_rag_system()
            except Exception as e:
                logger.warning("RAG system initialisation failed: %s", e)

        self._configure_provider(api_key, model, base_url)

    # -----------------------------------------------------------------------
    # Provider configuration
    # -----------------------------------------------------------------------

    def _configure_provider(
        self,
        api_key: Optional[str],
        model: Optional[str],
        base_url: Optional[str],
    ):
        if self.provider == "auto":
            if _secret_or_env("OPENROUTER_API_KEY"):
                self.provider = "openrouter"
            elif _secret_or_env("OPENAI_API_KEY"):
                self.provider = "openai"
            else:
                self.provider = "openrouter"

        if api_key:
            self.api_key = api_key
        elif self.provider == "openrouter":
            self.api_key = _secret_or_env("OPENROUTER_API_KEY") or ""
        else:
            self.api_key = _secret_or_env("OPENAI_API_KEY") or ""

        if model:
            self.model = model
        elif self.provider == "openrouter":
            self.model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1:free")
        else:
            self.model = "gpt-4o"

        if base_url:
            self.base_url = base_url
        elif self.provider == "openrouter":
            self.base_url = "https://openrouter.ai/api/v1"
        else:
            self.base_url = None

        self.fallback_mode = False
        if not self.api_key:
            self.fallback_mode = True
            logger.warning("No API key found — AI features will use simulation mode.")
            self.client = None
            return

        try:
            if _NEW_OPENAI_SDK:
                kwargs: dict = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self.client = OpenAI(**kwargs)
            else:
                import openai as _openai  # type: ignore
                _openai.api_key = self.api_key
                if self.base_url:
                    _openai.api_base = self.base_url
                self.client = _openai
        except Exception as e:
            logger.warning("AI client initialisation failed: %s", e)
            self.fallback_mode = True
            self.client = None
            return

        self.system_prompt = (
            "You are an expert astrophysicist and data scientist specialising in "
            "galaxy evolution, 21cm cosmology, and observational astronomy. "
            "Your role is to provide scientific insights, interpret results, and "
            "generate comprehensive reports for astronomical data analysis.\n\n"
            "Key areas of expertise:\n"
            "- Galaxy formation and evolution\n"
            "- 21cm intensity mapping and reionization\n"
            "- Galaxy cluster environments and quenching\n"
            "- JWST observations and spectroscopy\n"
            "- Stellar population synthesis and SED fitting\n"
            "- Statistical analysis of astronomical data\n\n"
            "Always provide scientifically accurate, well-referenced explanations "
            "appropriate for research publications or technical reports."
        )

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------

    def generate_insight(self, query: str, analysis_context: Dict[str, Any]) -> str:
        """
        Generate AI-powered scientific insights from a user query and analysis context.

        Parameters
        ----------
        query : str
            The user's question or request.
        analysis_context : dict
            Plain serialisable dict produced by app._extract_analysis_context().
            Must NOT be a SessionStateProxy — use _extract_analysis_context() in
            app.py before calling this method.

        Returns
        -------
        str
            AI-generated scientific insights.
        """
        # Infer analysis type from context keys for RAG storage
        analysis_type = _infer_analysis_type(analysis_context)

        # Store in RAG — safe because analysis_context is already sanitised
        if self.rag_system:
            self._store_analysis_results(analysis_context, analysis_type)

        if self.fallback_mode:
            return self._generate_fallback_insight(analysis_context, analysis_type)

        try:
            rag_context = ""
            if self.rag_system:
                rag_context = self.rag_system.generate_context(query, top_k=3)

            prompt = (
                f"{rag_context}\n\n"
                f"User question: {query}\n\n"
                f"Current analysis results:\n{_safe_json(analysis_context)}\n\n"
                "Provide a scientifically rigorous answer. "
                "Use the retrieved context where relevant."
            )

            return self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=1500,
                temperature=0.7,
            )

        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                return self._generate_fallback_insight(analysis_context, analysis_type)
            logger.error("generate_insight error: %s", e)
            return "AI analysis temporarily unavailable."

    def generate_template_analysis(
        self, template_name: str, analysis_context: Dict[str, Any]
    ) -> str:
        """
        Generate analysis based on predefined templates.

        Parameters
        ----------
        template_name : str
            Name of the analysis template.
        analysis_context : dict
            Plain serialisable dict produced by app._extract_analysis_context().
            Must NOT be a SessionStateProxy.
        """
        if self.fallback_mode:
            return self._generate_fallback_template_analysis(template_name)

        try:
            # Serialise context safely — never pass raw session_state
            context_json = _safe_json(analysis_context)

            template_prompts = {
                "Cosmic evolution timeline analysis": (
                    "Analyse the cosmic evolution timeline from our 21cm simulations. "
                    "Focus on key epochs: reionization, first galaxies, structure formation. "
                    "Provide a chronological narrative of early universe evolution.\n\n"
                    f"Available analysis results:\n{context_json}"
                ),
                "Environmental effects on galaxy properties": (
                    "Examine environmental effects on galaxy evolution in cluster settings. "
                    "Compare cluster vs field galaxies, discuss quenching mechanisms. "
                    "Analyse colour-magnitude relations and stellar population gradients.\n\n"
                    f"Available analysis results:\n{context_json}"
                ),
                "Spectroscopic vs photometric constraints": (
                    "Perform detailed analysis of JWST spectroscopic observations. "
                    "Focus on emission lines, stellar populations, chemical evolution. "
                    "Discuss implications for galaxy formation and evolution models.\n\n"
                    f"Available analysis results:\n{context_json}"
                ),
                "High-redshift galaxy formation insights": (
                    "Analyse high-redshift galaxy formation and early universe physics. "
                    "Connect 21cm signatures with galaxy observations. "
                    "Discuss implications for ΛCDM cosmology and galaxy formation models.\n\n"
                    f"Available analysis results:\n{context_json}"
                ),
            }

            prompt = template_prompts.get(
                template_name,
                f"Analyse the astronomical data for: {template_name}\n\n"
                f"Available results:\n{context_json}",
            )

            return self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=1500,
                temperature=0.7,
            )

        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                st.warning("🚦 Rate limit reached — switching to simulation mode.")
            else:
                st.warning(f"⚠️ AI service unavailable: {e}")
            return self._generate_fallback_template_analysis(template_name)

    def generate_comparative_analysis(self, results_summary: Dict[str, Any]) -> str:
        if self.fallback_mode:
            return _FALLBACK_COMPARATIVE

        try:
            rag_context = ""
            if self.rag_system:
                rag_context = self.rag_system.generate_context(
                    "comparative analysis cross-module galaxy evolution", top_k=5)

            prompt = (
                f"{rag_context}\n\n"
                "Provide a comprehensive comparative analysis across these "
                "multi-wavelength and multi-epoch astronomical results:\n\n"
                f"{_safe_json(results_summary)}\n\n"
                "Focus on: (1) connecting 21cm cosmic evolution with galaxy observations, "
                "(2) environmental effects, (3) JWST insights, (4) synthesis across "
                "cosmic time, (5) implications for galaxy formation models, "
                "(6) future observational priorities."
            )

            return self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=2000,
                temperature=0.7,
            )
        except Exception as e:
            return f"Comparative analysis temporarily unavailable: {e}"

    def generate_report_section(self, section_type: str, data: Dict[str, Any]) -> str:
        if self.fallback_mode:
            return self._generate_fallback_report_section(section_type)

        prompts = {
            "introduction": (
                "Write a scientific introduction for a galaxy evolution study. "
                "Include relevant background, motivation, and objectives.\n\n"
                f"Study context:\n{_safe_json(data)}"
            ),
            "methods": (
                "Write a methods section describing the analysis techniques used. "
                "Include technical details appropriate for peer review.\n\n"
                f"Methods used:\n{_safe_json(data)}"
            ),
            "results": (
                "Write a results section summarising key findings. "
                "Present results objectively with quantitative details.\n\n"
                f"Results:\n{_safe_json(data)}"
            ),
            "discussion": (
                "Write a discussion section interpreting results and implications. "
                "Include scientific interpretation, limitations, and future work.\n\n"
                f"Findings:\n{_safe_json(data)}"
            ),
        }
        prompt = prompts.get(section_type,
                             f"Generate {section_type} content for:\n{_safe_json(data)}")
        try:
            return self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
            )
        except Exception as e:
            return f"Report section generation temporarily unavailable: {e}"

    def suggest_next_steps(self, analysis_results: Dict[str, Any]) -> List[str]:
        if self.fallback_mode:
            return _FALLBACK_NEXT_STEPS

        try:
            prompt = (
                "Based on these analysis results, suggest 5-8 specific, actionable "
                "next steps for research. Format as a numbered list.\n\n"
                f"{_safe_json(analysis_results)}"
            )
            content = self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=800,
                temperature=0.8,
            )
            suggestions = [
                s.strip() for s in content.split("\n")
                if s.strip() and (s.strip()[0].isdigit() or s.strip().startswith("-"))
            ]
            return suggestions[:8]
        except Exception as e:
            return [f"Next steps analysis temporarily unavailable: {e}"]

    def check_api_status(self) -> bool:
        if self.fallback_mode:
            return False
        try:
            self._chat([{"role": "user", "content": "Ping"}],
                       max_tokens=5, temperature=0.0)
            return True
        except Exception:
            return False

    def get_rag_status(self) -> Dict[str, Any]:
        if not self.rag_system:
            return {"enabled": False, "reason": "RAG system not available", "stats": {}}
        try:
            return {"enabled": True, "reason": "RAG system operational",
                    "stats": self.rag_system.get_stats()}
        except Exception as e:
            return {"enabled": False, "reason": f"RAG system error: {e}", "stats": {}}

    def query_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.rag_system:
            return []
        try:
            results = self.rag_system.retrieve(query, top_k)
            return [
                {"content": doc["content"], "metadata": doc["metadata"],
                 "similarity": score}
                for doc, score in results
            ]
        except Exception as e:
            logger.warning("Knowledge base query failed: %s", e)
            return []

    # -----------------------------------------------------------------------
    # RAG storage — hardened against unhashable / non-serialisable input
    # -----------------------------------------------------------------------

    def _store_analysis_results(
        self, data_summary: Dict[str, Any], analysis_type: str
    ):
        """
        Store analysis results in the RAG knowledge base.

        data_summary MUST be a plain dict of serialisable scalars — never a
        SessionStateProxy.  If anything inside it is still non-serialisable
        _to_serializable() will convert it to a safe string representation
        before it reaches the RAG layer.
        """
        if not self.rag_system:
            return

        try:
            # Convert everything to serialisable form first
            safe_summary = _to_serializable(data_summary)

            # Build a human-readable interpretation string
            parts = []
            if isinstance(safe_summary, dict):
                for k, v in safe_summary.items():
                    if v is not None and not str(v).startswith("<"):
                        parts.append(f"{k}: {v}")
            interpretation = (
                "; ".join(parts)
                if parts
                else f"Analysis results for {analysis_type}"
            )

            source_map = {
                "cosmic_evolution": "cos_evo",
                "cluster_analysis": "cluster_analyzer",
                "jwst_spectroscopy": "jwst_analyzer",
            }

            self.rag_system.add_scientific_result(
                analysis_type=analysis_type,
                results=safe_summary,          # guaranteed serialisable
                interpretation=interpretation,
                source_module=source_map.get(analysis_type, "unknown"),
            )
            self.rag_system.save_knowledge_base()

        except Exception as e:
            # Log but never crash the UI over a RAG storage failure
            logger.warning("Failed to store analysis results in RAG system: %s", e)

    # -----------------------------------------------------------------------
    # Low-level chat abstraction
    # -----------------------------------------------------------------------

    def _chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        if _NEW_OPENAI_SDK:
            try:
                if self.use_responses_api:
                    instructions = None
                    user_inputs: List[Dict[str, Any]] = []
                    for m in messages:
                        if m["role"] == "system" and instructions is None:
                            instructions = m["content"]
                        else:
                            user_inputs.append({
                                "role": m["role"],
                                "content": [{"type": "input_text", "text": m["content"]}],
                            })
                    resp = self.client.responses.create(
                        model=self.model,
                        instructions=instructions,
                        input=user_inputs,
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return getattr(resp, "output_text", "").strip()
                else:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return completion.choices[0].message.content  # type: ignore
            except APIError as e:  # type: ignore[name-defined]
                raise RuntimeError(f"OpenAI API error: {e}") from e
            except Exception as e:
                raise RuntimeError(f"OpenAI request failed: {e}") from e
        else:
            try:
                completion = self.client.ChatCompletion.create(  # type: ignore
                    model=self.model if self.model.startswith("gpt-") else "gpt-4",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return completion.choices[0].message.content  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Legacy OpenAI request failed: {e}") from e

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ):
        if not _NEW_OPENAI_SDK or self.use_responses_api:
            yield self._chat(messages, temperature=temperature)
            return
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            for event in stream:  # type: ignore
                delta = getattr(event.choices[0].delta, "content", None)
                if delta:
                    yield delta
        except Exception as e:
            yield f"[Streaming failed: {e}]"

    # -----------------------------------------------------------------------
    # Fallback responses (no API key / rate limited)
    # -----------------------------------------------------------------------

    def _generate_fallback_insight(
        self, data_summary: Dict[str, Any], analysis_type: str
    ) -> str:
        return _FALLBACK_INSIGHTS.get(analysis_type, _FALLBACK_INSIGHTS["general"])

    def _generate_fallback_report_section(self, section_type: str) -> str:
        return _FALLBACK_REPORT_SECTIONS.get(
            section_type, _FALLBACK_REPORT_SECTIONS["introduction"])

    def _generate_fallback_template_analysis(self, template_name: str) -> str:
        return _FALLBACK_TEMPLATES.get(
            template_name,
            f"**{template_name}** — enable AI integration for detailed analysis.")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _secret_or_env(key: str) -> str:
    """Return value from Streamlit secrets or environment, or empty string."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, "")


def _infer_analysis_type(context: Dict[str, Any]) -> str:
    """Infer the dominant analysis type from context keys."""
    if "cosmic_evolution" in context:
        return "cosmic_evolution"
    if "cluster_analysis" in context:
        return "cluster_analysis"
    if "spectral_fit" in context:
        return "jwst_spectroscopy"
    return "general"


# ---------------------------------------------------------------------------
# Fallback text constants — kept out of the class to reduce clutter
# ---------------------------------------------------------------------------

_FALLBACK_COMPARATIVE = """
**AI Simulation Mode — Comparative Analysis**

🔬 **Cross-Module Insights**

📡 **21cm ↔ Galaxy Observations**
- Reionization signatures correlate with galaxy formation efficiency
- Power spectrum features match observed galaxy clustering

🌌 **Cluster Environment ↔ Individual Galaxies**
- Environmental quenching mechanisms confirmed across scales
- Stellar population properties show clear environmental dependence

🔭 **JWST Spectroscopy ↔ Broad-band Photometry**
- High-resolution spectra validate SED fitting assumptions
- Emission line diagnostics refine stellar population models

*Note: This is simulation mode. Set an API key in Streamlit secrets to enable full AI analysis.*
"""

_FALLBACK_NEXT_STEPS = [
    "🔬 Expand parameter space in 21cm simulations for broader redshift coverage",
    "📊 Increase galaxy sample size for improved statistical significance",
    "🌌 Include additional cluster environments (groups, filaments) for comparison",
    "🔭 Extend JWST spectroscopic analysis to include NIRCam imaging",
    "📈 Implement machine learning techniques for pattern recognition in SED fitting",
    "🎯 Focus on specific emission line diagnostics (metallicity, ionization parameter)",
    "🔄 Cross-validate results with independent observational datasets (COSMOS, CANDELS)",
    "📝 Prepare findings for submission to a peer-reviewed journal",
]

_FALLBACK_INSIGHTS = {
    "cosmic_evolution": """
**AI Simulation Mode — Cosmic Evolution Analysis**

🔬 **Key Scientific Findings**
- Reionization signatures detected in brightness temperature evolution
- Power spectrum analysis reveals clustering patterns consistent with CDM model
- Galaxy formation efficiency shows expected redshift dependence
- Ionization fraction evolution follows Pritchard & Loeb (2012) theoretical predictions

📊 **Statistical Insights**
- Strong correlation between halo mass and star formation activity
- Power spectrum peak indicates characteristic scale of first galaxies (~10 Mpc)
- Temperature fluctuations consistent with Wouthuysen-Field coupling at z~17-20

🚀 **Physical Interpretation**
The absorption trough in the global signal constrains the spin temperature coupling
epoch. The power spectrum amplitude at k~0.1 Mpc⁻¹ is sensitive to the ionization
state of the IGM and can be compared with future SKA observations.

*Note: Simulation mode — set OPENROUTER_API_KEY for full AI analysis.*
""",
    "cluster_analysis": """
**AI Simulation Mode — Galaxy Cluster Analysis**

🌌 **Galaxy Population**
- Clear red sequence / blue cloud bimodality (Bell+2004 colour cut)
- Mass-metallicity relation follows Tremonti+2004 trend
- Star formation quenching enhanced in dense environments (Peng+2010)
- Colour-magnitude diagram shows environmental evolutionary sequences

📈 **Environmental Effects**
- Cluster red fraction exceeds field by ~0.15 (consistent with Peng+2010 §4)
- Stellar mass function shifted to higher masses in clusters (Vulcani+2013)
- SFRs of cluster star-forming galaxies consistent with Speagle+2014 SFMS

🔍 **Stellar Populations**
- Age-metallicity degeneracy partially broken by g-r colour information
- Stellar mass function consistent with hierarchical assembly

*Note: Simulation mode — set OPENROUTER_API_KEY for full AI analysis.*
""",
    "jwst_spectroscopy": """
**AI Simulation Mode — JWST Spectroscopic Analysis**

🔬 **Spectral Features**
- Emission lines at correct observed wavelengths for input redshift
- [OIII]/Hβ ratio consistent with Pettini & Pagel (2004) O3N2 calibration
- Balmer decrement Hα/Hβ = 2.86 (Case B, Osterbrock & Ferland 2006)
- Dust attenuation applied via Calzetti+2000 k(λ) curve

⭐ **Physical Properties**
- Star formation rate from Hα luminosity (Kennicutt 1998 calibration)
- Metallicity from strong-line diagnostics
- Stellar mass from continuum SED fitting
- Age constraints from stellar absorption features

🌠 **Galaxy Evolution**
- Spectroscopic redshift precision σ_z ~ 0.001 (vs. photometric σ_z ~ 0.05)
- Chemical enrichment history traceable through abundance ratios
- Kinematic structure accessible via line widths

*Note: Simulation mode — set OPENROUTER_API_KEY for full AI analysis.*
""",
    "general": """
**AI Simulation Mode — General Analysis**

The analysis results have been processed. Key findings depend on which modules
have been run. Navigate to individual modules for detailed interpretation.

*Note: Simulation mode — set OPENROUTER_API_KEY for full AI analysis.*
""",
}

_FALLBACK_REPORT_SECTIONS = {
    "introduction": """
## Introduction

This study employs a multi-scale approach to investigate galaxy evolution across
cosmic time, combining 21cm reionization simulations, galaxy cluster environment
analysis, and JWST NIRSpec spectroscopy. By integrating these complementary
techniques, we probe galaxy formation from the Epoch of Reionization (z~15)
through the peak of cosmic star formation (z~2) to the present day.

The 21cm hyperfine transition of neutral hydrogen provides a unique tracer of
the intergalactic medium during reionization (Pritchard & Loeb 2012). Galaxy
cluster environments offer laboratories for studying environmental quenching
mechanisms (Peng+2010). JWST spectroscopy enables detailed stellar population
analysis at high redshifts (Carnall+2019).
""",
    "methods": """
## Methodology

**21cm Cosmological Simulations**
Reionization simulations use the 21cmFAST semi-numerical code (Mesinger+2011)
with Planck 2018 cosmological parameters. Brightness temperature cubes are
generated across 6 ≤ z ≤ 15. Power spectra are computed via spherical averaging
in Fourier space following Furlanetto+2006.

**Galaxy Cluster Analysis**
Galaxy catalogs are analysed using a double-Schechter stellar mass function
(Baldry+2012). Red fractions are quantified using the Peng+2010 mass- and
environment-quenching framework. SED fitting uses the Bagpipes Bayesian
framework (Carnall+2018) with a Calzetti+2000 dust law.

**JWST Spectroscopic Pipeline**
NIRSpec data are reduced through the standard STScI three-stage pipeline.
Optimal 1D extraction follows Horne (1986). Spectral fitting uses Bagpipes
with nebular emission (Ferland+2013) and Calzetti+2000 attenuation.
""",
    "results": """
## Results

**Cosmic Evolution**
The global 21cm signal shows the characteristic absorption trough at z~17-20
(δTb ~ -120 mK), consistent with Wouthuysen-Field coupling, followed by X-ray
heating and eventual emission as reionization progresses. The power spectrum
peaks near k ~ 0.1 Mpc⁻¹ during the midpoint of reionization (z ~ 8).

**Cluster Environment Effects**
Cluster galaxies show a red fraction ~0.15 higher than field galaxies at fixed
stellar mass, consistent with Peng+2010 environmental quenching predictions.
The cluster stellar mass function is enhanced at log(M*/M☉) > 10.8 relative
to the field (Vulcani+2013).

**JWST Spectroscopy**
Spectral fitting recovers stellar masses to ±0.15 dex and redshifts to
σ_z ~ 0.01, consistent with Carnall+2019 benchmark tests. Emission line
ratios are consistent with star-forming galaxy calibrations (Kewley+2001).
""",
    "discussion": """
## Discussion

The integrated analysis reveals coherent connections across cosmic time.
Reionization timing from 21cm simulations constrains the epoch at which
cluster progenitors first assembled, providing context for the environmental
quenching signatures observed at z ~ 1-2. JWST spectroscopy confirms the
stellar population properties inferred from photometric SED fitting.

**Limitations**
Mock data are grounded in published empirical relations but cannot capture
the full complexity of observed galaxy populations. Real observations will
introduce scatter from physical processes not included in the analytic models.

**Future Work**
Extended JWST surveys (JADES, CEERS) will provide larger samples. SKA
observations will directly measure the 21cm power spectrum. Improved
semi-analytic models will enable tighter cross-module comparisons.
""",
}

_FALLBACK_TEMPLATES = {
    "Cosmic evolution timeline analysis": """
## Cosmic Evolution Timeline (Simulation Mode)

**z > 20 — Pre-stellar era**
Dark matter halos collapse; primordial gas begins cooling in minihalos.

**z = 15–20 — Cosmic Dawn**
Population III stars form; Ly-α photons couple spin temperature to gas
temperature (Wouthuysen-Field effect). Global 21cm signal transitions
from zero to deep absorption (δTb ~ −120 mK, Cohen+2017).

**z = 10–15 — First galaxies**
Population II star formation begins; first metal enrichment; reionization
photons start escaping into the IGM.

**z = 6–10 — Epoch of Reionization**
Ionized bubbles percolate; 21cm power spectrum evolves rapidly;
Ly-α forest opacity increases (Fan+2006). Reionization completes at z ~ 6.

**z < 6 — Post-reionization**
Modern galaxy populations assembled; cluster environments begin quenching
star formation; JWST probes individual galaxy stellar populations.

*Simulation mode — enable AI for a personalised timeline based on your results.*
""",
    "Environmental effects on galaxy properties": """
## Environmental Effects on Galaxy Properties (Simulation Mode)

**Mass quenching** operates at all densities: above log(M*/M☉) ~ 10.5 the
red fraction rises steeply regardless of environment (Peng+2010).

**Environment quenching** adds ~0.15 to the red fraction at fixed mass for
cluster galaxies, driven by ram-pressure stripping, strangulation, and
tidal interactions.

**Stellar mass function**: cluster SMF is enhanced at the massive end
relative to field (Vulcani+2013), reflecting the earlier assembly of
massive galaxies in overdense regions.

**Star-forming main sequence**: cluster star-forming galaxies follow the
same Speagle+2014 SFMS as field galaxies at fixed mass and redshift,
but the quenched fraction is higher.

*Simulation mode — enable AI for analysis based on your specific results.*
""",
    "Spectroscopic vs photometric constraints": """
## Spectroscopic vs Photometric Constraints (Simulation Mode)

**Redshift precision**: spectroscopic z has σ ~ 0.001 vs photometric
σ_z ~ 0.03–0.05, removing catastrophic outliers that bias cluster membership.

**Stellar mass**: both methods agree to ~0.15 dex when the SED coverage
is good (Carnall+2019). Spectroscopy breaks the age-metallicity degeneracy.

**Star formation rate**: Hα-based SFR (Kennicutt 1998) is instantaneous
(< 10 Myr), while SED-based SFR averages over ~100 Myr. Both are needed
to characterise recent star formation history.

**Dust attenuation**: Balmer decrement (Hα/Hβ = 2.86 intrinsic, Case B)
provides a direct dust measurement independent of SED assumptions.

**Metallicity**: strong-line diagnostics (R23, O3N2, N2) give gas-phase
metallicity inaccessible from broad-band photometry alone.

*Simulation mode — enable AI for analysis based on your specific results.*
""",
    "High-redshift galaxy formation insights": """
## High-Redshift Galaxy Formation Insights (Simulation Mode)

**JWST has revealed** galaxies at z > 10 with stellar masses and UV
luminosities that challenge standard galaxy formation models, suggesting
either higher star formation efficiencies or a top-heavy IMF at early times.

**21cm observations** will directly constrain the ionizing photon budget
from these early galaxies, connecting JWST photometry with reionization.

**Stellar population archaeology**: rest-frame optical spectra from NIRSpec
provide age, metallicity, and SFH constraints that test semi-analytic models
and hydrodynamic simulations (FLARES, IllustrisTNG, FIRE).

**Chemical evolution**: [OIII]/Hβ ratios at z > 3 are systematically higher
than local galaxies (Kewley+2013), indicating harder ionizing spectra and/or
lower metallicities in the early universe.

*Simulation mode — enable AI for analysis based on your specific results.*
""",
}