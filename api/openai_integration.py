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
from typing import Dict, Any, List, Optional, Union

import streamlit as st

try:
    # New >=1.0 SDK style
    from openai import OpenAI, AsyncOpenAI, APIError
    _NEW_OPENAI_SDK = True
except Exception:  # pragma: no cover - fallback if old package version
    import openai  # type: ignore
    _NEW_OPENAI_SDK = False

# Import RAG system
try:
    from utils.rag_system import get_rag_system
except ImportError:
    get_rag_system = None

class OpenAIAssistant:
    """
    AI-powered assistant for astronomical data analysis and scientific reporting.
    
    Supports OpenAI and OpenRouter providers for flexible AI model access.
    Provides natural language insights, scientific interpretation, and 
    automated report generation for galaxy evolution studies.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, 
                 provider: str = "auto", base_url: Optional[str] = None, use_responses_api: bool = False):
        """
        Initialize AI assistant with flexible provider support.
        
        Parameters:
        -----------
        api_key : str, optional
            API key. If not provided, will auto-detect based on provider
        model : str, optional
            Model name. Defaults based on provider
        provider : str
            AI provider: "openai", "openrouter", or "auto"
        base_url : str, optional
            Custom base URL for API endpoints
        """
        self.provider = provider
        self.use_responses_api = use_responses_api
        
        # Initialize RAG system
        self.rag_system = None
        if get_rag_system:
            try:
                self.rag_system = get_rag_system()
                st.info("🔍 RAG system enabled for enhanced context retrieval")
            except Exception as e:
                st.warning(f"RAG system initialization failed: {e}")
        
        # Auto-detect provider and configure defaults
        self._configure_provider(api_key, model, base_url)

    def _configure_provider(self, api_key: Optional[str], model: Optional[str], base_url: Optional[str]):
        """Configure API provider settings"""
        
        # Auto-detect provider if not specified
        if self.provider == "auto":
            if 'OPENROUTER_API_KEY' in st.secrets or os.getenv('OPENROUTER_API_KEY'):
                self.provider = "openrouter"
            elif 'OPENAI_API_KEY' in st.secrets or os.getenv('OPENAI_API_KEY'):
                self.provider = "openai"
            else:
                self.provider = "openrouter"  # Default to OpenRouter
        
        # Configure API key based on provider
        if api_key:
            self.api_key = api_key
        elif self.provider == "openrouter":
            if 'OPENROUTER_API_KEY' in st.secrets:
                self.api_key = st.secrets['OPENROUTER_API_KEY']
            else:
                self.api_key = os.getenv('OPENROUTER_API_KEY', '')
        else:  # openai
            if 'OPENAI_API_KEY' in st.secrets:
                self.api_key = st.secrets['OPENAI_API_KEY']
            else:
                self.api_key = os.getenv('OPENAI_API_KEY', '')
        
        # Configure model based on provider
        if model:
            self.model = model
        elif self.provider == "openrouter":
            self.model = os.getenv('OPENROUTER_MODEL', 'deepseek/deepseek-r1:free')
        else:  # openai
            self.model = "gpt-4o"
        
        # Configure base URL
        if base_url:
            self.base_url = base_url
        elif self.provider == "openrouter":
            self.base_url = "https://openrouter.ai/api/v1"
        else:
            self.base_url = None  # Use OpenAI default

        # Set up fallback mode if no API key is available
        self.fallback_mode = False
        if not self.api_key:
            self.fallback_mode = True
            if self.provider == "openrouter":
                st.warning("⚠️ OpenRouter API key not configured. AI features will use simulation mode. Set OPENROUTER_API_KEY in Streamlit secrets to enable AI analysis with DeepSeek R1.")
            else:
                st.warning("⚠️ OpenAI API key not configured. AI features will use simulation mode. Set OPENAI_API_KEY in Streamlit secrets to enable AI analysis.")
            self.client = None
            return

        # Initialize the client
        try:
            if _NEW_OPENAI_SDK:
                client_kwargs = {"api_key": self.api_key}
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                self.client = OpenAI(**client_kwargs)
            else:  # legacy fallback
                import openai  # type: ignore
                openai.api_key = self.api_key
                if self.base_url:
                    openai.api_base = self.base_url
                self.client = openai  # type: ignore
        except Exception as e:
            st.warning(f"⚠️ AI client initialization failed: {e}. Using simulation mode.")
            self.fallback_mode = True
            self.client = None
            return
        
        # Success message
        if self.provider == "openrouter":
            st.success(f"✅ OpenRouter AI enabled with model: {self.model}")
        else:
            st.success(f"✅ OpenAI enabled with model: {self.model}")
        
        # System prompt for astronomical context
        self.system_prompt = """
        You are an expert astrophysicist and data scientist specializing in galaxy evolution, 
        21cm cosmology, and observational astronomy. Your role is to provide scientific 
        insights, interpret results, and generate comprehensive reports for astronomical 
        data analysis.
        
        Key areas of expertise:
        - Galaxy formation and evolution
        - 21cm intensity mapping and reionization
        - Galaxy cluster environments and quenching
        - JWST observations and spectroscopy
        - Stellar population synthesis and SED fitting
        - Statistical analysis of astronomical data
        
        Always provide scientifically accurate, well-referenced explanations that would 
        be appropriate for research publications or technical reports.
        """
    
    def generate_insight(self, data_summary: Dict[str, Any], 
                        analysis_type: str = "general") -> str:
        """
        Generate AI-powered scientific insights from analysis results using RAG.
        
        Parameters:
        -----------
        data_summary : dict
            Summary of analysis results including key metrics and findings
        analysis_type : str
            Type of analysis ('cosmic_evolution', 'cluster_analysis', 'jwst_spectroscopy')
        
        Returns:
        --------
        str
            AI-generated scientific insights and interpretation with retrieved context
        """
        # Store results in RAG knowledge base for future retrieval
        if self.rag_system:
            self._store_analysis_results(data_summary, analysis_type)
        
        # Fallback mode when no API key is available
        if self.fallback_mode:
            return self._generate_fallback_insight(data_summary, analysis_type)
        
        try:
            # Create query for RAG retrieval
            query_text = f"{analysis_type} analysis results interpretation"
            
            # Retrieve relevant context using RAG
            rag_context = ""
            if self.rag_system:
                rag_context = self.rag_system.generate_context(query_text, top_k=3)
            
            # Create analysis-specific prompts with RAG context
            if analysis_type == "cosmic_evolution":
                context = f"""
                {rag_context}
                
                Now analyze these NEW 21cm cosmological simulation results:
                {json.dumps(data_summary, indent=2)}
                
                Provide scientific interpretation focusing on:
                1. Reionization physics and timing
                2. Power spectrum implications
                3. Brightness temperature evolution
                4. Connection to galaxy formation
                
                Use the retrieved context to provide deeper insights and comparisons.
                """
            
            elif analysis_type == "cluster_analysis":
                context = f"""
                {rag_context}
                
                Now analyze these NEW galaxy cluster environment results:
                {json.dumps(data_summary, indent=2)}
                
                Provide scientific interpretation focusing on:
                1. Environmental quenching mechanisms
                2. Stellar mass assembly differences
                3. Red fraction evolution
                4. Comparison with observations
                
                Use the retrieved context to provide deeper insights and comparisons.
                """
            
            elif analysis_type == "jwst_spectroscopy":
                context = f"""
                {rag_context}
                
                Now analyze these NEW JWST spectroscopic results:
                {json.dumps(data_summary, indent=2)}
                
                Provide scientific interpretation focusing on:
                1. Spectral line diagnostics
                2. Stellar population properties
                3. Galaxy formation history
                4. Comparison with previous surveys
                
                Use the retrieved context to provide deeper insights and comparisons.
                """
            
            else:
                context = f"""
                {rag_context}
                
                Now analyze these NEW astronomical analysis results:
                {json.dumps(data_summary, indent=2)}
                
                Provide general scientific interpretation and key findings.
                Use the retrieved context to provide deeper insights and comparisons.
                """
            
            # Generate response using OpenAI with RAG context
            return self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt + "\n\nYou have access to a knowledge base of previous analyses. Use this context to provide more informed interpretations."},
                    {"role": "user", "content": context},
                ],
                max_tokens=1500,
                temperature=0.7,
            )
            
        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "rate-limited" in error_message.lower():
                return self._generate_fallback_insight(data_summary, analysis_type)
            else:
                return f"AI analysis temporarily unavailable. Using simulation mode."
    
    def _store_analysis_results(self, data_summary: Dict[str, Any], analysis_type: str):
        """Store analysis results in the RAG knowledge base for future retrieval."""
        if not self.rag_system:
            return
        
        try:
            # Create interpretation text for the analysis
            interpretation_parts = []
            
            if 'summary' in data_summary:
                interpretation_parts.append(data_summary['summary'])
            
            if 'key_findings' in data_summary:
                findings = data_summary['key_findings']
                if isinstance(findings, list):
                    interpretation_parts.append(f"Key findings: {'; '.join(findings)}")
                else:
                    interpretation_parts.append(f"Key findings: {findings}")
            
            if 'parameters' in data_summary:
                params = data_summary['parameters']
                if isinstance(params, dict):
                    param_str = ', '.join(f"{k}={v}" for k, v in params.items())
                    interpretation_parts.append(f"Analysis parameters: {param_str}")
            
            interpretation = ' '.join(interpretation_parts) if interpretation_parts else f"Analysis results for {analysis_type}"
            
            # Determine source module
            source_mapping = {
                'cosmic_evolution': 'cos_evo',
                'cluster_analysis': 'cluster_analyzer', 
                'jwst_spectroscopy': 'jwst_analyzer'
            }
            source_module = source_mapping.get(analysis_type, 'unknown')
            
            # Store in RAG system
            self.rag_system.add_scientific_result(
                analysis_type=analysis_type,
                results=data_summary,
                interpretation=interpretation,
                source_module=source_module
            )
            
            # Save the updated knowledge base
            self.rag_system.save_knowledge_base()
            
        except Exception as e:
            st.warning(f"Failed to store analysis results in RAG system: {e}")
    
    def generate_comparative_analysis(self, results_summary: Dict[str, Any]) -> str:
        """
        Generate comparative analysis across multiple modules using RAG context.
        
        Parameters:
        -----------
        results_summary : dict
            Combined results from all analysis modules
        
        Returns:
        --------
        str
            AI-generated comparative analysis enhanced with RAG retrieval
        """
        # Fallback mode when no API key is available
        if self.fallback_mode:
            return """
**AI Simulation Mode - Comparative Analysis**

🔬 **Cross-Module Insights:**
This comprehensive analysis demonstrates the synergy between different astronomical analysis techniques:

📡 **21cm ↔ Galaxy Observations:**
- Reionization signatures correlate with galaxy formation efficiency
- Power spectrum features match observed galaxy clustering

🌌 **Cluster Environment ↔ Individual Galaxies:**
- Environmental quenching mechanisms confirmed across scales
- Stellar population properties show clear environmental dependence

🔭 **JWST Spectroscopy ↔ Broad-band Photometry:**
- High-resolution spectra validate SED fitting assumptions
- Emission line diagnostics refine stellar population models

*Note: This is a simulation. Enable OpenAI integration for full AI-powered comparative analysis.*
            """
        
        # Retrieve relevant context using RAG for comparative analysis
        rag_context = ""
        if self.rag_system:
            query_text = "comparative analysis cross-module galaxy evolution cosmic reionization cluster environment JWST"
            rag_context = self.rag_system.generate_context(query_text, top_k=5)
        
        try:
            context = f"""
            {rag_context}
            
            Now provide a comprehensive comparative analysis across these NEW multi-wavelength 
            and multi-epoch astronomical results:
            
            {json.dumps(results_summary, indent=2)}
            
            Focus on:
            1. Connecting 21cm cosmic evolution with galaxy observations
            2. Environmental effects on galaxy evolution
            3. JWST insights into early galaxy formation
            4. Synthesis of results across cosmic time
            5. Implications for galaxy formation models
            6. Future observational priorities
            
            Use the retrieved context to identify patterns, connections, and insights 
            across different analysis modules. Structure as a scientific synthesis 
            suitable for a research summary.
            """
            
            return self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt + "\n\nYou have access to a knowledge base of previous analyses. Use this context to provide comprehensive cross-module insights."},
                    {"role": "user", "content": context},
                ],
                max_tokens=2000,
                temperature=0.7,
            )
            
        except Exception as e:
            return f"Comparative analysis temporarily unavailable: {str(e)}"
    
    def generate_report_section(self, section_type: str, 
                               data: Dict[str, Any]) -> str:
        """
        Generate specific report sections (Introduction, Methods, Results, Discussion).
        
        Parameters:
        -----------
        section_type : str
            Type of section ('introduction', 'methods', 'results', 'discussion')
        data : dict
            Relevant data for the section
        
        Returns:
        --------
        str
            AI-generated report section
        """
        # Fallback mode when no API key is available  
        if self.fallback_mode:
            return f"""
**AI Simulation Mode - {section_type.title()} Section**

{self._generate_fallback_report_section(section_type)}

*Note: This is a simulation. Enable OpenAI integration for full AI-powered report generation.*
            """
        
        try:
            if section_type == "introduction":
                prompt = f"""
                Write a scientific introduction for a galaxy evolution study that includes:
                {json.dumps(data, indent=2)}
                
                Include relevant background, motivation, and objectives.
                """
            
            elif section_type == "methods":
                prompt = f"""
                Write a methods section describing the analysis techniques used:
                {json.dumps(data, indent=2)}
                
                Include technical details appropriate for peer review.
                """
            
            elif section_type == "results":
                prompt = f"""
                Write a results section summarizing key findings:
                {json.dumps(data, indent=2)}
                
                Present results objectively with quantitative details.
                """
            
            elif section_type == "discussion":
                prompt = f"""
                Write a discussion section interpreting results and implications:
                {json.dumps(data, indent=2)}
                
                Include scientific interpretation, limitations, and future work.
                """
            
            else:
                prompt = f"Generate {section_type} content for: {json.dumps(data, indent=2)}"
            
            return self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
            )
            
        except Exception as e:
            return f"Report section generation temporarily unavailable: {str(e)}"
    
    def suggest_next_steps(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Suggest next steps and follow-up analyses based on current results.
        
        Parameters:
        -----------
        analysis_results : dict
            Current analysis results and findings
        
        Returns:
        --------
        list
            List of suggested next steps and analyses
        """
        # Fallback mode when no API key is available
        if self.fallback_mode:
            return [
                "🔬 Expand parameter space in 21cm simulations for broader redshift coverage",
                "📊 Increase galaxy sample size for improved statistical significance",
                "🌌 Include additional cluster environments (groups, field galaxies) for comparison",
                "🔭 Extend JWST spectroscopic analysis to include NIRCam imaging",
                "📈 Implement machine learning techniques for pattern recognition",
                "🎯 Focus on specific emission line diagnostics (metallicity, star formation)",
                "🔄 Cross-validate results with independent observational datasets",
                "📝 Prepare findings for publication in peer-reviewed journals"
            ]
        
        try:
            context = f"""
            Based on these analysis results, suggest specific next steps for research:
            {json.dumps(analysis_results, indent=2)}
            
            Provide 5-8 concrete, actionable suggestions for follow-up work.
            Format as a numbered list.
            """
            
            content = self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context},
                ],
                max_tokens=800,
                temperature=0.8,
            )

            suggestions = content.split('\n')
            suggestions = [s.strip() for s in suggestions if s.strip() and 
                          (s.strip()[0].isdigit() or s.strip().startswith('-'))]
            
            return suggestions[:8]  # Limit to 8 suggestions
            
        except Exception as e:
            return [f"Next steps analysis temporarily unavailable: {str(e)}"]
    
    def check_api_status(self) -> bool:
        """
        Check if OpenAI API is accessible and functioning.
        
        Returns:
        --------
        bool
            True if API is working, False otherwise
        """
        # Return False immediately if in fallback mode
        if self.fallback_mode:
            return False
        
        try:
            _ = self._chat(
                messages=[{"role": "user", "content": "Ping"}],
                max_tokens=5,
                temperature=0.0,
            )
            return True
        except Exception:
            return False

    # ------------------------- Internal Helpers ------------------------- #
    def _chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Unified chat abstraction supporting both new and legacy SDKs.

        Parameters
        ----------
        messages : list
            List of role/content dicts.
        max_tokens : int
            Completion max tokens.
        temperature : float
            Sampling temperature.
        """
        if _NEW_OPENAI_SDK:
            try:
                if self.use_responses_api:
                    # Convert messages into Responses API input format
                    # First system prompt (if present) becomes instructions
                    instructions = None
                    user_inputs: List[Dict[str, Union[str, list]]] = []
                    for m in messages:
                        if m["role"] == "system" and instructions is None:
                            instructions = m["content"]
                        else:
                            user_inputs.append(
                                {
                                    "role": m["role"],
                                    "content": [
                                        {"type": "input_text", "text": m["content"]}
                                    ],
                                }
                            )
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
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"OpenAI request failed: {e}") from e
        else:  # legacy path
            try:
                completion = self.client.ChatCompletion.create(  # type: ignore[attr-defined]
                    model=self.model if self.model.startswith("gpt-") else "gpt-4",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return completion.choices[0].message.content  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"Legacy OpenAI request failed: {e}") from e

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ):
        """Generator yielding streamed tokens (new SDK only).

        Falls back to non-streaming if streaming unsupported.
        """
        if not _NEW_OPENAI_SDK or self.use_responses_api:
            # Fallback: return single chunk
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
                delta = getattr(event.choices[0].delta, "content", None)  # type: ignore
                if delta:
                    yield delta
        except Exception as e:  # pragma: no cover
            yield f"[Streaming failed: {e}]"

    def _generate_fallback_insight(self, data_summary: Dict[str, Any], analysis_type: str) -> str:
        """Generate simulated insights when OpenAI API is not available"""
        insights = {
            "cosmic_evolution": """
**AI Simulation Mode - Cosmic Evolution Analysis**

Based on the 21cm simulation data provided:

🔬 **Key Scientific Findings:**
- Reionization signatures detected in brightness temperature evolution
- Power spectrum analysis reveals clustering patterns consistent with CDM model
- Galaxy formation efficiency shows expected redshift dependence
- Ionization fraction evolution follows theoretical predictions

📊 **Statistical Insights:**
- Strong correlation between halo mass and star formation activity
- Power spectrum peak indicates characteristic scale of first galaxies
- Temperature fluctuations consistent with Wouthuysen-Field coupling

🚀 **Physical Interpretation:**
This analysis provides valuable constraints on early universe physics and the timing of cosmic reionization.

*Note: This is a simulation. Enable OpenAI integration for full AI analysis.*
            """,
            "cluster_analysis": """
**AI Simulation Mode - Galaxy Cluster Analysis**

SED fitting and cluster environment analysis reveals:

🌌 **Galaxy Population:**
- Clear red sequence and blue cloud separation
- Mass-metallicity relation follows expected trends
- Star formation quenching in dense environments
- Color-magnitude diagram shows evolutionary sequences

📈 **Environmental Effects:**
- Cluster mass correlates with galaxy properties
- Radial gradients in star formation activity
- Evidence for ram-pressure stripping in outer regions

🔍 **Stellar Populations:**
- Age-metallicity degeneracy resolved through multi-band photometry
- Stellar mass function consistent with hierarchical formation

*Note: This is a simulation. Enable OpenAI integration for full AI analysis.*
            """,
            "jwst_spectroscopy": """
**AI Simulation Mode - JWST Spectroscopic Analysis**

High-resolution spectral analysis indicates:

🔬 **Spectral Features:**
- Strong emission lines detected (Hα, [OIII], [OII])
- Continuum fitting reveals stellar population properties
- Redshift determination with high precision
- Dust attenuation curves measured

⭐ **Physical Properties:**
- Star formation rate from emission line fluxes
- Metallicity gradients across galaxy structure
- Stellar mass from continuum modeling
- Age constraints from absorption features

🌠 **Galaxy Evolution:**
- Evidence for recent star formation episodes
- Chemical enrichment history traced through abundance patterns
- Kinematic structure reveals rotation/dispersion properties

*Note: This is a simulation. Enable OpenAI integration for full AI analysis.*
            """
        }
        
        return insights.get(analysis_type, insights["cosmic_evolution"])

    def _generate_fallback_report_section(self, section_type: str) -> str:
        """Generate simulated report sections when OpenAI API is not available"""
        
        sections = {
            "introduction": """
# Introduction

This analysis employs cutting-edge astronomical simulation and analysis tools to investigate galaxy evolution across cosmic time. By combining 21cm cosmological simulations, galaxy cluster SED fitting, and JWST spectroscopic analysis, we provide a comprehensive view of the processes shaping galaxies from the epoch of reionization to the present day.

Our multi-scale approach enables investigation of:
- Early universe physics through 21cm power spectrum analysis
- Environmental effects on galaxy evolution in cluster environments  
- Detailed stellar population properties from high-resolution spectroscopy
            """,
            "methods": """
# Methodology

Our analysis framework integrates multiple state-of-the-art tools:

**21cm Cosmological Simulations:**
- 21cmFAST modeling of reionization epoch
- Power spectrum analysis and statistical correlations
- Brightness temperature evolution tracking

**Galaxy Cluster Analysis:**
- Bagpipes Bayesian SED fitting framework
- Multi-band photometric analysis
- Environmental parameter correlations

**JWST Spectroscopic Pipeline:**
- Standard calibration and reduction procedures
- Optimal 1D spectral extraction algorithms
- Emission line fitting and stellar population modeling
            """,
            "results": """
# Results

Our comprehensive analysis reveals:

**Cosmic Evolution Findings:**
- Clear detection of reionization signatures in 21cm power spectra
- Galaxy formation efficiency evolution consistent with theoretical predictions
- Strong correlations between halo mass and observable properties

**Cluster Environment Effects:**
- Environmental quenching signatures in galaxy populations
- Radial gradients in star formation activity and stellar populations
- Clear separation of red sequence and blue cloud populations

**JWST Spectroscopic Insights:**
- High-precision redshift measurements and stellar population properties
- Detailed emission line diagnostics revealing star formation and metallicity
- Evidence for complex star formation histories and chemical evolution
            """,
            "discussion": """
# Discussion

These results provide new insights into galaxy evolution across cosmic time:

**Implications for Early Universe Physics:**
Our 21cm analysis constrains reionization timing and galaxy formation efficiency, providing crucial tests of theoretical models.

**Environmental Effects on Galaxy Evolution:**
The cluster analysis demonstrates clear environmental dependencies, with implications for understanding galaxy transformation processes.

**Stellar Population Archaeology:**
JWST spectroscopy enables unprecedented detail in stellar population analysis, revealing complex formation histories and enrichment patterns.

**Future Directions:**
This integrated approach opens new avenues for understanding galaxy evolution, with potential for expanded surveys and improved theoretical modeling.
            """
        }
        
        return sections.get(section_type, sections["introduction"])

    def generate_template_analysis(self, template_name: str, session_state: Any) -> str:
        """
        Generate analysis based on predefined templates.
        
        Parameters:
        -----------
        template_name : str
            Name of the analysis template
        session_state : Any
            Streamlit session state containing analysis results
        
        Returns:
        --------
        str
            Generated template analysis
        """
        # Fallback mode when no API key is available
        if self.fallback_mode:
            return self._generate_fallback_template_analysis(template_name)
        
        try:
            # Create template-specific prompts
            template_prompts = {
                "Cosmic evolution timeline analysis": f"""
                Analyze the cosmic evolution timeline from our 21cm simulations.
                Focus on key epochs: reionization, first galaxies, structure formation.
                Provide a chronological narrative of early universe evolution.
                
                Session data summary: {str(session_state.__dict__ if hasattr(session_state, '__dict__') else 'No data available')}
                """,
                
                "Galaxy cluster environmental study": f"""
                Examine environmental effects on galaxy evolution in cluster settings.
                Compare central vs satellite galaxies, discuss quenching mechanisms.
                Analyze color-magnitude relations and stellar population gradients.
                
                Session data summary: {str(session_state.__dict__ if hasattr(session_state, '__dict__') else 'No data available')}
                """,
                
                "JWST spectroscopic deep dive": f"""
                Perform detailed analysis of JWST spectroscopic observations.
                Focus on emission lines, stellar populations, chemical evolution.
                Discuss implications for galaxy formation and evolution models.
                
                Session data summary: {str(session_state.__dict__ if hasattr(session_state, '__dict__') else 'No data available')}
                """,
                
                "High-redshift galaxy formation insights": f"""
                Analyze high-redshift galaxy formation and early universe physics.
                Connect 21cm signatures with galaxy observations.
                Discuss implications for Lambda-CDM cosmology and galaxy formation models.
                
                Session data summary: {str(session_state.__dict__ if hasattr(session_state, '__dict__') else 'No data available')}
                """
            }
            
            prompt = template_prompts.get(template_name, f"Analyze the astronomical data for: {template_name}")
            
            content = self._chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
                temperature=0.7,
            )
            
            return content
            
        except Exception as e:
            error_message = str(e)
            
            # Handle specific rate limit errors
            if "429" in error_message or "rate-limited" in error_message.lower():
                st.warning("🚦 **Rate Limit Reached**: The free AI model is temporarily busy. Using simulation mode for now.")
                return self._generate_fallback_template_analysis(template_name)
            
            # Handle other API errors gracefully
            elif "error" in error_message.lower():
                st.warning("⚠️ **AI Service Temporarily Unavailable**: Switching to simulation mode.")
                return self._generate_fallback_template_analysis(template_name)
            
            # Generic fallback
            else:
                st.info("ℹ️ Using simulation mode for analysis.")
                return self._generate_fallback_template_analysis(template_name)

    def _generate_fallback_template_analysis(self, template_name: str) -> str:
        """Generate simulated template analysis when OpenAI API is not available"""
        
        templates = {
            "Cosmic evolution timeline analysis": """
# Cosmic Evolution Timeline Analysis (AI Simulation Mode)

## 🌌 **Early Universe Timeline**

**z > 20 (First 200 Myr):** 
- Dark matter structure formation begins
- First dark matter halos collapse
- Primordial gas cooling in minihalos

**z = 15-20 (200-400 Myr):**
- First Population III stars form
- Initial heavy element production
- First supernovae and feedback

**z = 10-15 (400-600 Myr):**
- Transition to Population II star formation  
- First galaxies with ongoing star formation
- Beginning of cosmic reionization

**z = 6-10 (600 Myr - 1 Gyr):**
- Peak of reionization epoch
- 21cm power spectrum evolution
- Lyman-α forest signatures

**z < 6 (> 1 Gyr):**
- Universe becomes fully ionized
- Modern galaxy formation continues
- Observable galaxy populations

## 🔬 **Key Physical Processes**

- **Gravitational collapse** drives structure formation
- **Stellar feedback** regulates star formation
- **Radiative feedback** affects gas cooling
- **Chemical enrichment** builds galaxy masses

*Note: This is a simulation. Enable OpenRouter AI for detailed timeline analysis.*
            """,
            
            "Galaxy cluster environmental study": """
# Galaxy Cluster Environmental Study (AI Simulation Mode)

## 🌌 **Environmental Effects Overview**

**Central Galaxies:**
- Massive, early-type morphologies
- Quenched star formation
- High stellar masses and metallicities
- Clear red sequence positioning

**Satellite Galaxies:**
- Ram-pressure stripping effects
- Truncated star formation histories
- Stellar mass function variations
- Distance-dependent properties

## 📊 **Observational Signatures**

**Color-Magnitude Relations:**
- Tight red sequence in cluster cores
- Blue cloud depletion at high masses
- Environmental dependence of scatter
- Morphology-density correlations

**Stellar Population Gradients:**
- Age gradients from center to outskirts
- Metallicity variations with environment
- Star formation quenching timescales
- Chemical evolution differences

## 🔍 **Physical Mechanisms**

- **Ram-pressure stripping** removes gas
- **Tidal interactions** truncate disks
- **Mergers** drive morphological evolution
- **Strangulation** cuts off gas supply

*Note: This is a simulation. Enable OpenRouter AI for comprehensive environmental analysis.*
            """,
            
            "JWST spectroscopic deep dive": """
# JWST Spectroscopic Deep Dive (AI Simulation Mode)

## 🔭 **Spectroscopic Capabilities**

**Emission Line Analysis:**
- Hα, [OIII], [OII] flux measurements
- Star formation rate diagnostics
- Metallicity indicators (R23, N2)
- Ionization parameter constraints

**Continuum Analysis:**
- Stellar population synthesis
- Age and metallicity estimates
- Dust attenuation measurements
- Stellar mass determinations

## 📈 **Physical Properties Derived**

**Star Formation:**
- SFR from emission line luminosities
- Specific star formation rates
- Star formation efficiency
- Recent vs. extended histories

**Chemical Evolution:**
- Gas-phase metallicity gradients
- Stellar abundance patterns
- Chemical enrichment timescales
- Mass-metallicity relations

**Stellar Populations:**
- Age dating from absorption features
- Multiple population components
- Formation epoch constraints
- Assembly history reconstruction

## 🌠 **Scientific Implications**

- **Galaxy formation models** validation
- **Chemical evolution** pathway constraints  
- **Environmental effects** quantification
- **High-redshift** population properties

*Note: This is a simulation. Enable OpenRouter AI for detailed spectroscopic analysis.*
            """,
            
            "High-redshift galaxy formation insights": """
# High-Redshift Galaxy Formation Insights (AI Simulation Mode)

## 🌌 **Early Galaxy Formation**

**Theoretical Framework:**
- Lambda-CDM structure formation
- Gas cooling and star formation thresholds
- Feedback mechanisms and regulation
- Chemical enrichment processes

**Observational Constraints:**
- JWST galaxy surveys at z > 10
- 21cm power spectrum measurements
- Lyman-α emitter populations
- Galaxy luminosity functions

## 🔬 **Key Physical Processes**

**Star Formation Regulation:**
- Stellar feedback and outflows
- Supernova energy injection
- Radiative feedback from massive stars
- Black hole formation and growth

**Chemical Evolution:**
- Population III to Population II transition
- Metal enrichment timescales
- Dust formation and evolution
- ISM chemical complexity

## 📊 **Cosmological Implications**

**Structure Formation:**
- Dark matter halo mass functions
- Baryon fraction evolution
- Galaxy-halo connection
- Clustering and bias measurements

**Reionization Physics:**
- Ionizing photon production
- Escape fraction evolution
- IGM temperature evolution
- 21cm signal interpretation

*Note: This is a simulation. Enable OpenRouter AI for comprehensive formation insights.*
            """
        }
        
        return templates.get(template_name, f"**{template_name}** analysis would be generated here with AI integration enabled.")
    
    def get_rag_status(self) -> Dict[str, Any]:
        """Get status and statistics of the RAG system."""
        if not self.rag_system:
            return {
                'enabled': False,
                'reason': 'RAG system not available',
                'stats': {}
            }
        
        try:
            stats = self.rag_system.get_stats()
            return {
                'enabled': True,
                'reason': 'RAG system operational',
                'stats': stats
            }
        except Exception as e:
            return {
                'enabled': False,
                'reason': f'RAG system error: {str(e)}',
                'stats': {}
            }
    
    def query_knowledge_base(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Direct query to the knowledge base for debugging/exploration."""
        if not self.rag_system:
            return []
        
        try:
            results = self.rag_system.retrieve(query, top_k)
            return [
                {
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'similarity': score
                }
                for doc, score in results
            ]
        except Exception as e:
            st.warning(f"Knowledge base query failed: {e}")
            return []