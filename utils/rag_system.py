"""Lightweight RAG System for Astro-AI

A retrieval-augmented generation system using only existing dependencies.
Stores astronomical knowledge and retrieves relevant context for LLM queries.
No external vector databases - uses numpy and basic similarity computation.
"""

import json
import pickle
import os
import re
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import math
import numpy as np
import streamlit as st


class AstroRAGSystem:
    """
    Lightweight Retrieval-Augmented Generation system for astronomical data.
    
    Uses TF-IDF with cosine similarity for document retrieval without external dependencies.
    Designed specifically for astrophysical contexts and scientific explanations.
    """
    
    def __init__(self, knowledge_base_path: str = "data/astro_knowledge_base.json"):
        """Initialize the RAG system with a knowledge base path."""
        self.knowledge_base_path = knowledge_base_path
        self.documents = []  # List of document dictionaries
        self.document_vectors = []  # TF-IDF vectors as numpy arrays
        self.vocabulary = {}  # word -> index mapping
        self.idf_scores = {}  # word -> IDF score
        self._ensure_data_directory()
        self._load_knowledge_base()
    
    def _ensure_data_directory(self):
        """Ensure the data directory exists."""
        os.makedirs(os.path.dirname(self.knowledge_base_path), exist_ok=True)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for scientific text."""
        # Convert to lowercase and extract words/scientific terms
        text = text.lower()
        # Keep scientific notation and technical terms
        tokens = re.findall(r'\b(?:\d+\.?\d*(?:e[+-]?\d+)?|\w+)\b', text)
        return [token for token in tokens if len(token) > 1]  # Filter very short tokens
    
    def _compute_tf_idf(self, documents: List[str]) -> np.ndarray:
        """Compute TF-IDF vectors for documents using numpy."""
        if not documents:
            return np.array([])
        
        # Tokenize all documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        
        # Build vocabulary
        all_words = set()
        for doc_tokens in tokenized_docs:
            all_words.update(doc_tokens)
        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        vocab_size = len(self.vocabulary)
        
        # Compute IDF scores
        doc_count = len(documents)
        word_doc_count = defaultdict(int)
        
        for doc_tokens in tokenized_docs:
            unique_words = set(doc_tokens)
            for word in unique_words:
                word_doc_count[word] += 1
        
        self.idf_scores = {
            word: math.log(doc_count / count) 
            for word, count in word_doc_count.items()
        }
        
        # Compute TF-IDF vectors
        tfidf_matrix = np.zeros((doc_count, vocab_size))
        
        for doc_idx, doc_tokens in enumerate(tokenized_docs):
            # Compute term frequencies
            tf_scores = Counter(doc_tokens)
            doc_length = len(doc_tokens)
            
            for word, tf_count in tf_scores.items():
                if word in self.vocabulary:
                    word_idx = self.vocabulary[word]
                    tf = tf_count / doc_length  # Normalized TF
                    idf = self.idf_scores[word]
                    tfidf_matrix[doc_idx, word_idx] = tf * idf
        
        return tfidf_matrix
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def add_document(self, content: str, metadata: Dict[str, Any]):
        """Add a document to the knowledge base."""
        doc = {
            'content': content,
            'metadata': metadata,
            'id': len(self.documents)
        }
        self.documents.append(doc)
        self._rebuild_vectors()
    
    def add_scientific_result(self, analysis_type: str, results: Dict[str, Any], 
                            interpretation: str, source_module: str):
        """Add scientific results with structured metadata."""
        # Create a comprehensive content string
        content_parts = [interpretation]
        
        # Add key results in readable format
        if 'key_metrics' in results:
            metrics_text = f"Key metrics: {', '.join(f'{k}={v}' for k, v in results['key_metrics'].items())}"
            content_parts.append(metrics_text)
        
        # Add method information
        if 'method' in results:
            content_parts.append(f"Analysis method: {results['method']}")
        
        content = ' '.join(content_parts)
        
        metadata = {
            'type': 'scientific_result',
            'analysis_type': analysis_type,
            'source_module': source_module,
            'results': results,
            'timestamp': st.session_state.get('last_analysis_time', 'unknown')
        }
        
        self.add_document(content, metadata)
    
    def _rebuild_vectors(self):
        """Rebuild TF-IDF vectors for all documents."""
        if not self.documents:
            self.document_vectors = []
            return
        
        contents = [doc['content'] for doc in self.documents]
        tfidf_matrix = self._compute_tf_idf(contents)
        self.document_vectors = [tfidf_matrix[i] for i in range(len(self.documents))]
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieve most relevant documents for a query."""
        if not self.documents or not self.vocabulary:
            return []
        
        # Convert query to TF-IDF vector
        query_tokens = self._tokenize(query)
        query_vector = np.zeros(len(self.vocabulary))
        
        if query_tokens:
            query_tf = Counter(query_tokens)
            query_length = len(query_tokens)
            
            for word, tf_count in query_tf.items():
                if word in self.vocabulary:
                    word_idx = self.vocabulary[word]
                    tf = tf_count / query_length
                    idf = self.idf_scores.get(word, 0)
                    query_vector[word_idx] = tf * idf
        
        # Compute similarities
        similarities = []
        for i, doc_vector in enumerate(self.document_vectors):
            sim = self._cosine_similarity(query_vector, doc_vector)
            similarities.append((self.documents[i], sim))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def generate_context(self, query: str, top_k: int = 3) -> str:
        """Generate context string for RAG prompting."""
        retrieved_docs = self.retrieve(query, top_k)
        
        if not retrieved_docs:
            return "No relevant context found in knowledge base."
        
        context_parts = ["=== RETRIEVED CONTEXT ==="]
        
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            if score > 0.01:  # Only include reasonably similar documents
                context_parts.append(f"\n[Context {i}] (Relevance: {score:.3f})")
                context_parts.append(f"Type: {doc['metadata'].get('analysis_type', 'general')}")
                context_parts.append(f"Source: {doc['metadata'].get('source_module', 'unknown')}")
                context_parts.append(f"Content: {doc['content']}")
        
        if len(context_parts) == 1:  # Only header added
            return "No sufficiently relevant context found."
        
        context_parts.append("\n=== END CONTEXT ===")
        return '\n'.join(context_parts)
    
    def populate_with_defaults(self):
        """Populate knowledge base with default astrophysical knowledge."""
        default_docs = [
            {
                'content': '21cm intensity mapping is a technique to study the cosmic dark ages and reionization. The 21cm line of neutral hydrogen provides a powerful probe of the early universe structure formation.',
                'metadata': {'type': 'explanation', 'analysis_type': 'cosmic_evolution', 'source_module': 'default'}
            },
            {
                'content': 'Galaxy clusters are the largest gravitationally bound structures in the universe. Environmental effects in clusters can quench star formation through ram-pressure stripping and tidal interactions.',
                'metadata': {'type': 'explanation', 'analysis_type': 'cluster_analysis', 'source_module': 'default'}
            },
            {
                'content': 'JWST spectroscopy enables detailed studies of galaxy formation and evolution through precise measurements of stellar populations, gas kinematics, and chemical abundances.',
                'metadata': {'type': 'explanation', 'analysis_type': 'jwst_spectroscopy', 'source_module': 'default'}
            },
            {
                'content': 'Bagpipes is a Bayesian spectral energy distribution fitting tool that models stellar populations, star formation histories, and dust attenuation in galaxies.',
                'metadata': {'type': 'explanation', 'analysis_type': 'sed_fitting', 'source_module': 'default'}
            },
            {
                'content': 'Reionization is the phase transition when the first stars and galaxies ionized the neutral hydrogen in the intergalactic medium. This occurred between redshifts 6-15.',
                'metadata': {'type': 'explanation', 'analysis_type': 'cosmic_evolution', 'source_module': 'default'}
            }
        ]
        
        for doc_data in default_docs:
            self.add_document(doc_data['content'], doc_data['metadata'])
    
    def save_knowledge_base(self):
        """Save the knowledge base to disk."""
        data = {
            'documents': self.documents,
            'vocabulary': self.vocabulary,
            'idf_scores': self.idf_scores
        }
        
        with open(self.knowledge_base_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save vectors separately (binary format)
        vector_path = self.knowledge_base_path.replace('.json', '_vectors.pkl')
        with open(vector_path, 'wb') as f:
            pickle.dump(self.document_vectors, f)
    
    def _load_knowledge_base(self):
        """Load the knowledge base from disk."""
        if not os.path.exists(self.knowledge_base_path):
            self.populate_with_defaults()
            self.save_knowledge_base()
            return
        
        try:
            with open(self.knowledge_base_path, 'r') as f:
                data = json.load(f)
            
            self.documents = data.get('documents', [])
            self.vocabulary = data.get('vocabulary', {})
            self.idf_scores = data.get('idf_scores', {})
            
            # Load vectors
            vector_path = self.knowledge_base_path.replace('.json', '_vectors.pkl')
            if os.path.exists(vector_path):
                with open(vector_path, 'rb') as f:
                    self.document_vectors = pickle.load(f)
            else:
                self._rebuild_vectors()
                
        except Exception as e:
            st.warning(f"Could not load knowledge base: {e}. Using defaults.")
            self.populate_with_defaults()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            'total_documents': len(self.documents),
            'vocabulary_size': len(self.vocabulary),
            'analysis_types': list(set(doc['metadata'].get('analysis_type', 'unknown') 
                                    for doc in self.documents)),
            'source_modules': list(set(doc['metadata'].get('source_module', 'unknown') 
                                     for doc in self.documents))
        }


# Global instance for the app
@st.cache_resource
def get_rag_system():
    """Get or create the global RAG system instance."""
    return AstroRAGSystem()