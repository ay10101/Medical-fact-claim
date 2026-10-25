"""
RAG Engine Module
Handles multilingual embeddings generation via SentenceTransformer and
dense vector search using FAISS.
"""

import os
import sys
import re

# Prevent macOS OpenMP / Rust tokenizers thread conflict
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import faiss
except Exception:
    faiss = None
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

try:
    from src.preprocessing import preprocess_text
    from src.live_retriever import fetch_live_clinical_evidence
except ModuleNotFoundError:
    from preprocessing import preprocess_text
    from live_retriever import fetch_live_clinical_evidence


class NumpyIndexFlatIP:
    """Exact inner-product cosine similarity index fallback when FAISS C++ DLL is restricted."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = None

    def add(self, vectors: np.ndarray):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query_vector: np.ndarray, k: int):
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        scores = np.dot(self.vectors, query_vector.T).flatten()
        top_k = min(k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        return np.array([top_scores], dtype=np.float32), np.array([top_indices], dtype=np.int64)


def create_vector_index(dimension: int):
    """Instantiates FAISS IndexFlatIP if available; otherwise falls back to pure-NumPy index."""
    if faiss is not None:
        try:
            return faiss.IndexFlatIP(dimension)
        except Exception:
            pass
    return NumpyIndexFlatIP(dimension)


class HealthcareRAGEngine:
    """
    RAG engine using paraphrase-multilingual-MiniLM-L12-v2 and FAISS IndexFlatIP
    for cross-lingual semantic matching between English/Hindi/Hinglish claims
    and trusted clinical evidence. Supports both offline dataset indexing and
    real-time live API retrieval.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        csv_path: str = None,
        df: pd.DataFrame = None,
    ):
        # Hardware auto-detection: GPU if available, else CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)

        # Load evidence dataframe
        if df is not None:
            self.df = df.copy()
        elif csv_path and os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Either a valid DataFrame or an existing CSV path must be provided.")

        self.index = None
        self._build_index()

    def _build_index(self):
        """Encodes evidence texts and populates the FAISS inner-product index, caching embeddings for instant startup."""
        cache_path = os.path.join(os.path.dirname(__file__), "..", "data", "evidence_embeddings.npy")
        embeddings = None
        if os.path.exists(cache_path):
            try:
                cached = np.load(cache_path)
                if cached.shape[0] == len(self.df):
                    embeddings = cached.astype(np.float32)
                    print(f"Loaded {len(embeddings)} precomputed embeddings from cache.")
            except Exception as e:
                print(f"Cache load skipped: {e}")

        if embeddings is None:
            print("Encoding evidence dataset for RAG vector index (this is saved for subsequent instant boots)...")
            texts = self.df["evidence_text"].tolist()
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True,
                device=self.device
            )
            embeddings = np.array(embeddings, dtype=np.float32)
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, embeddings)
                print(f"Saved {len(embeddings)} embeddings to cache at {cache_path}")
            except Exception as e:
                print(f"Cache save skipped: {e}")

        dimension = embeddings.shape[1]
        self.index = create_vector_index(dimension)
        self.index.add(embeddings)

    def retrieve_evidence(self, claim: str, top_k: int = 3):
        """
        Retrieves top_k closest evidence snippets for the given claim from local dataset.
        Returns a list of dictionaries with evidence metadata and cosine similarity scores.
        """
        cleaned_claim = preprocess_text(claim)
        query_vector = self.model.encode(
            [cleaned_claim],
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device
        )
        query_vector = np.array(query_vector, dtype=np.float32)

        scores, indices = self.index.search(query_vector, min(top_k, len(self.df)))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            row = self.df.iloc[idx].to_dict()
            row["similarity_score"] = float(round(float(score), 4))
            row["rank"] = rank + 1
            results.append(row)

        return results

    def retrieve_live_evidence(self, claim: str, top_k: int = 3):
        """
        Dynamically fetches live clinical papers & articles from PubMed and Wikipedia APIs,
        combines them with gold-standard Cochrane clinical records, and ranks them by semantic similarity.
        """
        url_match = re.search(r"https?://\S+", claim)
        cleaned_claim = claim.strip() if url_match else preprocess_text(claim)
        live_articles = fetch_live_clinical_evidence(cleaned_claim)

        # Live mode must remain source-pure: curated evidence is only a fallback when live APIs return nothing.
        if not live_articles:
            return []

        # If claim was a URL and direct article was fetched, use the article's title as query for semantic comparison
        query_text = cleaned_claim
        if url_match and live_articles[0].get("article_title"):
            query_text = live_articles[0]["article_title"]

        # Encode live retrieved texts dynamically
        live_texts = [a.get("evidence_text", "") for a in live_articles]
        embeddings = self.model.encode(
            live_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Build an ephemeral in-memory index for live results
        dim = embeddings.shape[1]
        ephemeral_index = create_vector_index(dim)
        ephemeral_index.add(embeddings)

        # Encode query and search ephemeral index
        query_vector = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device
        )
        query_vector = np.array(query_vector, dtype=np.float32)

        search_k = min(top_k, len(live_articles))
        scores, indices = ephemeral_index.search(query_vector, search_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            item = dict(live_articles[idx])
            semantic_score = float(round(float(score), 4))
            correlation_bonus = float(item.get("correlation_score", 0.0))
            item["semantic_similarity_score"] = semantic_score
            item["similarity_score"] = float(round(semantic_score + correlation_bonus, 4))
            item["rank"] = rank + 1

            # If stance is already established from curated dataset, preserve it
            if "stance" in item and item["stance"] in ["supports", "refutes", "uncertain"]:
                results.append(item)
                continue

            # Infer clinical stance from study title and explanation text
            text_corpus = (
                item.get("evidence_text", "") + " " +
                item.get("claim_topic", "") + " " +
                item.get("clinical_explanation", "")
            ).lower()

            if any(term in text_corpus for term in [
                "ineffective", "no effect", "debunk", "myth", "harmful", "toxic", "adverse",
                "contradicts", "disproven", "does not cure", "does not treat", "does not prevent"
            ]):
                item["stance"] = "refutes"
            elif any(term in text_corpus for term in [
                "unproven", "insufficient evidence", "not proven", "lacks evidence", "lacks conclusive",
                "inconclusive", "unclear", "further study needed", "further research is needed",
                "limited evidence", "conflicting evidence", "preliminary evidence", "uncertain"
            ]):
                item["stance"] = "uncertain"
            elif any(term in text_corpus for term in [
                "effective", "effectiveness", "proven", "reduces", "reduce", "reducing",
                "supports", "beneficial", "protective", "protection", "prevention", "prevent",
                "efficacy", "mitigate", "control transmission", "source control", "lowers risk",
                "contain the spread", "randomized controlled trial", "recommended"
            ]):
                item["stance"] = "supports"
            else:
                item["stance"] = "uncertain"

            results.append(item)

        # Keep semantically strong results, but prefer evidence that names the same condition/intervention.
        results.sort(key=lambda item: item.get("similarity_score", 0.0), reverse=True)
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
        return results



class ExternalHealthcareRAGEngine:
    """Live-only RAG engine. It never loads, indexes, or searches the local evidence dataset."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    @staticmethod
    def _infer_stance(item: dict) -> str:
        text = " ".join([
            item.get("evidence_text", ""), item.get("claim_topic", ""),
            item.get("clinical_explanation", ""),
        ]).lower()
        if any(term in text for term in ("ineffective", "no effect", "harmful", "toxic", "does not cure", "does not treat", "does not prevent")):
            return "refutes"
        if any(term in text for term in ("unproven", "insufficient evidence", "not proven", "inconclusive", "unclear", "further research", "limited evidence", "conflicting evidence")):
            return "uncertain"
        if any(term in text for term in ("effective", "effectiveness", "reduces", "beneficial", "protective", "prevention", "efficacy", "randomized controlled trial")):
            return "supports"
        return "uncertain"

    def retrieve_live_evidence(self, claim: str, top_k: int = 3):
        url_match = re.search(r"https?://\S+", claim)
        cleaned_claim = claim.strip() if url_match else preprocess_text(claim)
        articles = fetch_live_clinical_evidence(cleaned_claim)
        if not articles:
            return []

        # If claim was a URL, use the first article title as the semantic matching anchor
        query_text = cleaned_claim
        if url_match and articles[0].get("article_title"):
            query_text = articles[0]["article_title"]

        texts = [item.get("evidence_text", "") for item in articles]
        embeddings = np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, device=self.device), dtype=np.float32)
        query_embedding = np.asarray(self.model.encode([query_text], normalize_embeddings=True, show_progress_bar=False, device=self.device), dtype=np.float32)[0]
        scores = np.dot(embeddings, query_embedding)

        results = []
        for item, semantic_score in zip(articles, scores):
            result = dict(item)
            result["semantic_similarity_score"] = float(round(float(semantic_score), 4))
            result["similarity_score"] = float(round(float(semantic_score) + float(result.get("correlation_score", 0.0)), 4))
            if result.get("stance") not in {"supports", "refutes", "uncertain"}:
                result["stance"] = self._infer_stance(result)
            results.append(result)

        results.sort(key=lambda item: item["similarity_score"], reverse=True)
        for rank, item in enumerate(results[:top_k], start=1):
            item["rank"] = rank
        return results[:top_k]