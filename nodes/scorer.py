"""
Scoring Node.

Combines a mathematical similarity signal (cosine similarity between
candidate and JD embeddings) with LLM-based reasoning to produce a
final 0-100 match score and explanation. This node's output directly
feeds the Conditional Router, which decides Shortlist / Manual Review /
Reject based on state["score"].
"""

import os

import numpy as np
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from prompts.templates import SCORING_PROMPT
from schemas.state import PipelineState, ScoreResult


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Standard cosine similarity: the cosine of the angle between two
    vectors. 1.0 means "pointing the same direction" (very similar
    meaning), 0 means unrelated, -1 means opposite.
    """
    a, b = np.array(vec_a), np.array(vec_b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _compute_similarity_score(candidate_summary: str, jd_vector_store: Chroma) -> float:
    """
    Embed the candidate's skills/experience summary and compare it
    against the average embedding of all JD chunks. Averaging the JD
    chunk vectors gives us one representative "meaning" for the whole
    JD to compare the candidate against, rather than picking just one
    chunk (which the Shortlist Node does differently, via top-k retrieval).

    Returns a similarity score scaled to 0-100 for easier comparison
    alongside the LLM's own 0-100 score.
    """
    embeddings = OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    candidate_vector = embeddings.embed_query(candidate_summary)

    jd_data = jd_vector_store.get(include=["embeddings"])
    jd_vectors = jd_data["embeddings"]
    if jd_vectors is None or len(jd_vectors) == 0:
        return 0.0

    average_jd_vector = np.mean(jd_vectors, axis=0).tolist()
    similarity = _cosine_similarity(candidate_vector, average_jd_vector)

    # Cosine similarity from embedding models is typically in a fairly
    # narrow positive band (roughly 0.2-0.8) rather than spanning the
    # full -1..1 range, so we clip negatives to 0 and scale to 0-100.
    return round(max(0.0, similarity) * 100, 1)


def _get_scoring_chain():
    llm = ChatOllama(
        model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )
    structured_llm = llm.with_structured_output(ScoreResult)
    return SCORING_PROMPT | structured_llm


def score_candidate_node(state: PipelineState, jd_vector_store: Chroma, jd_text: str) -> dict:
    """
    Score how well the candidate matches the job description.

    Unlike the parser/extraction nodes, this node needs two extra
    pieces of data beyond `state`: the JD vector store (for the
    similarity signal) and the raw JD text (for the LLM's reasoning
    prompt). These are passed in as explicit arguments rather than
    stored in PipelineState, since the vector store is a live object,
    not serializable state — see nodes/jd_embedder.py for why.

    In graph.py, we wrap this with a small lambda/partial that supplies
    jd_vector_store and jd_text, since LangGraph nodes normally take
    only `state` as their argument.
    """
    if state.get("error"):
        return {"score": None, "reasoning": None}

    extracted_info = state.get("extracted_info")
    if extracted_info is None:
        return {"score": None, "reasoning": None, "error": "Cannot score: no extracted candidate info available."}

    candidate_summary = (
        f"Skills: {', '.join(extracted_info.skills)}. "
        f"Years of experience: {extracted_info.years_experience}. "
        f"Education: {', '.join(extracted_info.education)}. "
        f"Certifications: {', '.join(extracted_info.certifications)}. "
        f"Past companies: {', '.join(extracted_info.past_companies)}."
    )

    try:
        similarity_score = _compute_similarity_score(candidate_summary, jd_vector_store)
    except Exception as exc:
        return {"score": None, "reasoning": None, "error": f"Similarity computation failed: {exc}"}

    try:
        chain = _get_scoring_chain()
        result = chain.invoke(
            {
                "candidate_summary": candidate_summary,
                "jd_text": jd_text,
                "similarity_score": similarity_score,
            }
        )
    except Exception as exc:
        return {"score": None, "reasoning": None, "error": f"LLM scoring failed: {exc}"}

    return {"score": result.score, "reasoning": result.reasoning, "error": None}
