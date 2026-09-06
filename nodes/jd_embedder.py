"""
JD Embedding Node.

Turns the job description into a searchable vector store (ChromaDB).
This is the RAG "indexing" half of the pipeline — the Scoring Node and
Shortlist Node later do the "retrieval" half, pulling the most relevant
JD chunks back out for a given candidate.

This node is only run ONCE per batch (the JD is the same for every
candidate), not once per resume — see main.py for how it's wired.
"""

import os

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_CHROMA_PERSIST_DIR = "data/chroma_db"
_JD_COLLECTION_NAME = "job_description"


def _get_embeddings():
    """
    The embedding model wraps Ollama's nomic-embed-text model: given a
    string, it returns a vector (list of floats) representing that
    string's meaning. Both indexing (this node) and retrieval (scoring/
    shortlist nodes) must use the SAME embedding model, otherwise the
    vectors aren't comparable to each other.
    """
    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def embed_job_description(jd_text: str) -> Chroma:
    """
    Chunk the job description, embed each chunk, and store them in a
    persisted ChromaDB collection. Returns the Chroma vector store,
    which acts as a retriever for later nodes.

    Chunking strategy: RecursiveCharacterTextSplitter tries to split on
    natural boundaries first (double newlines / paragraphs), falling
    back to single newlines, then sentences, then words, only splitting
    mid-word as a last resort. This keeps each chunk topically coherent
    (e.g. the "Requirements" section stays together) rather than cutting
    text at an arbitrary character count.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,  # slight overlap so a concept split across the boundary isn't lost
    )
    chunks = splitter.split_text(jd_text)

    # Re-creating the collection from scratch each run keeps this node
    # idempotent — running the pipeline twice with an edited JD won't
    # leave stale chunks from a previous version mixed in.
    vector_store = Chroma(
        collection_name=_JD_COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=_CHROMA_PERSIST_DIR,
    )
    existing_ids = vector_store.get()["ids"]
    if existing_ids:
        vector_store.delete(ids=existing_ids)

    vector_store.add_texts(chunks)
    return vector_store


def jd_embedding_node(state: dict) -> dict:
    """
    LangGraph node wrapper around embed_job_description.

    Note: this node doesn't store the vector store object itself in
    PipelineState (LangGraph state should stay serializable-ish data,
    not live objects), so main.py calls embed_job_description() once
    up front and passes the resulting retriever directly into the
    scoring/shortlist nodes at graph-build time instead of through state.
    This function exists mainly for symmetry/testing and for a batch
    entry point that only has jd_text in state.
    """
    jd_text = state.get("jd_text", "")
    if not jd_text.strip():
        return {"error": "No job description text provided to embed."}

    try:
        embed_job_description(jd_text)
    except Exception as exc:
        return {"error": f"Failed to embed job description: {exc}"}

    return {"error": None}
