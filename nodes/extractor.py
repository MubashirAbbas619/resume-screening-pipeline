"""
Information Extraction Node.

Second node in the pipeline. Takes the raw resume text produced by the
Resume Parser Node and asks a local LLM (via Ollama) to convert it into
a structured ExtractedResume object.
"""

import os

from langchain_ollama import ChatOllama

from prompts.templates import EXTRACTION_PROMPT
from schemas.state import ExtractedResume, PipelineState


def _get_extraction_chain():
    """
    Build the LLM + structured-output chain used to extract resume data.

    We read model config from environment variables (set via .env) so the
    model name / server URL aren't hardcoded. `with_structured_output`
    is what turns free-text LLM output into a validated ExtractedResume
    instance automatically — see the docstring on ExtractedResume in
    schemas/state.py for what fields it expects.
    """
    llm = ChatOllama(
        model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,  # deterministic-ish output; we want consistent extraction, not creativity
    )
    structured_llm = llm.with_structured_output(ExtractedResume)
    # The "|" here builds a LangChain "chain": the prompt template's
    # output (a formatted list of chat messages) is piped directly into
    # the structured LLM as its input. This is LangChain Expression
    # Language (LCEL) — chains built by piping components together.
    return EXTRACTION_PROMPT | structured_llm


def extract_information_node(state: PipelineState) -> dict:
    """
    Extract structured candidate info from state["resume_text"].

    Returns a partial state update:
      - on success: {"extracted_info": ExtractedResume(...)}
      - on failure: {"extracted_info": None, "error": "<what went wrong>"}

    If a previous node already set state["error"] (e.g. the resume
    failed to parse), we skip extraction entirely and pass the error
    through unchanged — there's nothing useful to extract from empty text.
    """
    if state.get("error"):
        return {"extracted_info": None}

    resume_text = state.get("resume_text", "")
    if not resume_text.strip():
        return {"extracted_info": None, "error": "No resume text available to extract from."}

    try:
        chain = _get_extraction_chain()
        extracted = chain.invoke({"resume_text": resume_text})
    except Exception as exc:
        # Broad except is intentional: local LLMs can fail extraction in
        # many ways (malformed JSON, connection errors to the Ollama
        # server, validation errors against the Pydantic schema). Any of
        # these should become a graceful error, not a crashed batch run.
        return {"extracted_info": None, "error": f"Information extraction failed: {exc}"}

    if not extracted.email:
        return {
            "extracted_info": extracted,
            "error": "Extraction succeeded but no email was found — cannot track this candidate.",
        }

    return {"extracted_info": extracted, "error": None}
