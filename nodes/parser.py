"""
Resume Parser Node.

This is the first node in the pipeline graph. Its only job is to turn a
resume file (PDF or DOCX) on disk into plain text so every later node
can work with it as ordinary strings.
"""

from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader

from schemas.state import PipelineState

# Map file extensions to the LangChain loader class that knows how to
# read them. Both loaders return a list of LangChain `Document` objects,
# each with a `.page_content` string attribute — that's the common
# interface we rely on below, regardless of which loader was used.
_LOADERS_BY_EXTENSION = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
}


def parse_resume_node(state: PipelineState) -> dict:
    """
    Load the resume file at state["resume_path"] and extract its raw text.

    Returns a partial state update:
      - on success: {"resume_text": "<extracted text>"}
      - on failure: {"resume_text": "", "error": "<what went wrong>"}

    Setting `error` (rather than raising) lets the graph keep processing
    the rest of a batch even if one resume is corrupted or unsupported —
    downstream nodes check `state["error"]` and skip their work if it's set.
    """
    resume_path = Path(state["resume_path"])

    if not resume_path.exists():
        return {"resume_text": "", "error": f"Resume file not found: {resume_path}"}

    loader_class = _LOADERS_BY_EXTENSION.get(resume_path.suffix.lower())
    if loader_class is None:
        return {
            "resume_text": "",
            "error": f"Unsupported file type '{resume_path.suffix}'. Only .pdf and .docx are supported.",
        }

    try:
        loader = loader_class(str(resume_path))
        documents = loader.load()
    except Exception as exc:
        # Broad except is intentional here: PDF/DOCX parsing libraries can
        # raise many different exception types for corrupted or malformed
        # files, and we want all of them to become a graceful `error`
        # state instead of crashing the batch run.
        return {"resume_text": "", "error": f"Failed to parse resume '{resume_path.name}': {exc}"}

    resume_text = "\n".join(doc.page_content for doc in documents).strip()

    if not resume_text:
        return {"resume_text": "", "error": f"Resume '{resume_path.name}' parsed but contained no text."}

    return {"resume_text": resume_text, "error": None}
