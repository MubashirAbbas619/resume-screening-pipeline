"""
State and structured-output schemas for the resume screening pipeline.

There are two different *kinds* of schema in this file, and it's worth
being clear about the difference up front:

1. Pydantic models (ExtractedResume, ScoreResult) describe data we ask
   the LLM to produce. We hand the model's JSON schema to the LLM via
   LangChain's `with_structured_output(...)`, and get back a validated
   Python object instead of raw text we'd have to parse ourselves.

2. PipelineState (a TypedDict) describes the shared "clipboard" that
   LangGraph passes between every node in the graph. Each node reads
   some fields off it and returns a dict of the fields it wants to
   update. LangGraph merges that returned dict into the running state
   before handing it to the next node.
"""

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class ExtractedResume(BaseModel):
    """
    Structured facts pulled out of a raw resume by the Information
    Extraction Node. This is the "form" we ask the LLM to fill in from
    unstructured resume text.
    """

    name: str = Field(description="Candidate's full name")
    email: str = Field(description="Candidate's email address")
    skills: list[str] = Field(
        default_factory=list, description="Technical and professional skills listed or implied"
    )
    years_experience: float = Field(
        default=0.0, description="Total years of professional experience, estimated if not stated explicitly"
    )
    education: list[str] = Field(
        default_factory=list, description="Degrees and institutions, e.g. 'B.Sc. Computer Science, XYZ University'"
    )
    certifications: list[str] = Field(
        default_factory=list, description="Professional certifications, if any"
    )
    past_companies: list[str] = Field(
        default_factory=list, description="Names of previous employers"
    )


class ScoreResult(BaseModel):
    """
    Output of the Scoring Node: how well a candidate matches the job
    description, plus the reasoning behind that number.
    """

    score: int = Field(ge=0, le=100, description="Match score from 0 (no fit) to 100 (perfect fit)")
    reasoning: str = Field(description="Short explanation of why the candidate does or doesn't fit")


class PipelineState(TypedDict):
    """
    The shared state object passed between every LangGraph node.

    Every node function has the signature `def node(state: PipelineState) -> dict`.
    A node reads whichever fields it needs from `state` and returns a
    dict containing only the fields it wants to change — LangGraph takes
    care of merging that partial update into the full state before the
    next node runs. Fields a node doesn't return are left untouched.
    """

    # --- Inputs ---
    resume_path: str          # file path to the resume being processed (PDF or DOCX)
    jd_text: str               # full text of the job description

    # --- Set by Resume Parser Node ---
    resume_text: str           # raw text extracted from the resume file

    # --- Set by Information Extraction Node ---
    extracted_info: Optional[ExtractedResume]

    # --- Set by Candidate Memory check (repeat-applicant lookup) ---
    is_repeat_applicant: bool
    candidate_history: dict    # this candidate's prior record, if any (empty dict if new)

    # --- Set by Scoring Node ---
    score: Optional[int]
    reasoning: Optional[str]

    # --- Set by Conditional Router (used for logging/debugging, routing itself
    #     is decided by a separate function, not stored ahead of time) ---
    decision: Optional[Literal["shortlist", "manual_review", "reject"]]

    # --- Set by outcome nodes (only one of these gets filled per run) ---
    interview_questions: list[str]
    rejection_email: Optional[str]

    # --- Error handling ---
    # If any node hits a problem (corrupted file, missing fields, etc.)
    # it sets this field. Downstream nodes check it and skip their work
    # so one bad resume doesn't crash the whole batch.
    error: Optional[str]
