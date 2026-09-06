"""
Prompt templates used across the pipeline's LLM-calling nodes.

Keeping prompts in one file (separate from node logic) makes them easy
to find, tweak, and reuse without digging through business logic.
"""

from langchain_core.prompts import ChatPromptTemplate

# --- Information Extraction Node ---
#
# We ask the LLM to act as a resume-parsing assistant. Because we call
# this prompt via `with_structured_output(ExtractedResume)`, we do NOT
# need to describe the exact JSON shape in the prompt text itself —
# LangChain handles telling the model what fields/types are expected.
# The prompt just needs to give good *instructions* for how to fill
# those fields in accurately.
EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a resume-parsing assistant. Extract these 7 fields from the "
            "resume text and output every single one, even if empty: "
            "name, email, skills, years_experience, education, certifications, past_companies.\n\n"
            "Rules:\n"
            "- skills: list each individual skill separately (e.g. 'Python', 'Go'), not grouped.\n"
            "- years_experience: a number. Estimate from employment dates if not stated directly.\n"
            "- Do not omit any field from your output, even if its value is an empty list.",
        ),
        (
            "human",
            "Resume text:\n---\n{resume_text}\n---\n\n"
            "Now output all 7 fields: name, email, skills, years_experience, education, "
            "certifications, past_companies.",
        ),
    ]
)


# --- Scoring Node ---
#
# The LLM receives BOTH the candidate summary/JD text AND a pre-computed
# cosine similarity score (see nodes/scorer.py). It's told to treat the
# similarity score as one input signal among several, not the sole
# answer — this is the "cosine similarity + LLM-based reasoning" combo
# requested in the project spec.
SCORING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict hiring evaluator. Score how well a candidate matches a job "
            "description on a scale of 0-100.\n\n"
            "Scoring method (follow in order):\n"
            "1. List the job's required skills/qualifications one by one.\n"
            "2. For each requirement, check whether the candidate's summary shows a clear "
            "match, a partial match, or no match at all.\n"
            "3. Base your score primarily on the fraction of requirements clearly met. "
            "A candidate missing most core requirements (e.g. a frontend developer applying "
            "to a backend role, or a data scientist applying to a role needing REST API/"
            "microservices/Docker experience they don't have) should score well below 50, "
            "even if their resume is well-written or their background is impressive in a "
            "different area.\n"
            "4. A pre-computed embedding similarity score is provided as a minor sanity-check "
            "signal only — it is not a reliable measure of actual skill overlap and should "
            "NOT meaningfully influence your score. Do not anchor on it.\n\n"
            "Be honest and specific in your reasoning about which requirements are met or missing.",
        ),
        (
            "human",
            "Job Description:\n---\n{jd_text}\n---\n\n"
            "Candidate Summary:\n---\n{candidate_summary}\n---\n\n"
            "(For reference only, do not anchor on this) embedding similarity: {similarity_score}\n\n"
            "Provide a final match score (0-100) and a short reasoning explanation that "
            "references specific matched or missing requirements.",
        ),
    ]
)
