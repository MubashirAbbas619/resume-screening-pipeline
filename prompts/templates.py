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
