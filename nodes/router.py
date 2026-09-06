"""
Conditional Router.

Not a LangGraph node in the usual sense — this function is registered
via `graph.add_conditional_edges(...)` rather than `graph.add_node(...)`.
Its only job is to read the current state and return a short string
label that tells LangGraph which node to run next. It does not modify
state itself.
"""

from schemas.state import PipelineState

SHORTLIST_THRESHOLD = 75
MANUAL_REVIEW_THRESHOLD = 50


def route_by_score(state: PipelineState) -> str:
    """
    Decide the next node based on state["score"].

    Returns one of: "shortlist", "manual_review", "reject".

    If a prior node failed (state["error"] is set) or scoring never
    produced a score, we route to "reject" as a safe default — an
    unscoreable candidate shouldn't silently disappear from the batch,
    but they also can't be shortlisted or manually reviewed without a
    reasoning summary to review, so treating them like a reject (with
    the error message standing in for a reasoning summary) keeps every
    resume accounted for in the final output.
    """
    if state.get("error") or state.get("score") is None:
        return "reject"

    score = state["score"]
    if score >= SHORTLIST_THRESHOLD:
        return "shortlist"
    elif score >= MANUAL_REVIEW_THRESHOLD:
        return "manual_review"
    else:
        return "reject"
