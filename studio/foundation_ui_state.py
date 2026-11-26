"""Pure helpers for foundation-model UI state; no Streamlit dependency."""

RUNNING_STATES = {"starting", "running", "finalizing"}


def is_foundation_training_running(status: dict | None) -> bool:
    """Return a strict bool suitable for Streamlit disabled= arguments."""
    status = status or {}
    return bool(status.get("state") in RUNNING_STATES or bool(status.get("process_alive", False)))
