# File: core/reflective_cognition/ExecutionFeedback.py

from datetime import datetime

def build_reflection(task, outcome_status="success", details=""):
    """
    Build a structured reflection entry based on the task execution result.
    This memory is meant for the reflection pipeline or EmotionCore to process later.
    """

    # Use tags + description from the task to give context
    return {
        "type": "reflection",
        "source": "TaskExecutor",
        "timestamp": datetime.utcnow().isoformat(),
        "content": f"Executed task: '{task.get('description', 'unknown')}'. Outcome: {outcome_status}. {details}",
        "tags": task.get("tags", []),
        "emotion": "satisfaction" if outcome_status == "success" else "frustration"
    }
