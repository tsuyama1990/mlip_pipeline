from enum import StrEnum


class WorkflowStatus(StrEnum):
    """
    Enum representing the current status of the active learning workflow.
    """

    IDLE = "IDLE"
    EXPLORATION = "EXPLORATION"
    LABELING = "LABELING"
    TRAINING = "TRAINING"
    POST_PROCESSING = "POST_PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
