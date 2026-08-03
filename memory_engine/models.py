from enum import StrEnum

from pydantic import BaseModel, Field


class MemoryType(StrEnum):
    PREFERENCE = "preference"
    GOAL = "goal"
    PERSONAL_FACT = "personal_fact"
    RECURRING_PATTERN = "recurring_pattern"
    DECISION = "decision"
    VALUE = "value"
    RELATIONSHIP = "relationship"


class MemoryCandidate(BaseModel):
    content: str = Field(min_length=1)
    memory_type: MemoryType
    confidence: float = Field(ge=0.0, le=1.0)


class Intent(StrEnum):
    """
    The caller's classified goal for a turn — used only as a lens by the
    memory reranker (a reflection-driven turn and a decision-driven turn
    can reasonably prioritize different memories). Classifying intent
    itself is outside this module's scope.
    """

    REFLECTION = "reflection"
    DECISION_SUPPORT = "decision_support"
    GOAL_PLANNING = "goal_planning"
    EXPERIMENT_DESIGN = "experiment_design"
    LEARNING = "learning"
    GENERAL_CONVERSATION = "general_conversation"
