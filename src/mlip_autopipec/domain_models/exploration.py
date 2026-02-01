from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

class ExplorationTask(BaseModel):
    """
    Domain model representing an exploration task/strategy decided by the policy.
    See SPEC.md Section 3.1.
    """
    model_config = ConfigDict(extra="forbid")

    method: Literal["MD", "MC", "Minimization", "Static", "aKMC"]
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    steps: Optional[int] = None
    modifiers: List[str] = Field(default_factory=list, description="Modifiers like 'swap', 'strain', 'defect'")
