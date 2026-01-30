import logging

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.job import JobResult
from mlip_autopipec.orchestration.orchestrator import Orchestrator

logger = logging.getLogger("mlip_autopipec")


def run_one_shot(config: Config) -> JobResult:
    """
    Execute the One-Shot Pipeline (Cycle 02).
    Delegates to the Orchestrator.
    """
    orchestrator = Orchestrator(config)
    return orchestrator.run_pipeline()
