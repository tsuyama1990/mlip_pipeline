from mlip_autopipec.config.models import SystemConfig


class WorkflowManager:
    """
    The central orchestrator for the MLIP-AutoPipe workflow.

    This class is responsible for initializing and coordinating the various
    modules (DFT, Explorer, etc.) and managing the main active learning loop.
    It operates on a fully validated SystemConfig.
    """

    def __init__(self, system_config: SystemConfig):
        """
        Initializes the WorkflowManager.

        Args:
            system_config: A fully validated Pydantic model of the entire
                           system configuration.
        """
        self.system_config = system_config

    def run(self):
        """
        Executes the main active learning workflow.
        """
        # This is a placeholder implementation that will be fleshed out in future cycles.
        # For now, we just demonstrate that the workflow can be initiated.
        print("WorkflowManager: run() called")
        print(f"Project: {self.system_config.project_name}")

