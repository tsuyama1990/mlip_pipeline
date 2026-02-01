from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.domain_models.exploration import ExplorationTask

class AdaptivePolicy:
    """
    The Decision Engine for the Exploration Phase.
    Decides the strategy based on cycle count and material configuration.
    See SPEC.md Section 3.1 and Cycle 07 specs.
    """

    def decide(self, cycle: int, config: Config) -> ExplorationTask:
        """
        Decide the exploration task.

        Args:
            cycle: Current generation number.
            config: Global configuration.

        Returns:
            ExplorationTask describing what to do.
        """
        # Cycle 0: Cold Start
        if cycle == 0:
            return ExplorationTask(
                method="Static",
                modifiers=["strain", "rattle"]
            )

        # Adaptive Logic
        policy_conf = config.policy

        if policy_conf.is_metal:
            # Metals: Use Hybrid MD/MC (Atom Swap) to explore chemical ordering
            return ExplorationTask(
                method="MD",
                modifiers=["swap"],
                temperature=config.md.temperature,
                steps=config.md.n_steps
            )
        else:
            # Insulators/Ceramics: Use Defect Sampling to explore bond breaking/vacancies
            # Logic: Simple MD is insufficient for hard materials.
            return ExplorationTask(
                method="Static",
                modifiers=["defect"]
            )
