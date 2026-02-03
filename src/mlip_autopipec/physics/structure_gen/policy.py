from ase import Atoms

from mlip_autopipec.domain_models.exploration import ExplorationMethod, ExplorationTask


class AdaptivePolicy:
    def decide_strategy(
        self, structure: Atoms, current_uncertainty: float
    ) -> list[ExplorationTask]:
        """
        Analyzes the state and returns a list of tasks.
        For Cycle 03, we return a mix of Strain and Defect tasks.
        """
        tasks = []

        # 1. Strain Task (Elastic exploration)
        # Higher uncertainty -> larger strain range?
        # For now, fixed.
        tasks.append(
            ExplorationTask(
                method=ExplorationMethod.STATIC,
                modifiers=["strain"],
                parameters={"strain_range": 0.1},
            )
        )

        # 2. Defect Task (Robustness)
        tasks.append(
            ExplorationTask(
                method=ExplorationMethod.STATIC,
                modifiers=["defect"],
                parameters={"defect_type": "vacancy"},
            )
        )

        return tasks
