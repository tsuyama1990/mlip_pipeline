from ase import Atoms

from mlip_autopipec.domain_models.exploration import (
    ExplorationTask,
    StaticParameters,
    StaticTask,
)


class AdaptivePolicy:
    def decide_strategy(
        self, structure: Atoms, current_uncertainty: float
    ) -> list[ExplorationTask]:
        """
        Analyzes the state and returns a list of tasks.
        For Cycle 03, we return a mix of Strain and Defect tasks.
        """
        tasks: list[ExplorationTask] = []

        # 1. Strain Task (Elastic exploration)
        # Higher uncertainty -> larger strain range?
        # For now, fixed.
        tasks.append(
            StaticTask(
                modifiers=["strain"],
                parameters=StaticParameters(strain_range=0.1),
            )
        )

        # 2. Defect Task (Robustness)
        tasks.append(
            StaticTask(
                modifiers=["defect"],
                parameters=StaticParameters(defect_type="vacancy"),
            )
        )

        return tasks
