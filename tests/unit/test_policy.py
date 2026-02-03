from ase import Atoms

from mlip_autopipec.domain_models.exploration import ExplorationMethod
from mlip_autopipec.physics.structure_gen.policy import AdaptivePolicy


def test_policy_decision() -> None:
    policy = AdaptivePolicy()
    atoms = Atoms("Cu")

    tasks = policy.decide_strategy(atoms, current_uncertainty=0.5)
    assert len(tasks) >= 2

    # Check first task is Strain
    task0 = tasks[0]
    assert task0.method == ExplorationMethod.STATIC
    assert "strain" in task0.modifiers
    assert task0.parameters["strain_range"] == 0.1

    # Check second task is Defect
    task1 = tasks[1]
    assert task1.method == ExplorationMethod.STATIC
    assert "defect" in task1.modifiers
    assert task1.parameters["defect_type"] == "vacancy"
