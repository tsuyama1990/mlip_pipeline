from ase.build import bulk

from mlip_autopipec.domain_models.exploration import ExplorationMethod, StaticTask
from mlip_autopipec.physics.structure_gen.strategies import StrainGenerator


def test_manual_generation_pipeline() -> None:
    # Simulate what the Explorer would do

    # 1. Input Structure
    atoms = bulk("Al", "fcc", a=4.05)

    # 2. Policy Decision
    tasks = [StaticTask(modifiers=["strain"])]

    # 3. Execution
    candidates = []
    for task in tasks:
        # Check method explicitly or via isinstance
        if task.method == ExplorationMethod.STATIC and "strain" in task.modifiers:
            # We assume static task has static parameters with default if not specified
            strain_rng = task.parameters.strain_range
            gen = StrainGenerator(strain_range=strain_rng)
            new_structs = gen.generate(atoms, count=2)
            candidates.extend(new_structs)

    # 4. Assertions
    assert len(candidates) == 2
