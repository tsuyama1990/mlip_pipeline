from ase.build import bulk

from mlip_autopipec.domain_models.exploration import ExplorationMethod, ExplorationTask
from mlip_autopipec.physics.structure_gen.strategies import StrainGenerator


def test_manual_generation_pipeline() -> None:
    # Simulate what the Explorer would do

    # 1. Input Structure
    atoms = bulk("Al", "fcc", a=4.05)

    # 2. Policy Decision
    tasks = [ExplorationTask(method=ExplorationMethod.STATIC, modifiers=["strain"])]

    # 3. Execution
    candidates = []
    for task in tasks:
        if task.method == ExplorationMethod.STATIC and "strain" in task.modifiers:
            gen = StrainGenerator(strain_range=0.05)
            new_structs = gen.generate(atoms, count=2)
            candidates.extend(new_structs)

    # 4. Assertions
    assert len(candidates) == 2
