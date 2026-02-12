from pathlib import Path

from mlip_autopipec.domain_models.datastructures import (
    Potential,
    Structure,
    ValidationResult,
)
from mlip_autopipec.dynamics import MockDynamics
from mlip_autopipec.generator import MockGenerator
from mlip_autopipec.oracle import MockOracle
from mlip_autopipec.trainer import MockTrainer
from mlip_autopipec.validator import MockValidator


def test_component_integration(tmp_path: Path) -> None:
    """
    Verifies that the output of one component is valid input for the next.
    """
    # 1. Generator -> Structures
    generator = MockGenerator()
    context = {"temperature": 300.0}
    structures_iter = generator.explore(context)

    # Consume iterator for verification (it's exhausted after this)
    structures_list = list(structures_iter)
    assert len(structures_list) > 0
    assert all(isinstance(s, Structure) for s in structures_list)

    # Re-create iterator for Oracle
    structures_iter = generator.explore(context)

    # 2. Oracle -> Dataset
    oracle = MockOracle()
    # Oracle returns Iterator[Structure] now
    labeled_structures_iter = oracle.compute(structures_iter)

    # Consume for verification
    labeled_structures_list = list(labeled_structures_iter)
    assert len(labeled_structures_list) > 0
    assert all(s.label_status == "labeled" for s in labeled_structures_list)

    # Re-create iter for Trainer
    # MockOracle is deterministic but we need to feed it fresh structure iter.
    structures_iter = generator.explore(context)
    labeled_structures_iter = oracle.compute(structures_iter)

    # 3. Trainer -> Potential
    trainer = MockTrainer(work_dir=tmp_path)
    # Trainer accepts Iterable
    potential = trainer.train(labeled_structures_iter)

    assert isinstance(potential, Potential)
    assert potential.path.exists()

    # 4. Dynamics -> Trajectory
    dynamics = MockDynamics()
    # Use the first structure from the list as initial configuration
    initial_structure = structures_list[0]
    # Dynamics returns Iterator[Structure]
    trajectory_iter = dynamics.simulate(potential, initial_structure)

    trajectory_list = list(trajectory_iter)
    assert len(trajectory_list) > 0
    assert all(isinstance(s, Structure) for s in trajectory_list)

    # 5. Validator -> Result
    validator = MockValidator()
    result = validator.validate(potential)

    assert isinstance(result, ValidationResult)
    assert result.passed is True
