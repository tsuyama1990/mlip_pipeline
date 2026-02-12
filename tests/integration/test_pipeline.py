from pathlib import Path

from mlip_autopipec.domain_models.datastructures import (
    Dataset,
    Potential,
    Structure,
    Trajectory,
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
    dataset = oracle.compute(structures_iter)

    assert isinstance(dataset, Dataset)
    # len(dataset) might be different if filtered, but mock preserves count
    assert all(s.label_status == "labeled" for s in dataset.structures)

    # 3. Trainer -> Potential
    trainer = MockTrainer(work_dir=tmp_path)
    potential = trainer.train(dataset)

    assert isinstance(potential, Potential)
    assert potential.path.exists()

    # 4. Dynamics -> Trajectory
    dynamics = MockDynamics()
    # Use the first structure from the dataset as initial configuration
    initial_structure = dataset.structures[0]
    trajectory = dynamics.simulate(potential, initial_structure)

    assert isinstance(trajectory, Trajectory)
    assert len(trajectory.structures) > 0

    # 5. Validator -> Result
    validator = MockValidator()
    result = validator.validate(potential)

    assert isinstance(result, ValidationResult)
    assert result.passed is True
