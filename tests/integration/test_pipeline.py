from pathlib import Path

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.workflow import ValidationResult
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

    # Do NOT consume entire iterator into list.
    # Check first item.
    first_structure = next(structures_iter)
    assert isinstance(first_structure, Structure)

    # Re-create iterator for Oracle
    structures_iter = generator.explore(context)

    # 2. Oracle -> Dataset
    oracle = MockOracle()
    # Oracle returns Iterator[Structure]
    labeled_structures_iter = oracle.compute(structures_iter)

    # Check first labeled item
    first_labeled = next(labeled_structures_iter)
    assert first_labeled.label_status == "labeled"

    # Re-create iter for Trainer
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
    # Use the first structure
    initial_structure = first_labeled
    # Dynamics returns Iterator[Structure]
    trajectory_iter = dynamics.simulate(potential, initial_structure)

    # Check first frame
    first_frame = next(trajectory_iter)
    assert isinstance(first_frame, Structure)

    # 5. Validator -> Result
    validator = MockValidator()
    result = validator.validate(potential)

    assert isinstance(result, ValidationResult)
    assert result.passed is True
