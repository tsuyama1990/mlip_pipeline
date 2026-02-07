from pathlib import Path

import numpy as np
import pytest

from mlip_autopipec.domain_models import Structure
from mlip_autopipec.infrastructure.mocks import (
    MockDynamics,
    MockOracle,
    MockStructureGenerator,
    MockTrainer,
)


def test_mock_components_interaction(tmp_path: Path) -> None:
    # 1. Setup
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    cell = np.eye(3)
    species = ["H", "H"]
    initial_structure = Structure(positions=positions, cell=cell, species=species)

    # 2. Generator
    generator = MockStructureGenerator()
    candidates = generator.generate(initial_structure, strategy="random")
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.positions.shape == (2, 3)

    # 3. Oracle
    oracle = MockOracle()
    labeled_structure = oracle.compute(candidate)
    assert labeled_structure.energy is not None
    assert labeled_structure.forces is not None
    assert labeled_structure.forces.shape == (2, 3)

    # 4. Trainer (Audit: Updated to use List/Iterable instead of Dataset)
    trainer = MockTrainer()
    workdir = tmp_path / "training"
    potential_path = trainer.train([labeled_structure], params={}, workdir=workdir)
    assert potential_path.exists()
    assert potential_path.name == "potential.yace"

    # 5. Dynamics
    dynamics = MockDynamics()
    result = dynamics.run(potential=potential_path, structure=initial_structure)
    assert result.status in ["halted", "converged"]
    assert len(result.trajectory) > 0
    assert isinstance(result.trajectory[0], Structure)

def test_mock_trainer_path_traversal(tmp_path: Path) -> None:
    # Audit: Test security fix
    trainer = MockTrainer()
    # Try to write outside project (e.g., to root /)
    # This is tricky in a container, but let's try a parent relative path that goes way up
    # Since we allow /tmp (which tmp_path is in), we need to try something else disallowed.
    # But for unit testing, we just check that normal usage works and suspicious fails.

    # Construct a path that resolves to something outside CWD and /tmp
    # For safety, let's just use ".." inside tmp_path which should resolve to valid tmp_path parent
    # but still be inside /tmp tree, so it should PASS.
    # To fail, we need to target e.g. /etc or /var if we had permissions, or just outside the allowed root.

    # Let's mock the check logic or trust the implementation:
    # Implementation checks: (workdir_path.is_relative_to(cwd) or workdir_path.is_relative_to(temp))

    # A path completely outside:
    bad_workdir = Path("/invalid_root/mlip_run")
    # This should fail

    with pytest.raises(ValueError, match="Security Violation"):
        trainer.train([], {}, bad_workdir)
