import numpy as np
import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import (
    DFTOracleConfig,
    FullConfig,
    M3GNetGeneratorConfig,
    OrchestratorConfig,
    PacemakerTrainerConfig,
    RandomGeneratorConfig,
)
from mlip_autopipec.domain_models.datastructures import Structure, WorkflowState
from mlip_autopipec.domain_models.enums import (
    GeneratorType,
    OracleType,
    TaskStatus,
    TrainerType,
)


def test_structure_valid() -> None:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]])
    forces = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]])
    s = Structure(atoms=atoms, provenance="test", forces=forces)
    assert s.provenance == "test"
    assert s.forces is not None
    assert s.forces.shape == (2, 3)
    assert s.energy is None


def test_structure_invalid_forces_length() -> None:
    atoms = Atoms("H")
    forces = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # Shape mismatch
    with pytest.raises(ValidationError, match="Forces array length"):
        Structure(atoms=atoms, provenance="test", forces=forces)


def test_structure_invalid_forces_dim() -> None:
    atoms = Atoms("H")
    forces = np.array([[0.0, 0.0]])  # Shape (1, 2) instead of (1, 3)
    with pytest.raises(ValidationError, match="Forces array must have shape"):
        Structure(atoms=atoms, provenance="test", forces=forces)


def test_workflow_state_defaults() -> None:
    state = WorkflowState(iteration=0)
    assert state.status == TaskStatus.PENDING
    assert state.current_potential_path is None


def test_full_config_valid() -> None:
    conf = FullConfig(
        orchestrator=OrchestratorConfig(work_dir="work_dir"),
        generator=RandomGeneratorConfig(),
        oracle=DFTOracleConfig(command="pw.x"),
        trainer=PacemakerTrainerConfig(),
    )
    assert conf.orchestrator.max_iterations == 10
    assert conf.generator.type == GeneratorType.RANDOM
    assert conf.oracle.type == OracleType.QUANTUM_ESPRESSO
    assert conf.trainer.type == TrainerType.PACEMAKER


def test_full_config_m3gnet() -> None:
    conf = FullConfig(
        orchestrator=OrchestratorConfig(work_dir="work_dir"),
        generator=M3GNetGeneratorConfig(model_path="path/to/model"),
        oracle=DFTOracleConfig(command="pw.x"),
        trainer=PacemakerTrainerConfig(),
    )
    assert conf.generator.type == GeneratorType.M3GNET
    assert conf.generator.model_path == "path/to/model"


def test_config_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        OrchestratorConfig(work_dir="work_dir", extra_field="bad")


def test_discriminator_loading() -> None:
    # Test that discriminator works with dict input (simulating yaml load)
    data = {
        "orchestrator": {"work_dir": "wd"},
        "generator": {"type": "M3GNET", "model_path": "foo"},
        "oracle": {"type": "QUANTUM_ESPRESSO", "command": "cmd"},
        "trainer": {"type": "PACEMAKER"},
    }
    conf = FullConfig.model_validate(data)
    assert isinstance(conf.generator, M3GNetGeneratorConfig)
    assert conf.generator.model_path == "foo"


def test_discriminator_invalid_type() -> None:
    data = {
        "orchestrator": {"work_dir": "wd"},
        "generator": {"type": "INVALID"},
        "oracle": {"type": "QUANTUM_ESPRESSO", "command": "cmd"},
        "trainer": {"type": "PACEMAKER"},
    }
    with pytest.raises(ValidationError):
        FullConfig.model_validate(data)
