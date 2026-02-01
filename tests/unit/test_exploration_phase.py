import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import ase.build

from mlip_autopipec.domain_models.config import Config, PolicyConfig
from mlip_autopipec.domain_models.workflow import WorkflowState, WorkflowPhase
from mlip_autopipec.domain_models.exploration import ExplorationTask
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.orchestration.phases.exploration import ExplorationPhase

@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.policy = MagicMock(spec=PolicyConfig)
    config.structure_gen = MagicMock()
    config.md = MagicMock()
    config.lammps = MagicMock()
    config.potential = MagicMock()
    config.orchestrator = MagicMock()
    config.orchestrator.uncertainty_threshold = 5.0
    return config

@pytest.fixture
def mock_state():
    return WorkflowState(
        project_name="Test",
        dataset_path=Path("data.pckl"),
        current_phase=WorkflowPhase.EXPLORATION,
        generation=1,
        latest_potential_path=Path("pot.yace")
    )

@patch("mlip_autopipec.orchestration.phases.exploration.AdaptivePolicy")
@patch("mlip_autopipec.orchestration.phases.exploration.ase.io.write")
@patch("mlip_autopipec.orchestration.phases.exploration.StructureGenFactory")
def test_exploration_static_defects(MockFactory, MockWrite, MockPolicy, mock_config, mock_state, tmp_path):
    # Setup Policy to return Static Defect
    policy_instance = MockPolicy.return_value
    policy_instance.decide.return_value = ExplorationTask(
        method="Static",
        modifiers=["defect"]
    )

    # Setup Initial Structure
    atoms = ase.build.bulk("Si")
    struct = Structure.from_ase(atoms)
    MockFactory.get_generator.return_value.generate.return_value = struct

    phase = ExplorationPhase()
    result = phase.execute(mock_state, mock_config, tmp_path)

    # Assertions
    policy_instance.decide.assert_called_with(mock_state.generation, mock_config)

    # Since we passed a file object, check the file object's name or property
    assert MockWrite.called
    args, kwargs = MockWrite.call_args
    # args[0] is the file handle
    assert hasattr(args[0], "name")
    assert str(args[0].name).endswith("dump.lammpstrj")

    assert result.max_gamma > mock_config.orchestrator.uncertainty_threshold
    assert result.trajectory_path.name == "dump.lammpstrj"

@patch("mlip_autopipec.orchestration.phases.exploration.AdaptivePolicy")
@patch("mlip_autopipec.orchestration.phases.exploration.LammpsRunner")
@patch("mlip_autopipec.orchestration.phases.exploration.StructureGenFactory")
def test_exploration_md_fallback(MockFactory, MockRunner, MockPolicy, mock_config, mock_state, tmp_path):
    # Setup Policy to return MD
    policy_instance = MockPolicy.return_value
    policy_instance.decide.return_value = ExplorationTask(
        method="MD",
        modifiers=[]
    )

    atoms = ase.build.bulk("Si")
    struct = Structure.from_ase(atoms)
    MockFactory.get_generator.return_value.generate.return_value = struct

    # Mock LammpsRunner.run result
    mock_result = MagicMock()
    mock_result.max_gamma = 0.5
    MockRunner.return_value.run.return_value = mock_result

    phase = ExplorationPhase()
    result = phase.execute(mock_state, mock_config, tmp_path)

    # Should call LammpsRunner
    assert MockRunner.return_value.run.called
    assert result == mock_result

@patch("mlip_autopipec.orchestration.phases.exploration.AdaptivePolicy")
@patch("mlip_autopipec.orchestration.phases.exploration.ase.io.write")
@patch("mlip_autopipec.orchestration.phases.exploration.StructureGenFactory")
def test_exploration_static_strain(MockFactory, MockWrite, MockPolicy, mock_config, mock_state, tmp_path):
    # Test Strain Strategy path
    policy_instance = MockPolicy.return_value
    policy_instance.decide.return_value = ExplorationTask(
        method="Static",
        modifiers=["strain"]
    )

    atoms = ase.build.bulk("Si")
    struct = Structure.from_ase(atoms)
    MockFactory.get_generator.return_value.generate.return_value = struct

    phase = ExplorationPhase()
    phase.execute(mock_state, mock_config, tmp_path)

    assert MockWrite.called
    args, kwargs = MockWrite.call_args
    assert hasattr(args[0], "name")
    assert str(args[0].name).endswith("dump.lammpstrj")
