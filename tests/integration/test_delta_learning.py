"""Integration test for Delta Learning (Initial Potential Passing)."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

from ase import Atoms

from pyacemaker.core.config import (
    DFTConfig,
    DynamicsEngineConfig,
    OracleConfig,
    OrchestratorConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    StructureGeneratorConfig,
    TrainerConfig,
)
from pyacemaker.domain_models.models import Potential, PotentialType, StructureMetadata
from pyacemaker.orchestrator import Orchestrator
from pyacemaker.trainer.pacemaker import PacemakerTrainer


def test_delta_learning_parameter_passing(tmp_path: Path) -> None:
    """Verify that initial_potential is correctly passed to PacemakerWrapper."""

    # 1. Setup Configuration
    project_dir = tmp_path / "delta_test"
    project_dir.mkdir()

    config = PYACEMAKERConfig(
        version="0.1.0",
        project=ProjectConfig(name="DeltaTest", root_dir=project_dir),
        orchestrator=OrchestratorConfig(),
        structure_generator=StructureGeneratorConfig(),
        oracle=OracleConfig(dft=DFTConfig(pseudopotentials={"Fe": "pot"}), mock=True),
        trainer=TrainerConfig(potential_type="pace", mock=False),
        dynamics_engine=DynamicsEngineConfig(mock=True),
    )

    orchestrator = Orchestrator(config, base_dir=tmp_path)

    # Ensure trainer is PacemakerTrainer
    assert isinstance(orchestrator.trainer, PacemakerTrainer)
    trainer = orchestrator.trainer

    # 2. Prepare Mock Data
    def mock_data_stream() -> Iterator[StructureMetadata]:
        for _ in range(5):
            s = StructureMetadata()
            s.energy = -1.0
            s.forces = [[0.0, 0.0, 0.0]]
            s.features["atoms"] = Atoms("Fe", positions=[[0,0,0]])
            yield s

    # 3. Create a Mock Initial Potential
    initial_pot_path = project_dir / "base.yace"
    initial_pot_path.touch()
    initial_pot = Potential(
        path=initial_pot_path,
        type=PotentialType.PACE,
        version="1.0",
        metrics={},
        parameters={}
    )

    # 4. Mock Wrapper
    # Use context manager to patch `train_from_input` and `_generate_input_yaml`
    with patch.object(trainer.wrapper, "train_from_input") as mock_train_wrapper, \
         patch.object(trainer, "_generate_input_yaml", wraps=trainer._generate_input_yaml) as mock_gen_yaml:

        # Mock return value to be a valid path
        mock_train_wrapper.return_value = project_dir / "delta.yace"
        (project_dir / "delta.yace").touch()

        # 5. Call Trainer.train with initial_potential and weight_dft
        trainer.train(mock_data_stream(), initial_potential=initial_pot, weight_dft=10.0)

        # 6. Verify Wrapper Call
        mock_train_wrapper.assert_called_once()
        args, kwargs = mock_train_wrapper.call_args

        # Check explicit kwargs passed to wrapper
        assert kwargs["initial_potential"] == initial_pot_path

        # Check input.yaml path structure
        input_yaml_path = args[0]
        assert input_yaml_path.name == "input.yaml"

        # 7. Verify _generate_input_yaml called with correct config
        mock_gen_yaml.assert_called_once()
        # args[0] is config dict
        config_arg = mock_gen_yaml.call_args[0][0]

        # Check weight_dft mapping
        assert config_arg.get("w_energy") == 10.0
        assert config_arg.get("w_forces") == 10.0
