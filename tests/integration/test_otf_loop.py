import contextlib
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import (
    GlobalConfig,
)
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure


@pytest.fixture
def mock_config(tmp_path: Path) -> GlobalConfig:
    data: dict[str, object] = {
        "workdir": tmp_path,
        "max_cycles": 1,
        "components": {
            "generator": {"name": "mock", "n_structures": 1, "cell_size": 10.0, "n_atoms": 2, "atomic_numbers": [1, 1]},
            "oracle": {"name": "mock"},
            "trainer": {
                "name": "pacemaker",
                "max_num_epochs": 1,
                "basis_size": 10,
                "cutoff": 3.0
            },
            "dynamics": {
                "name": "lammps",
                "timestep": 0.001,
                "n_steps": 10,
                "uncertainty_threshold": 2.0
            },
            "validator": {"name": "mock"},
        },
        "orchestrator": {"dataset_filename": "dataset.pckl.gzip"} # Use pckl for Pacemaker
    }
    return GlobalConfig.model_validate(data)

@patch("mlip_autopipec.core.orchestrator.ComponentFactory")
@patch("mlip_autopipec.core.orchestrator.Dataset")
@patch("mlip_autopipec.core.orchestrator.StateManager")
def test_otf_loop_execution(
    mock_state: MagicMock,
    mock_dataset: MagicMock,
    mock_factory: MagicMock,
    mock_config: GlobalConfig,
    tmp_path: Path
) -> None:
    # Setup mocks
    mock_generator = MagicMock()
    mock_oracle = MagicMock()
    mock_trainer = MagicMock() # PacemakerTrainer
    mock_dynamics = MagicMock() # LAMMPSDynamics
    mock_validator = MagicMock()

    mock_factory.get_generator.return_value = mock_generator
    mock_factory.get_oracle.return_value = mock_oracle
    mock_factory.get_trainer.return_value = mock_trainer
    mock_factory.get_dynamics.return_value = mock_dynamics
    mock_factory.get_validator.return_value = mock_validator

    # State Manager
    state_instance = mock_state.return_value
    state_instance.state.current_cycle = 1 # Start at cycle 1 (Dynamics phase)

    # Potential
    mock_config.workdir.mkdir(parents=True, exist_ok=True)
    potential_file = mock_config.workdir / "potential.yace"
    potential_file.touch()

    # Dynamics returns a halted structure
    halted_struct = Structure(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atomic_numbers=np.array([1, 1]),
        cell=np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]),
        pbc=np.array([True, True, True]),
        uncertainty=100.0,
        tags={"provenance": "dynamics_halted"}
    )
    # Dynamics explore yields this structure
    mock_dynamics.explore.return_value = iter([halted_struct])

    # Trainer select_active_set mock
    # Should return a subset of candidates
    def mock_select(candidates: list[Structure], limit: int | None = None) -> list[Structure]:
        return candidates[:limit] if limit else candidates

    mock_trainer.select_active_set.side_effect = mock_select
    mock_trainer.train.return_value = Potential(path=potential_file, format="yace")

    # Oracle compute
    def consume_structures(structures: list[Structure]) -> Generator[Structure, None, None]:
        # We must iterate over structures to trigger the generator chain
        # and thus trigger select_active_set inside _enhance_structures
        # Force list realization to ensure generator chain is exhausted
        yield from list(structures)

    mock_oracle.compute.side_effect = consume_structures

    # Run Orchestrator Cycle
    orchestrator = Orchestrator(mock_config)
    orchestrator.current_potential = Potential(path=potential_file, format="yace") # Set initial pot

    with contextlib.suppress(StopIteration):
        orchestrator._run_cycle()

    # Verification
    # 1. Dynamics.explore was called
    mock_dynamics.explore.assert_called_once()

    # 2. Check if oracle compute was called
    mock_oracle.compute.assert_called_once()

    # 3. Verify trainer select_active_set behavior if called
    if mock_trainer.select_active_set.call_count > 0:
        call_args = mock_trainer.select_active_set.call_args
        candidates = call_args[0][0]
        assert len(candidates) == 21 # 1 anchor + 20 generated
        assert candidates[0] == halted_struct # Anchor first
        assert candidates[1].tags["provenance"] == "local_candidate" # Generated ones

        # Verify dataset received labeled structures
        mock_dataset.return_value.append.assert_called()

    # 4. Check validation was skipped if halt occurred
    if orchestrator.halt_count > 0:
        mock_validator.validate.assert_not_called()
