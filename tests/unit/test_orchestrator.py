import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from ase import Atoms
from ase.io import write
from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.domain_models.potential import Potential, ExplorationResult
from mlip_autopipec.domain_models.structure import Dataset, Structure
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.infrastructure.mocks import MockOracle, MockTrainer, MockExplorer, MockValidator
from mlip_autopipec.main import Orchestrator

def test_orchestrator_initialization(tmp_path: Path) -> None:
    config = GlobalConfig(
        work_dir=tmp_path,
        max_cycles=1,
        oracle={"type": "mock"},
        trainer={"type": "mock"},
        explorer={"type": "mock"}
    )

    orch = Orchestrator(config)
    assert orch.config == config
    assert isinstance(orch.oracle, MockOracle)
    assert isinstance(orch.trainer, MockTrainer)
    assert isinstance(orch.explorer, MockExplorer)
    assert isinstance(orch.validator, MockValidator)

def test_orchestrator_extract_structures(tmp_path: Path) -> None:
    config = GlobalConfig(work_dir=tmp_path, max_cycles=1)
    orch = Orchestrator(config)

    # Create a dummy dump file
    dump_file = tmp_path / "dump.xyz"
    atoms1 = Atoms('H', positions=[[0, 0, 0]])
    atoms2 = Atoms('He', positions=[[1, 0, 0]])
    write(dump_file, [atoms1, atoms2]) # type: ignore[no-untyped-call]

    result = ExplorationResult(
        halted=True,
        dump_file=dump_file,
        high_gamma_frames=[0, 1]
    )

    structures = orch._extract_structures(result)
    assert len(structures) == 2
    assert structures[0].atoms.get_chemical_formula() == "H"
    assert structures[1].atoms.get_chemical_formula() == "He"
    assert structures[0].metadata["source"] == "cycle_0_frame_0"

def test_orchestrator_run_loop(tmp_path: Path) -> None:
    config = GlobalConfig(work_dir=tmp_path, max_cycles=2)

    # Mock Components
    mock_oracle = MagicMock()
    mock_trainer = MagicMock()
    mock_explorer = MagicMock()
    mock_validator = MagicMock()

    # Mock Returns
    mock_explorer.explore.return_value = ExplorationResult(
        halted=True,
        dump_file=Path("dump.xyz"),
        high_gamma_frames=[1, 2]
    )
    mock_oracle.compute.return_value = Dataset(structures=[], name="labeled")
    mock_trainer.train.return_value = Potential(path=Path("pot.yace"))
    mock_validator.validate.return_value = ValidationResult(passed=True)

    # We patch _extract_structures because we don't want to rely on real file IO in this specific test
    with patch.object(Orchestrator, '_extract_structures') as mock_extract:
        from mlip_autopipec.domain_models.structure import Structure
        from ase import Atoms
        mock_extract.return_value = [Structure(atoms=Atoms('H'), metadata={})]

        orch = Orchestrator(
            config,
            oracle=mock_oracle,
            trainer=mock_trainer,
            explorer=mock_explorer,
            validator=mock_validator
        )

        orch.run()

        assert mock_explorer.explore.call_count == 2
        assert mock_oracle.compute.call_count == 2
        assert mock_trainer.train.call_count == 2
