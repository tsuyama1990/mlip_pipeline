"""Tests for MACE Distillation Workflow."""

from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from ase import Atoms

from pyacemaker.core.base import Metrics, ModuleResult
from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.core.interfaces import (
    DynamicsEngine,
    Oracle,
    StructureGenerator,
    Trainer,
    UncertaintyModel,
    Validator,
)
from pyacemaker.domain_models.models import (
    Potential,
    StructureMetadata,
    StructureStatus,
    UncertaintyState,
)
from pyacemaker.orchestrator import Orchestrator


def create_dummy_structure():
    s = StructureMetadata(id=uuid4())
    s.features["atoms"] = Atoms("Fe", positions=[[0,0,0]], cell=[2,2,2])
    return s

class MockOracle(Oracle, UncertaintyModel):
    def __init__(self, config) -> None:
        super().__init__(config)

    def compute_batch(self, structures):
        for s in structures:
            s.status = StructureStatus.CALCULATED
            s.energy = -1.0
            s.forces = [[0.0, 0.0, 0.0]]
            yield s

    def compute_uncertainty(self, structures):
        for s in structures:
            s.uncertainty_state = UncertaintyState(gamma_max=0.9, gamma_mean=0.5) # High uncertainty
            yield s

    def run(self):
        return ModuleResult(status="success", metrics=Metrics())

def test_mace_distillation_workflow(tmp_path):
    # Setup Config
    # We need to ensure required fields for config are present
    # creating minimal config
    config_dict = {
        "project": {"name": "test", "root_dir": tmp_path},
        "oracle": {
            "dft": {
                "code": "vasp",
                "pseudopotentials": {"Fe": "pot"},
                "command": "run"
            },
            "mace": {"model_path": "medium"}
        },
        "distillation": {
            "enable_mace_distillation": True,
            "step1_direct_sampling": {"target_points": 2},
            "step4_surrogate_sampling": {"target_points": 2}
        },
        "version": "0.1.0" # Assuming version 0.1.0
    }
    config = PYACEMAKERConfig(**config_dict)

    # Mock Modules
    mock_sg = MagicMock(spec=StructureGenerator)
    mock_sg.generate_direct_samples.return_value = iter([
        create_dummy_structure(),
        create_dummy_structure()
    ])

    mock_oracle = MockOracle(config)

    mock_trainer = MagicMock(spec=Trainer)
    mock_trainer.train.return_value = Potential(path=Path("pot.yace"), type="PACE", version="1.0", metrics={}, parameters={})

    mock_dyn = MagicMock(spec=DynamicsEngine)
    # Return 2 structures
    mock_dyn.run_exploration.return_value = iter([
        create_dummy_structure(),
        create_dummy_structure()
    ])

    mock_val = MagicMock(spec=Validator)

    mock_mace_trainer = MagicMock(spec=Trainer)
    mock_mace_trainer.train.return_value = Potential(path=Path("mace.model"), type="MACE", version="1.0", metrics={}, parameters={})

    # Orchestrator
    orch = Orchestrator(
        config,
        structure_generator=mock_sg,
        oracle=mock_oracle,
        trainer=mock_trainer,
        dynamics_engine=mock_dyn,
        validator=mock_val,
        mace_trainer=mock_mace_trainer
    )

    # Mock MaceSurrogateOracle inside Orchestrator (it's instantiated in run)
    # We can patch 'pyacemaker.orchestrator.MaceSurrogateOracle'
    with patch('pyacemaker.orchestrator.MaceSurrogateOracle') as MockMaceOracleCls:
        # The mock instance returned by constructor
        mock_mace_instance = MockMaceOracleCls.return_value
        # compute_batch should return calculated structures
        def mock_compute_batch(structures):
            for s in structures:
                s.status = StructureStatus.CALCULATED
                s.energy = -2.0 # Surrogate energy
                s.forces = [[0.1, 0.1, 0.1]]
                yield s
        mock_mace_instance.compute_batch.side_effect = mock_compute_batch

        # Run
        result = orch.run()

    assert result.status == "success"

    # Verify calls
    mock_sg.generate_direct_samples.assert_called_once()
    # mock_oracle.compute_uncertainty is called in Step 2 loop
    # We loop 3 times in implementation.
    # It might be called multiple times.

    # mock_mace_trainer.train called in Step 2 (3 times) + maybe Step 3
    assert mock_mace_trainer.train.called

    assert mock_dyn.run_exploration.called # Step 4

    # mock_trainer.train called Step 6 and Step 7
    assert mock_trainer.train.call_count >= 1
