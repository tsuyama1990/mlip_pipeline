"""UAT for Cycle 02."""

from pathlib import Path

import pytest

from pyacemaker.core.config import (
    CONSTANTS,
    DFTConfig,
    DistillationConfig,
    MaceConfig,
    OracleConfig,
    ProjectConfig,
    PYACEMAKERConfig,
    Step1DirectSamplingConfig,
    Step2ActiveLearningConfig,
)
from pyacemaker.modules.mace_workflow import MaceDistillationWorkflow
from pyacemaker.oracle.dataset import DatasetManager
from pyacemaker.orchestrator import Orchestrator


@pytest.fixture
def uat_config(tmp_path: Path) -> PYACEMAKERConfig:
    """UAT configuration."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    return PYACEMAKERConfig(
        version=CONSTANTS.default_version,
        project=ProjectConfig(name="UAT_Cycle02", root_dir=project_dir),
        oracle=OracleConfig(
            dft=DFTConfig(pseudopotentials={"Fe": "Fe.pbe"}),
            mace=MaceConfig(model_path="medium", mock=True),
            mock=True,
        ),
        distillation=DistillationConfig(
            enable_mace_distillation=True,
            step1_direct_sampling=Step1DirectSamplingConfig(target_points=50),
            step2_active_learning=Step2ActiveLearningConfig(
                n_select=10,
                cycles=1,
                uncertainty_threshold=0.0,  # Low threshold to ensure selection in mock
            ),
        ),
    )


def test_scenario_01_intelligent_structure_generation(uat_config: PYACEMAKERConfig) -> None:
    """Scenario 01: Intelligent Structure Generation."""
    # Orchestrator delegates to Workflow. We instantiate workflow directly for granular test.
    orchestrator = Orchestrator(uat_config)
    workflow = MaceDistillationWorkflow(
        config=uat_config,
        dataset_manager=orchestrator.dataset_manager,
        dataset_path=orchestrator.dataset_path,
        oracle=orchestrator.oracle,
        mace_oracle=orchestrator.mace_oracle,  # type: ignore[arg-type]
        trainer=orchestrator.trainer,
        mace_trainer=orchestrator.mace_trainer,  # type: ignore[arg-type]
        dynamics_engine=orchestrator.dynamics_engine,
        structure_generator=orchestrator.structure_generator,
        validation_path=orchestrator.validation_path,
        training_path=orchestrator.training_path,
    )

    # Use internal workflow method
    pool_path = workflow._step1_direct_sampling(uat_config.distillation)

    assert pool_path.exists()

    manager = DatasetManager()
    structures = list(manager.load_iter(pool_path))

    assert len(structures) == 50


def test_scenario_02_active_learning_selection(uat_config: PYACEMAKERConfig) -> None:
    """Scenario 02: Active Learning Selection."""
    uat_config.distillation.step2_active_learning.uncertainty_threshold = 0.4

    orchestrator = Orchestrator(uat_config)
    workflow = MaceDistillationWorkflow(
        config=uat_config,
        dataset_manager=orchestrator.dataset_manager,
        dataset_path=orchestrator.dataset_path,
        oracle=orchestrator.oracle,
        mace_oracle=orchestrator.mace_oracle,  # type: ignore[arg-type]
        trainer=orchestrator.trainer,
        mace_trainer=orchestrator.mace_trainer,  # type: ignore[arg-type]
        dynamics_engine=orchestrator.dynamics_engine,
        structure_generator=orchestrator.structure_generator,
        validation_path=orchestrator.validation_path,
        training_path=orchestrator.training_path,
    )

    # Step 1 first to generate pool
    pool_path = workflow._step1_direct_sampling(uat_config.distillation)

    # Step 2
    workflow._step2_active_learning_loop(uat_config.distillation, pool_path)

    # Verify dataset file exists and has entries
    assert orchestrator.dataset_path.exists()

    manager = DatasetManager()
    labeled = list(manager.load_iter(orchestrator.dataset_path))

    assert len(labeled) > 0
    # Should be 10 selected (n_select=10)
    assert len(labeled) == 10

    # Verify energy is calculated
    for atoms in labeled:
        # MockOracle sets energy
        assert atoms.get_potential_energy() is not None
