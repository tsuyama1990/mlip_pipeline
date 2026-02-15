"""Unit tests for Dynamics Engine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.core.config import DynamicsEngineConfig, PYACEMAKERConfig
from pyacemaker.domain_models.models import (
    Potential,
    PotentialType,
    StructureMetadata,
    StructureStatus,
)
from pyacemaker.modules.dynamics_engine import LAMMPSEngine, MDInterface, PotentialHelper


@pytest.fixture
def mock_config(tmp_path: Path) -> MagicMock:
    """Create a mock configuration."""
    project = MagicMock()
    project.root_dir = tmp_path

    dynamics = DynamicsEngineConfig(
        engine="lammps",
        gamma_threshold=5.0,
        timestep=0.001,
        temperature=300.0,
        hybrid_baseline="zbl",
        mock=True,
    )

    config = MagicMock(spec=PYACEMAKERConfig)
    config.project = project
    config.dynamics_engine = dynamics
    return config


def test_potential_helper_zbl() -> None:
    """Test PotentialHelper with ZBL baseline."""
    helper = PotentialHelper()
    cmds = helper.get_lammps_commands(
        potential_path=Path("test.yace"),
        baseline_type="zbl",
        elements=["Fe", "Pt"],
    )
    # Check pair_style
    assert any("pair_style hybrid/overlay pace" in c and "zbl" in c for c in cmds)

    # Check pace coeff
    pace_cmd = next((c for c in cmds if "pair_coeff" in c and "pace" in c), None)
    assert pace_cmd is not None
    assert "test.yace" in pace_cmd

    # Check zbl coeff
    zbl_cmd = next((c for c in cmds if "pair_coeff" in c and "zbl" in c), None)
    assert zbl_cmd is not None


def test_potential_helper_lj() -> None:
    """Test PotentialHelper with LJ baseline."""
    helper = PotentialHelper()
    cmds = helper.get_lammps_commands(
        potential_path=Path("test.yace"),
        baseline_type="lj",
        elements=["Fe", "Pt"],
    )
    assert any("pair_style hybrid/overlay pace" in c and "lj/cut" in c for c in cmds)


def test_potential_helper_fallback() -> None:
    """Test PotentialHelper with fallback/none baseline."""
    helper = PotentialHelper()
    cmds = helper.get_lammps_commands(
        potential_path=Path("test.yace"),
        baseline_type="none",
        elements=["Fe", "Pt"],
    )
    assert any("pair_style pace" in c for c in cmds)
    assert not any("hybrid" in c for c in cmds)


def test_md_interface_init(mock_config: MagicMock) -> None:
    """Test MDInterface initialization."""
    md = MDInterface(mock_config)
    assert md.config == mock_config
    assert md.params.hybrid_baseline == "zbl"


def test_md_interface_missing_work_dir(mock_config: MagicMock) -> None:
    """Test MDInterface raises error when work_dir is None."""
    md = MDInterface(mock_config)
    with pytest.raises(ValueError, match="work_dir must be provided"):
        md.run_md(
            StructureMetadata(), Potential(path=Path("p"), type=PotentialType.PACE, version="1.0.0")
        )


@patch("subprocess.run")
def test_md_interface_run_halt(mock_run: MagicMock, mock_config: MagicMock, tmp_path: Path) -> None:
    """Test MDInterface run detecting halt."""
    md = MDInterface(mock_config)

    # Mock extraction method
    with patch.object(md, "_extract_bad_structure") as mock_extract:
        mock_extract.return_value = StructureMetadata(
            status=StructureStatus.CALCULATED,
            energy=-100.0,
            forces=[[0.0, 0.0, 0.0]],
        )

        # Simulate log file with halt message
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        log_file = work_dir / "log.lammps"
        log_file.write_text("Fix halt condition met\n")

        structure = StructureMetadata()
        potential = Potential(path=Path("p.yace"), type=PotentialType.PACE, version="1.0.0")

        halt_info = md.run_md(structure, potential, work_dir)

        assert halt_info.halted
        assert halt_info.structure is not None
        mock_extract.assert_called_once()


def test_lammps_engine_integration(mock_config: MagicMock) -> None:
    """Test LAMMPSEngine integration."""
    engine = LAMMPSEngine(mock_config)
    # Check if engine uses MDInterface
    assert hasattr(engine, "md")
    assert isinstance(engine.md, MDInterface)

    # Test basic run
    res = engine.run()
    assert res.status == "success"

    # Test run_production
    res_prod = engine.run_production(
        Potential(path=Path("p"), type=PotentialType.PACE, version="1.0.0")
    )
    assert res_prod == "mock_production_result"


def test_lammps_engine_exploration(mock_config: MagicMock) -> None:
    """Test LAMMPSEngine exploration loop."""
    engine = LAMMPSEngine(mock_config)
    potential = Potential(path=Path("p.yace"), type=PotentialType.PACE, version="1.0.0")

    # Mock MDInterface.run_md to return Halted info sometimes
    # We want to yield at least one structure

    # Mock secrets.SystemRandom to control halt probability in the loop
    # The loop uses: if secrets.SystemRandom().random() < float(probability):
    # We want it to be True once

    with patch("pyacemaker.modules.dynamics_engine.secrets.SystemRandom") as mock_random:
        # First call < 0.3 (True), others > 0.3 (False)
        mock_random.return_value.random.side_effect = [0.1, 0.9, 0.9, 0.9]

        # We also need to ensure MDInterface sees the log file created by the loop logic
        # OR we can mock md.run_md directly which is safer/cleaner

        from pyacemaker.domain_models.models import HaltInfo

        s = StructureMetadata(tags=["halted"])
        h_true = HaltInfo(halted=True, step=10, max_gamma=20.0, structure=s)
        h_false = HaltInfo(halted=False)

        engine.md = MagicMock()
        engine.md.run_md.side_effect = [h_true, h_false, h_false]

        # Run exploration
        iterator = engine.run_exploration(potential)
        results = list(iterator)

        assert len(results) == 1
        assert results[0] == s
        assert engine.md.run_md.call_count >= 1
