import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from mlip_autopipec.domain_models import Structure
from mlip_autopipec.infrastructure.oracle import DFTManager

# Helper to find python executable
PYTHON_EXE = sys.executable

@pytest.fixture
def mock_pw_script(tmp_path: Path) -> Path:
    # Copy mock script to temp dir to avoid import issues or path issues
    source = Path(__file__).parent.parent / "fixtures" / "mock_pw.py"
    dest = tmp_path / "mock_pw.py"
    shutil.copy(source, dest)
    return dest

@pytest.fixture
def pseudo_dir(tmp_path: Path) -> Path:
    p_dir = tmp_path / "pseudos"
    p_dir.mkdir()
    (p_dir / "H.upf").touch()
    return p_dir

def test_dft_pipeline_with_mock_pw(mock_pw_script: Path, pseudo_dir: Path, tmp_path: Path) -> None:
    # 1. Arrange
    # Use the mock script as the command
    command = f"{PYTHON_EXE} {mock_pw_script}"

    params = {
        "command": command,
        "pseudo_dir": pseudo_dir,
        "pseudopotentials": {"H": "H.upf"},
        "kspacing": 0.04
    }

    # Run in a temp directory because ASE creates files
    # We can change CWD or rely on ASE to use CWD.
    # DFTManager.compute() uses ASE Atoms which usually runs in CWD.
    # It's safer to change directory.

    cwd = Path.cwd()
    try:
        import os
        os.chdir(tmp_path)

        manager = DFTManager(params)

        structure = Structure(
            positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]]), # H2
            cell=np.eye(3) * 10.0,
            species=["H", "H"]
        )

        # 2. Act
        result = manager.compute(structure)

        # 3. Assert
        # Check energy (mock returns -13.60567890 Ry)
        # 1 Ry = 13.605693009 eV (approx)
        # ASE converts Ry to eV.
        # -13.60567890 Ry * 13.6056980659...
        # Just check it is not None and negative.
        assert result.energy is not None
        assert result.energy < -100.0 # -13.6 Ry is about -185 eV

        # Check forces
        # Mock returns:
        # atom 1: 0.001 0.002 0.003 (Ry/au)
        # atom 2: -0.001 -0.002 -0.003
        # ASE converts units.
        assert result.forces is not None
        assert result.forces.shape == (2, 3)
        # Check rough values (Ry/au to eV/A is approx 25.7)
        # 0.001 * 25.7 = 0.0257
        assert np.allclose(result.forces[0], [0.0257, 0.0514, 0.0771], atol=0.1)

        # Check stress
        assert result.stress is not None
        assert result.stress.shape == (3, 3)

    finally:
        os.chdir(cwd)
