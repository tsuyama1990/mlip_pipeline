from pathlib import Path
from unittest.mock import patch

from ase.atoms import Atoms

from mlip_autopipec.inference.uq import UncertaintyChecker


def test_parse_dump_empty(tmp_path: Path) -> None:
    dump_file = tmp_path / "dump.gamma"
    dump_file.touch()

    checker = UncertaintyChecker(uq_threshold=5.0)
    uncertain_atoms = checker.parse_dump(dump_file)
    assert len(uncertain_atoms) == 0


def test_parse_dump_with_atoms(tmp_path: Path) -> None:
    dump_file = tmp_path / "dump.gamma"
    # Write dummy content so size > 0
    dump_file.write_text("dummy content")

    atoms1 = Atoms("Al", positions=[[0, 0, 0]], cell=[3, 3, 3])
    atoms1.info["timestep"] = 100

    # Patch ase.io.read to return list
    with patch("ase.io.read") as mock_read:
        mock_read.return_value = [atoms1]

        checker = UncertaintyChecker(uq_threshold=5.0)
        uncertain_atoms = checker.parse_dump(dump_file)

        assert len(uncertain_atoms) == 1
        assert uncertain_atoms[0].info["src_md_step"] == 100


def test_parse_dump_logic(tmp_path: Path) -> None:
    dump_file = tmp_path / "dump.gamma"
    dump_file.write_text("dummy content")

    atoms_high = Atoms("Al", positions=[[0, 0, 0]])
    atoms_high.info = {"timestep": 100}

    with patch("ase.io.read") as mock_read:
        mock_read.return_value = [atoms_high]

        checker = UncertaintyChecker(uq_threshold=5.0)
        uncertain = checker.parse_dump(dump_file)

        assert len(uncertain) == 1
        assert uncertain[0].info["src_md_step"] == 100
