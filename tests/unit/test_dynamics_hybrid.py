from pathlib import Path

from mlip_autopipec.components.dynamics.hybrid import generate_pair_style
from mlip_autopipec.domain_models.config import PhysicsBaselineConfig
from mlip_autopipec.domain_models.potential import Potential


def test_generate_pair_style_zbl(tmp_path: Path) -> None:
    potential_path = tmp_path / "test.yace"
    potential_path.touch()
    potential = Potential(path=potential_path, format="yace", species=["Cu", "Au"])
    baseline = PhysicsBaselineConfig(type="zbl")

    pair_style, pair_coeff = generate_pair_style(potential, baseline)

    # Check pair_style
    assert "hybrid/overlay" in pair_style
    assert "pace" in pair_style
    assert "zbl" in pair_style

    # Check pair_coeff
    # Expect lines for pace and lines for zbl
    # pace: pair_coeff * * pace /tmp/test.yace Cu Au
    # zbl: pair_coeff 1 1 zbl 29 29 ...

    lines = pair_coeff.split("\n")
    pace_line = next(line for line in lines if "pace" in line)
    assert str(potential_path) in pace_line
    assert "Cu Au" in pace_line

    from ase.data import atomic_numbers

    # Check ZBL lines (Cu=29, Au=79)
    zbl_lines = [line for line in lines if "zbl" in line]
    assert len(zbl_lines) >= 3  # 1-1, 1-2, 2-2

    z_cu = atomic_numbers["Cu"]
    z_au = atomic_numbers["Au"]

    # Check atomic numbers are correct
    # 1 1 -> 29 29
    # 1 2 -> 29 79
    # 2 2 -> 79 79
    assert any(f"1 1 zbl {z_cu} {z_cu}" in line for line in zbl_lines)
    assert any(f"1 2 zbl {z_cu} {z_au}" in line for line in zbl_lines)
    assert any(f"2 2 zbl {z_au} {z_au}" in line for line in zbl_lines)


def test_generate_pair_style_no_baseline(tmp_path: Path) -> None:
    potential_path = tmp_path / "test.yace"
    potential_path.touch()
    potential = Potential(path=potential_path, format="yace", species=["Cu"])
    pair_style, pair_coeff = generate_pair_style(potential, None)

    assert pair_style == "pair_style pace"
    assert pair_coeff == f"pair_coeff * * pace {potential_path} Cu"
