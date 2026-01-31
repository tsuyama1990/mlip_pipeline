from pathlib import Path

from mlip_autopipec.domain_models.config import PotentialConfig
from mlip_autopipec.physics.validation.utils import get_validation_calculator


def test_get_validation_calculator(tmp_path: Path) -> None:
    pot_config = PotentialConfig(elements=["Si"], cutoff=5.0)
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    calc = get_validation_calculator(pot_path, pot_config, tmp_path)

    assert calc.parameters["specorder"] == ["Si"]
    assert "pace" in calc.parameters["pair_style"]
    assert len(calc.parameters["pair_coeff"]) == 1
    assert "pot.yace" in calc.parameters["pair_coeff"][0]


def test_get_validation_calculator_hybrid(tmp_path: Path) -> None:
    pot_config = PotentialConfig(
        elements=["Si", "C"],
        cutoff=5.0,
        pair_style="hybrid/overlay",
        zbl_inner_cutoff=0.5,
        zbl_outer_cutoff=1.5,
    )
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    calc = get_validation_calculator(pot_path, pot_config, tmp_path)

    assert "hybrid/overlay" in calc.parameters["pair_style"]
    assert "zbl" in calc.parameters["pair_style"]
    # check pair coefficients
    coeffs = calc.parameters["pair_coeff"]
    assert any("pace" in c for c in coeffs)
    assert any("zbl" in c for c in coeffs)
