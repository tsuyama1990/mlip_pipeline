import pytest

from mlip_autopipec.trainer.delta_learning import DeltaLearning


def test_delta_learning_zbl() -> None:
    elements = ["Fe", "Pt"]
    config_str = DeltaLearning.get_config(elements, "zbl")
    assert "pair_style: zbl" in config_str
    # Fe is 26, Pt is 78.
    # The string should likely contain the atomic numbers.
    assert "26" in config_str
    assert "78" in config_str

def test_delta_learning_lj() -> None:
    elements = ["Fe", "Pt"]
    config_str = DeltaLearning.get_config(elements, "lj")
    assert "pair_style: lj" in config_str

def test_delta_learning_none() -> None:
    elements = ["Fe", "Pt"]
    config_str = DeltaLearning.get_config(elements, None)
    assert config_str == ""

def test_delta_learning_invalid() -> None:
    elements = ["Fe"]
    with pytest.raises(ValueError, match="Unsupported delta learning baseline"):
        DeltaLearning.get_config(elements, "invalid_potential")
