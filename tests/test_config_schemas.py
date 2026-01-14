# ruff: noqa: D100, D103
import pytest
from pydantic import ValidationError

from mlip_autopipec.config_schemas import (
    AlloyParams,
    CrystalParams,
    GeneratorParams,
    TargetSystem,
)


def test_target_system_valid_data() -> None:
    """Test that TargetSystem model validates with correct data."""
    data = {"elements": ["Ni", "Al"], "composition": {"Ni": 0.5, "Al": 0.5}}
    system = TargetSystem(**data)
    assert system.elements == ["Ni", "Al"]
    assert system.composition == {"Ni": 0.5, "Al": 0.5}


def test_target_system_composition_sum_error() -> None:
    """Test that composition fractions not summing to 1.0 raises ValueError."""
    data = {"elements": ["Ni", "Al"], "composition": {"Ni": 0.5, "Al": 0.6}}
    with pytest.raises(ValidationError, match="Composition fractions must sum to 1.0"):
        TargetSystem(**data)


def test_target_system_composition_elements_mismatch_error() -> None:
    """Test that mismatch between elements and composition keys raises ValueError."""
    data = {"elements": ["Ni", "Co"], "composition": {"Ni": 0.5, "Al": 0.5}}
    with pytest.raises(
        ValidationError, match="Elements in composition must match the elements list"
    ):
        TargetSystem(**data)


def test_alloy_params_valid() -> None:
    """Test that valid AlloyParams data is parsed correctly."""
    valid_data = {
        "sqs_supercell_size": [3, 3, 3],
        "strain_magnitudes": [0.9, 1.0, 1.1],
        "rattle_std_devs": [0.05, 0.1],
    }
    params = AlloyParams(**valid_data)
    assert params.sqs_supercell_size == [3, 3, 3]
    assert params.strain_magnitudes == [0.9, 1.0, 1.1]
    assert params.rattle_std_devs == [0.05, 0.1]


def test_alloy_params_invalid_supercell() -> None:
    """Test that an invalid sqs_supercell_size raises a ValidationError."""
    with pytest.raises(ValidationError):
        AlloyParams(sqs_supercell_size=[2, 2])  # Too short


def test_crystal_params_valid() -> None:
    """Test that valid CrystalParams data is parsed correctly."""
    valid_data = {"defect_types": ["vacancy", "interstitial"]}
    params = CrystalParams(**valid_data)
    assert params.defect_types == ["vacancy", "interstitial"]


def test_crystal_params_invalid_defect_type() -> None:
    """Test that an invalid defect_type raises a ValidationError."""
    with pytest.raises(ValidationError):
        CrystalParams(defect_types=["vacancy", "dislocation"])  # Invalid literal


def test_generator_params_composition() -> None:
    """Test that GeneratorParams correctly composes Alloy and Crystal params."""
    valid_data = {
        "alloy": {"sqs_supercell_size": [4, 4, 4]},
        "crystal": {"defect_types": ["interstitial"]},
    }
    params = GeneratorParams(**valid_data)
    assert params.alloy.sqs_supercell_size == [4, 4, 4]
    assert params.crystal.defect_types == ["interstitial"]
    # Check that defaults are still applied for other fields
    assert params.alloy.strain_magnitudes == [0.95, 1.0, 1.05]


def test_generator_params_defaults() -> None:
    """Test that GeneratorParams uses default factories correctly."""
    params = GeneratorParams()
    assert isinstance(params.alloy, AlloyParams)
    assert isinstance(params.crystal, CrystalParams)
    assert params.alloy.sqs_supercell_size == [2, 2, 2]
    assert params.crystal.defect_types == ["vacancy"]
