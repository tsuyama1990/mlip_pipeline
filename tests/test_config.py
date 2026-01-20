import pytest
from pydantic import ValidationError

from mlip_autopipec.config.models import (
    Composition,
    MinimalConfig,
    Resources,
    SystemConfig,
    TargetSystem,
)


def test_minimal_config_valid():
    """Test creating a valid MinimalConfig."""
    config = MinimalConfig(
        project_name="TestProject",
        target_system=TargetSystem(
            elements=["Al", "Cu"], composition=Composition({"Al": 0.5, "Cu": 0.5})
        ),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=4),
    )
    assert config.project_name == "TestProject"
    assert config.target_system.elements == ["Al", "Cu"]


def test_minimal_config_invalid_composition_sum_low():
    """Test that composition validation fails if sum < 1.0."""
    with pytest.raises(ValidationError) as excinfo:
        MinimalConfig(
            project_name="TestProject",
            target_system=TargetSystem(elements=["Al"], composition=Composition({"Al": 0.9})),
            resources=Resources(dft_code="quantum_espresso", parallel_cores=4),
        )
    assert "Composition fractions must sum to 1.0" in str(excinfo.value)


def test_minimal_config_invalid_composition_sum_high():
    """Test that composition validation fails if sum > 1.0."""
    with pytest.raises(ValidationError) as excinfo:
        MinimalConfig(
            project_name="TestProject",
            target_system=TargetSystem(
                elements=["Al", "Cu"], composition=Composition({"Al": 0.6, "Cu": 0.5})
            ),
            resources=Resources(dft_code="quantum_espresso", parallel_cores=4),
        )
    assert "Composition fractions must sum to 1.0" in str(excinfo.value)


def test_minimal_config_invalid_composition_type():
    """Test that composition values must be floats."""
    with pytest.raises(ValidationError):
        MinimalConfig(
            project_name="TestProject",
            target_system=TargetSystem(
                elements=["Al"],
                composition=Composition({"Al": "half"}),  # type: ignore
            ),
            resources=Resources(dft_code="quantum_espresso", parallel_cores=4),
        )


def test_minimal_config_invalid_element():
    """Test invalid chemical symbol."""
    with pytest.raises(ValidationError) as excinfo:
        MinimalConfig(
            project_name="TestProject",
            target_system=TargetSystem(elements=["Xy"], composition=Composition({"Xy": 1.0})),
            resources=Resources(dft_code="quantum_espresso", parallel_cores=4),
        )
    assert "not a valid chemical symbol" in str(excinfo.value)


def test_minimal_config_element_mismatch():
    """Test mismatch between elements list and composition keys."""
    with pytest.raises(ValidationError) as excinfo:
        MinimalConfig(
            project_name="TestProject",
            target_system=TargetSystem(
                elements=["Al", "Cu"],
                composition=Composition({"Al": 1.0}),  # Missing Cu
            ),
            resources=Resources(dft_code="quantum_espresso", parallel_cores=4),
        )
    assert "Composition keys must match the elements list" in str(excinfo.value)


def test_resources_positive_cores():
    """Test that parallel_cores must be positive."""
    with pytest.raises(ValidationError) as excinfo:
        MinimalConfig(
            project_name="TestProject",
            target_system=TargetSystem(elements=["Al"], composition=Composition({"Al": 1.0})),
            resources=Resources(dft_code="quantum_espresso", parallel_cores=0),
        )
    assert "Input should be greater than 0" in str(excinfo.value)


def test_system_config_immutability(tmp_path):
    """Test that SystemConfig is immutable."""
    minimal = MinimalConfig(
        project_name="TestProject",
        target_system=TargetSystem(elements=["Al"], composition=Composition({"Al": 1.0})),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=4),
    )

    config = SystemConfig(
        minimal=minimal,
        working_dir=tmp_path / "work",
        db_path=tmp_path / "work" / "db.db",
        log_path=tmp_path / "work" / "log.log",
    )

    with pytest.raises(ValidationError):
        config.project_name = "NewName"
