import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import (
    ComponentsConfig,
    EONDynamicsConfig,
    LAMMPSDynamicsConfig,
)
from mlip_autopipec.domain_models.enums import DynamicsType


def test_lammps_config_valid():
    config = LAMMPSDynamicsConfig(
        name=DynamicsType.LAMMPS,
        timestep=0.001,
        n_steps=5000,
        temperature=300.0,
        pressure=0.0,
        thermo_freq=100,
        uncertainty_threshold=0.5,
    )
    assert config.name == DynamicsType.LAMMPS
    assert config.n_steps == 5000
    assert config.uncertainty_threshold == 0.5


def test_lammps_config_default():
    config = LAMMPSDynamicsConfig(name=DynamicsType.LAMMPS)
    assert config.timestep == 0.001
    assert config.uncertainty_threshold == 5.0


def test_eon_config_valid():
    config = EONDynamicsConfig(
        name=DynamicsType.EON,
        temperature=500.0,
        n_events=2000,
        supercell=[2, 2, 2],
        prefactor=1e13,
        uncertainty_threshold=1.0,
    )
    assert config.name == DynamicsType.EON
    assert config.temperature == 500.0
    assert config.supercell == [2, 2, 2]


def test_eon_config_default():
    config = EONDynamicsConfig(name=DynamicsType.EON)
    assert config.temperature == 300.0
    assert config.supercell == [1, 1, 1]
    assert config.uncertainty_threshold == 5.0


def test_dynamics_union():
    # Test that we can parse into the Union type
    # Direct instantiation of Union isn't how Pydantic works,
    # but we can simulate it via a container model or TypeAdapter if available.
    # Here we can just check if ComponentsConfig accepts it.
    pass  # We test this via ComponentsConfig below


def test_components_config_with_eon():
    # Minimal valid components config
    data = {
        "generator": {
            "name": "mock",
            "n_structures": 10,
            "cell_size": 10.0,
            "n_atoms": 2,
            "atomic_numbers": [1, 1],
        },
        "oracle": {"name": "mock"},
        "trainer": {"name": "mock"},
        "dynamics": {
            "name": "eon",
            "temperature": 600.0,
            "n_events": 500,
            "uncertainty_threshold": 2.0,
        },
        "validator": {"name": "mock"},
    }
    config = ComponentsConfig(**data)
    assert isinstance(config.dynamics, EONDynamicsConfig)
    assert config.dynamics.temperature == 600.0


def test_extra_forbid():
    with pytest.raises(ValidationError):
        EONDynamicsConfig(name=DynamicsType.EON, extra_field="bad")
