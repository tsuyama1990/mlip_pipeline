"""Tests for configuration domain models."""

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import (
    AdaptivePolicyConfig,
    Config,
    ExplorationConfig,
    PotentialConfig,
    TrainerConfig,
)


def test_config_valid(temp_dir: object) -> None:
    """Test creating a valid configuration."""
    data = {
        "project_name": "test_project",
        "potential": {
            "elements": ["Si"],
            "cutoff": 5.0,
            "seed": 123
        },
        "logging": {
            "level": "DEBUG",
            "file_path": "test.log"
        },
        "structure_gen": {
            "strategy": "random",
            "policy": {
                "md_mc_ratio": 0.5
            }
        },
        "oracle": {
            "k_points_density": 0.05
        },
        "trainer": {
            "engine": "mace",
            "max_epochs": 200
        }
    }
    config = Config(**data)  # type: ignore[arg-type]
    assert config.project_name == "test_project"
    assert config.potential.cutoff == 5.0
    assert config.structure_gen.policy.md_mc_ratio == 0.5
    assert config.oracle.k_points_density == 0.05
    assert config.trainer.max_epochs == 200


def test_config_defaults() -> None:
    """Test default values."""
    config = Config(project_name="default_test")
    assert config.potential.cutoff == 5.0  # From default
    assert config.structure_gen.policy.md_mc_ratio == 0.1
    assert config.trainer.engine == "mace"
    assert config.oracle.code == "qe"


def test_config_invalid_cutoff() -> None:
    """Test invalid cutoff."""
    data = {
        "project_name": "test_project",
        "potential": {
            "elements": ["Si"],
            "cutoff": -1.0,  # Invalid
            "seed": 123
        }
    }
    with pytest.raises(ValidationError) as excinfo:
        Config(**data)  # type: ignore[arg-type]
    assert "Cutoff must be greater than 0" in str(excinfo.value)


def test_config_invalid_elements() -> None:
    """Test empty elements list."""
    with pytest.raises(ValidationError):
        PotentialConfig(elements=[], cutoff=5.0)


def test_config_invalid_project_name() -> None:
    """Test empty project name."""
    with pytest.raises(ValidationError):
        Config(
            project_name="  ",
            potential=PotentialConfig(elements=["Si"], cutoff=5.0)
        )

def test_adaptive_policy_validation() -> None:
    """Test validation for adaptive policy."""
    with pytest.raises(ValidationError):
        AdaptivePolicyConfig(md_mc_ratio=1.5)  # Must be <= 1

    with pytest.raises(ValidationError):
        AdaptivePolicyConfig(md_mc_ratio=-0.1)  # Must be >= 0

def test_trainer_validation() -> None:
    """Test validation for trainer."""
    with pytest.raises(ValidationError):
        TrainerConfig(max_epochs=0)

def test_exploration_config_valid() -> None:
    """Test valid exploration config."""
    config = ExplorationConfig(
        strategy="random",
        supercell_size=[2, 2, 2],
        rattle_amplitude=0.05,
        num_candidates=10,
        composition="Si2"
    )
    assert config.strategy == "random"
    assert config.supercell_size == [2, 2, 2]
    assert config.rattle_amplitude == 0.05
    assert config.num_candidates == 10
    assert config.composition == "Si2"

def test_exploration_config_invalid() -> None:
    """Test invalid exploration config."""
    # Invalid supercell
    with pytest.raises(ValidationError):
        ExplorationConfig(supercell_size=[1, 1]) # Wrong length

    with pytest.raises(ValidationError):
        ExplorationConfig(supercell_size=[0, 1, 1]) # Not positive

    # Invalid rattle
    with pytest.raises(ValidationError):
        ExplorationConfig(rattle_amplitude=-0.1)

    # Invalid candidates
    with pytest.raises(ValidationError):
        ExplorationConfig(num_candidates=0)
