import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.exploration import (
    AKMCTask,
    ExplorationMethod,
    MDTask,
    StaticParameters,
    StaticTask,
)


def test_exploration_method_values() -> None:
    assert ExplorationMethod.MD.value == "molecular_dynamics"
    assert ExplorationMethod.STATIC.value == "static_displacement"
    assert ExplorationMethod.AKMC.value == "adaptive_kmc"


def test_static_task_valid() -> None:
    task = StaticTask(
        modifiers=["strain"],
        parameters=StaticParameters(strain_range=0.1)
    )
    assert task.method == ExplorationMethod.STATIC
    assert task.parameters.strain_range == 0.1
    assert "strain" in task.modifiers


def test_md_task_defaults() -> None:
    task = MDTask()
    assert task.method == ExplorationMethod.MD
    # MDParameters default values
    assert task.parameters.local_displacement_range == 0.05
    assert task.parameters.local_sampling_count == 5


def test_akmc_task_valid() -> None:
    task = AKMCTask()
    assert task.method == ExplorationMethod.AKMC


def test_static_task_invalid_field() -> None:
    with pytest.raises(ValidationError):
        StaticTask(extra_field="bad")
