import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.exploration import ExplorationMethod, ExplorationTask


def test_exploration_method_values() -> None:
    assert ExplorationMethod.MD.value == "molecular_dynamics"
    assert ExplorationMethod.STATIC.value == "static_displacement"
    assert ExplorationMethod.AKMC.value == "adaptive_kmc"


def test_exploration_task_valid() -> None:
    task = ExplorationTask(
        method=ExplorationMethod.MD,
        parameters={"temp": 300},
        modifiers=["strain"],
    )
    assert task.method == ExplorationMethod.MD
    assert task.parameters["temp"] == 300
    assert "strain" in task.modifiers


def test_exploration_task_defaults() -> None:
    task = ExplorationTask(method=ExplorationMethod.STATIC)
    assert task.parameters == {}
    assert task.modifiers == []


def test_exploration_task_invalid_method() -> None:
    with pytest.raises(ValidationError):
        ExplorationTask(method="invalid")  # type: ignore[arg-type]


def test_exploration_task_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        ExplorationTask(method=ExplorationMethod.STATIC, extra_field="bad")  # type: ignore[call-arg]
