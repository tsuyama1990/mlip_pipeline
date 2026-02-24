"""Unit tests for PipelineState domain model."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.state import PipelineState


def test_pipeline_state_creation():
    """Test valid PipelineState creation."""
    state = PipelineState(
        current_step=1,
        completed_steps=[],
        artifacts={},
    )
    assert state.current_step == 1
    assert state.completed_steps == []
    assert state.artifacts == {}


def test_pipeline_state_validation():
    """Test PipelineState validation constraints."""
    # Test valid steps
    for i in range(1, 9): # 1-8 allowed
        PipelineState(current_step=i)

    # Test invalid steps
    with pytest.raises(ValidationError):
        PipelineState(current_step=0)

    with pytest.raises(ValidationError):
        PipelineState(current_step=9)


def test_pipeline_state_defaults():
    """Test default values."""
    state = PipelineState(current_step=1)
    assert state.completed_steps == []
    assert state.artifacts == {}
    assert state.metadata == {}


def test_pipeline_state_artifacts_path():
    """Test artifacts store Path objects correctly."""
    path = Path("/tmp/test.txt")
    state = PipelineState(
        current_step=1,
        artifacts={"dataset": path}
    )
    assert isinstance(state.artifacts["dataset"], Path)
    assert state.artifacts["dataset"] == path


def test_pipeline_state_serialization():
    """Test JSON serialization and deserialization."""
    state = PipelineState(
        current_step=3,
        completed_steps=[1, 2],
        artifacts={"step1": Path("step1.data")},
        metadata={"iteration": 5}
    )

    # Serialize
    json_str = state.model_dump_json()

    # Deserialize
    loaded = PipelineState.model_validate_json(json_str)

    assert loaded.current_step == 3
    assert loaded.completed_steps == [1, 2]
    assert loaded.artifacts["step1"] == Path("step1.data")
    assert loaded.metadata["iteration"] == 5


def test_pipeline_state_extra_forbid():
    """Test that extra fields are forbidden."""
    with pytest.raises(ValidationError) as excinfo:
        PipelineState(current_step=1, extra_field="bad")

    assert "extra_forbidden" in str(excinfo.value)
    assert "Extra inputs are not permitted" in str(excinfo.value)
