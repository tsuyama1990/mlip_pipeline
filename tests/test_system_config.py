import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.system import WorkflowConfig


def test_workflow_config_checkpoint_extension():
    """Test validation of checkpoint filename extension."""
    with pytest.raises(ValidationError, match="must have a .json extension"):
        WorkflowConfig(checkpoint_filename="bad.txt")

    wc = WorkflowConfig(checkpoint_filename="good.json")
    assert wc.checkpoint_filename == "good.json"
