from typing import Any

import pytest


@pytest.fixture
def valid_config_dict() -> dict[str, Any]:
    return {
        "project_name": "TestProject",
        "orchestrator": {
            "work_dir": "test_work_dir",
            "max_iterations": 10,
            "state_file": "workflow_state.json"
        },
        "generator": {
            "type": "MOCK",
            "enabled": True
        },
        "oracle": {
            "type": "MOCK",
            "enabled": True
        },
        "trainer": {
            "type": "MOCK",
            "enabled": False
        },
        "dynamics": {
            "type": "MOCK",
            "enabled": False
        }
    }
