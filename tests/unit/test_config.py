from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import GlobalConfig


def test_config_valid(tmp_path: Path) -> None:
    components: dict[str, dict[str, Any]] = {
        "generator": {},
        "oracle": {},
        "trainer": {},
        "dynamics": {},
        "validator": {},
    }

    config = GlobalConfig(
        workdir=tmp_path, max_cycles=10, logging_level="INFO", components=components
    )
    assert config.max_cycles == 10


def test_config_missing_components(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Missing component configurations"):
        GlobalConfig(
            workdir=tmp_path,
            max_cycles=10,
            components={"generator": {}},  # Missing others
        )
