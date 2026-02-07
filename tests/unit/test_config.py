from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.config_model import (
    ExplorerConfig,
    GlobalConfig,
    OracleConfig,
    TrainerConfig,
)


def test_valid_config(tmp_path: Path) -> None:
    work_dir = tmp_path / "test_work_dir"
    config = GlobalConfig(
        work_dir=work_dir,
        max_cycles=5,
        oracle=OracleConfig(type="mock"),
        trainer=TrainerConfig(type="mock"),
        explorer=ExplorerConfig(type="mock")
    )
    assert config.work_dir == work_dir
    assert config.max_cycles == 5
    assert config.oracle.type == "mock"

def test_invalid_config_missing_field() -> None:
    # Testing that work_dir is required
    with pytest.raises(ValidationError):
        GlobalConfig(max_cycles=5) # type: ignore[call-arg]

def test_invalid_config_type_mismatch(tmp_path: Path) -> None:
    work_dir = tmp_path / "test_work_dir"
    # Need to construct raw dict to trigger pydantic validation error on nested type
    # because constructing OracleConfig(type="invalid") would fail immediately if validated there,
    # but type is Literal.
    # However, if we pass a dict to GlobalConfig constructor, Pydantic parses it.

    # We must use kwargs or dict unpacking that MyPy accepts or ignore.
    # To satisfy MyPy, we should construct objects properly or use `type: ignore` for intentional invalid input testing.

    with pytest.raises(ValidationError):
        GlobalConfig(
            work_dir=work_dir,
            oracle={"type": "invalid_type"} # type: ignore[arg-type]
        )

def test_defaults(tmp_path: Path) -> None:
    work_dir = tmp_path / "test_work_dir"
    config = GlobalConfig(work_dir=work_dir)
    assert config.max_cycles == 10
    assert config.oracle.type == "mock"
    assert config.trainer.type == "mock"
    assert config.explorer.type == "mock"
