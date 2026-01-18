from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.inference import (
    EmbeddingConfig,
    InferenceConfig,
    InferenceResult,
)


def test_inference_config_valid(tmp_path: Path) -> None:
    potential_file = tmp_path / "model.yace"
    potential_file.touch()

    config = InferenceConfig(
        temperature=300.0,
        potential_path=potential_file
    )
    assert config.ensemble == "nvt"
    assert config.steps == 10000
    assert config.uq_threshold == 5.0

def test_inference_config_invalid_potential_path(tmp_path: Path) -> None:
    # Explicit check for non-existent file
    with pytest.raises(ValidationError) as excinfo:
        InferenceConfig(
            temperature=300.0,
            potential_path=tmp_path / "non_existent.yace"
        )
    assert "Potential file" in str(excinfo.value) or "Path does not point to a file" in str(excinfo.value)

def test_inference_config_negative_temp(tmp_path: Path) -> None:
    potential_file = tmp_path / "model.yace"
    potential_file.touch()
    with pytest.raises(ValidationError):
        InferenceConfig(
            temperature=-10.0,
            potential_path=potential_file
        )

def test_inference_result_valid() -> None:
    res = InferenceResult(
        succeeded=True,
        final_structure=Path("final.xyz"),
        uncertain_structures=[Path("dump_1.xyz")],
        max_gamma_observed=1.2
    )
    assert res.succeeded is True
    assert len(res.uncertain_structures) == 1

def test_inference_result_negative_gamma() -> None:
    with pytest.raises(ValidationError):
        InferenceResult(
            succeeded=True,
            max_gamma_observed=-1.0
        )

def test_embedding_config_valid() -> None:
    config = EmbeddingConfig(core_radius=3.0, buffer_width=1.5)
    assert config.core_radius == 3.0
    assert config.buffer_width == 1.5
    assert config.box_size == 9.0  # 2 * (3.0 + 1.5)

def test_embedding_config_defaults() -> None:
    config = EmbeddingConfig()
    assert config.core_radius == 4.0
    assert config.buffer_width == 2.0
    assert config.box_size == 12.0

def test_embedding_config_invalid() -> None:
    with pytest.raises(ValidationError):
        EmbeddingConfig(core_radius=-1.0)
