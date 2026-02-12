import os
from pathlib import Path

import pytest

from mlip_autopipec.core.config_parser import load_config
from mlip_autopipec.domain_models.config import FullConfig


def test_config_parser_valid(tmp_path: Path) -> None:
    work_dir = tmp_path / "test_run"
    config_content = f"""
    orchestrator:
      work_dir: {work_dir}
      max_iterations: 5
    generator:
      type: RANDOM
      num_structures: 10
    oracle:
      type: QUANTUM_ESPRESSO
      command: pw.x
    trainer:
      type: PACEMAKER
    """
    p = tmp_path / "config.yaml"
    p.write_text(config_content)

    conf = load_config(p)
    assert isinstance(conf, FullConfig)
    assert conf.orchestrator.max_iterations == 5
    assert conf.orchestrator.work_dir == work_dir


def test_config_parser_env_substitution(tmp_path: Path) -> None:
    os.environ["TEST_DIR"] = str(tmp_path)
    config_content = """
    orchestrator:
      work_dir: ${TEST_DIR}/run_01
    generator:
      type: RANDOM
    oracle:
      type: QUANTUM_ESPRESSO
      command: pw.x
    trainer:
      type: PACEMAKER
    """
    p = tmp_path / "config.yaml"
    p.write_text(config_content)

    conf = load_config(p)
    assert conf.orchestrator.work_dir == tmp_path / "run_01"


def test_config_parser_missing_env(tmp_path: Path) -> None:
    # Ensure strict mode? Usually KeyError if env var missing
    config_content = """
    orchestrator:
      work_dir: ${MISSING_VAR}/run_01
    generator:
      type: RANDOM
    oracle:
      type: QUANTUM_ESPRESSO
      command: pw.x
    trainer:
      type: PACEMAKER
    """
    p = tmp_path / "config.yaml"
    p.write_text(config_content)

    # Depending on implementation, might raise error or keep as is.
    # We'll assume strict behavior or explicit error.
    with pytest.raises(ValueError, match="Environment variable"):
        load_config(p)
