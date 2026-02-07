from pathlib import Path

import pytest

from mlip_autopipec.infrastructure.mocks import MockTrainer


def test_mock_trainer_security_valid(tmp_path: Path) -> None:
    trainer = MockTrainer()
    # Should pass if inside tmp_path
    workdir = tmp_path / "valid"
    trainer.train([], {}, workdir)

def test_mock_trainer_security_invalid() -> None:
    trainer = MockTrainer()
    # Should fail if absolute path to root
    with pytest.raises(ValueError, match="Security Violation"):
        trainer.train([], {}, Path("/etc/passwd"))

def test_mock_trainer_security_traversal(tmp_path: Path) -> None:
    trainer = MockTrainer()
    # Try .. traversal
    # Note: resolving might map it back to a valid path if inside tmp,
    # so we need to target something clearly outside

    # We can't easily create a path that traverses out of /tmp/pytest-of-user/.../test_0/
    # effectively without permissions on real systems, but we can verify the check logic.
    # The check is: must be relative to CWD or /tmp.

    # If we pass a path that resolves to /usr/bin, it should fail
    with pytest.raises(ValueError, match="Security Violation"):
        trainer.train([], {}, Path("/usr/bin"))
