from pathlib import Path

import pytest

from mlip_autopipec.infrastructure.mocks import MockTrainer


def test_mock_trainer_path_traversal(tmp_path: Path) -> None:
    """
    Test that MockTrainer raises ValueError when trying to write outside the configured workdir.
    """
    # Create a safe root dir for CWD simulation
    safe_root = tmp_path / "safe_root"
    safe_root.mkdir()

    # We will simulate CWD being inside safe_root
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(safe_root)

    # Initialize trainer
    # params passed to init are stored in self.params but train takes separate params
    trainer = MockTrainer(params={})

    # Attempt to write to a path outside via traversal (../malicious)
    # This resolves to safe_root.parent / "malicious"
    # safe_root.parent is tmp_path.
    # tmp_path is usually inside /tmp/pytest-of-user/...

    # IMPORTANT: The MockTrainer allows paths inside /tmp.
    # So if we are running in /tmp (which pytest usually does), we need to be careful.

    # If safe_root is /tmp/pytest/safe_root, then ../malicious is /tmp/pytest/malicious.
    # This IS relative to /tmp. So it might PASS the check for /tmp!

    # To properly test the failure, we need a path that is NOT in /tmp either.
    # e.g. /opt/malicious (absolute path)

    malicious_abs_path = Path("/opt/malicious_dir")

    # But wait, if we run as root, we might have permissions, but MockTrainer should block it logic-wise.

    with pytest.raises(ValueError, match="Security Violation"):
        trainer.train([], params={}, workdir=malicious_abs_path)

def test_mock_trainer_safe_path(tmp_path: Path) -> None:
    """Test that a safe path inside CWD is allowed."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)

    workdir_name = "safe_dir"
    workdir = tmp_path / workdir_name
    # train creates the dir

    trainer = MockTrainer(params={})

    # Should not raise
    trainer.train([], params={}, workdir=workdir)

    assert (workdir / "potential.yace").exists()
