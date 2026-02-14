"""Tests for BaseModule."""

from typing import Any

from pyacemaker.core.base import BaseModule
from pyacemaker.core.config import PYACEMAKERConfig


class ConcreteModule(BaseModule):
    """Concrete implementation of BaseModule for testing."""

    def run(self) -> dict[str, Any]:
        """Run method."""
        self.logger.info("Running module")
        return {"status": "success"}


def test_base_module_initialization() -> None:
    """Test BaseModule initialization and logger setup."""
    config_data = {
        "project": {"name": "Test", "root_dir": "."},
        "dft": {"code": "qe"},
    }
    config = PYACEMAKERConfig(**config_data)  # type: ignore[arg-type]

    module = ConcreteModule(config)

    assert module.config == config
    assert module.logger is not None

    # Verify run execution
    result = module.run()
    assert result == {"status": "success"}
