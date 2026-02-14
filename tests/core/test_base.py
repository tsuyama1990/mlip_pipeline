"""Tests for BaseModule."""

from pyacemaker.core.base import BaseModule, ModuleResult
from pyacemaker.core.config import CONSTANTS, PYACEMAKERConfig


class ConcreteModule(BaseModule):
    """Concrete implementation of BaseModule for testing."""

    def run(self) -> ModuleResult:
        """Run method."""
        self.logger.info("Running module")
        return ModuleResult(status="success")


def test_base_module_initialization() -> None:
    """Test BaseModule initialization and logger setup."""
    # Skip file checks
    original_skip = CONSTANTS.skip_file_checks
    CONSTANTS.skip_file_checks = True

    try:
        config_data = {
            "version": "0.1.0",
            "project": {"name": "Test", "root_dir": "."},
            "oracle": {
                "dft": {
                    "code": "qe",
                    "pseudopotentials": {"Fe": "Fe.pbe.UPF"},
                    "parameters": {},
                },
                "mock": True,
            },
        }
        config = PYACEMAKERConfig(**config_data)  # type: ignore[arg-type]

        module = ConcreteModule(config)

        assert module.config == config
        assert module.logger is not None

        # Verify run execution
        result = module.run()
        assert result.status == "success"
    finally:
        CONSTANTS.skip_file_checks = original_skip
