import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import (
    DynamicsConfig,
    GeneratorConfig,
    GlobalConfig,
    OracleConfig,
    OrchestratorConfig,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.domain_models.enums import (
    DynamicsType,
    ExecutionMode,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)


def main() -> None:
    print("Starting UAT for Cycle 01")  # noqa: T201

    # Create a mock config
    config = GlobalConfig(
        orchestrator=OrchestratorConfig(
            max_cycles=1,
            work_dir=Path("./test_uat_output"),
            execution_mode=ExecutionMode.MOCK,
            cleanup_on_exit=True
        ),
        generator=GeneratorConfig(type=GeneratorType.MOCK),
        oracle=OracleConfig(type=OracleType.MOCK),
        trainer=TrainerConfig(type=TrainerType.MOCK),
        dynamics=DynamicsConfig(type=DynamicsType.MOCK),
        validator=ValidatorConfig(type=ValidatorType.MOCK),
    )

    # Initialize Orchestrator
    orchestrator = Orchestrator(config)

    # Run
    orchestrator.run()

    print("UAT Cycle 01 completed successfully")  # noqa: T201

if __name__ == "__main__":
    main()
