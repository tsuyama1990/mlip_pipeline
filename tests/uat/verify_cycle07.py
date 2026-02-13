import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from ase import Atoms

# We import EONDriver after setting up paths, but inside the test function to handle ImportError if not implemented yet
# or we use ComponentFactory.
from mlip_autopipec.core.factory import ComponentFactory
from mlip_autopipec.domain_models.config import (
    ActiveLearningConfig,
    DynamicsConfig,
    EONConfig,
    GeneratorConfig,
    GlobalConfig,
    OracleConfig,
    OrchestratorConfig,
    SystemConfig,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.domain_models.datastructures import Potential, Structure
from mlip_autopipec.domain_models.enums import DynamicsType, ExecutionMode


def test_uat_cycle07_eon_integration() -> None:
    print("Starting UAT Cycle 07: EON Integration")  # noqa: T201

    work_dir = Path("uat_work_dir_cycle07")
    work_dir.mkdir(exist_ok=True)

    # 1. Configure EON
    eon_conf = EONConfig(
        temperature=300.0,
        search_method="akmc",
        client_path="mock_eon_client",
        server_script_name="potential_server.py"
    )

    config = GlobalConfig(
        orchestrator=OrchestratorConfig(work_dir=work_dir, execution_mode=ExecutionMode.MOCK),
        generator=GeneratorConfig(),
        oracle=OracleConfig(),
        trainer=TrainerConfig(),
        dynamics=DynamicsConfig(type=DynamicsType.EON, eon=eon_conf),
        validator=ValidatorConfig(),
        system=SystemConfig(),
        active_learning=ActiveLearningConfig()
    )

    # 2. Use Factory to create EONDriver
    factory = ComponentFactory(config)
    try:
        driver = factory.create_dynamics(work_dir)
    except ValueError as e:
        print(f"Factory failed as expected (not implemented): {e}")
        # Fail the test if we expect it to work (Red phase expects failure, but verify script should crash)
        # However, UAT usually runs against installed package.
        # Since we are in dev, this imports from src.
        # If factory is not updated, it raises ValueError.
        raise

    print(f"Created Dynamics Driver: {type(driver)}")

    # 3. Mock dependencies for Simulate
    potential = Potential(path=Path("dummy.yace"), format="yace")
    structure = Structure(
        atoms=Atoms("Cu", positions=[[0,0,0]], cell=[5,5,5], pbc=True),
        provenance="seed"
    )

    # 4. Mock Subprocess (EON execution)
    # We need to mock subprocess.run to avoid actual execution
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        # We also need to mock file reading since driver will try to read results
        # Assuming parse_results reads some file

        # Create a smarter exists mock that returns True generally but maybe we handle directories specifically?
        # Simpler: Mock rmtree to avoid failure when we force exists=True
        with patch("shutil.rmtree"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("ase.io.read") as mock_read, \
             patch("builtins.open", new_callable=MagicMock), \
             patch("mlip_autopipec.dynamics.eon_driver.write"), \
             patch("shutil.copy"): # Mock copy too

             # Mock reading results
             # Run simulation
             results = list(driver.simulate(potential, structure))

    print(f"Simulation returned {len(results)} frames")

    # 5. Verify EON command
    mock_run.assert_called()
    args = mock_run.call_args[0][0]
    print(f"Called command: {args}")

    assert "mock_eon_client" in args[0] or "mock_eon_client" in args

    print("UAT Cycle 07 PASSED")

if __name__ == "__main__":
    test_uat_cycle07_eon_integration()
