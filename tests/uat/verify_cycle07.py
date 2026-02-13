from pathlib import Path
from unittest.mock import MagicMock, patch

from ase import Atoms

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
from mlip_autopipec.domain_models.enums import DynamicsType, ExecutionMode
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure


def test_uat_cycle07_eon_integration() -> None:
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
    driver = factory.create_dynamics(work_dir)

    # 3. Mock dependencies for Simulate
    potential = Potential(path=Path("dummy.yace"), format="yace")
    structure = Structure(
        atoms=Atoms("Cu", positions=[[0,0,0]], cell=[5,5,5], pbc=True),
        provenance="seed"
    )

    # 4. Mock Subprocess (EON execution)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        # Create a smarter exists mock that returns True generally but maybe we handle directories specifically?
        # Simpler: Mock rmtree to avoid failure when we force exists=True
        with patch("shutil.rmtree"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("ase.io.read") as mock_read, \
             patch("builtins.open", new_callable=MagicMock), \
             patch("mlip_autopipec.dynamics.eon_driver.write"), \
             patch("shutil.copy"):

             mock_read.return_value = structure.atoms
             # Run simulation
             _ = list(driver.simulate(potential, structure))

    # 5. Verify EON command
    mock_run.assert_called()
    args = mock_run.call_args[0][0]

    assert "mock_eon_client" in args[0] or "mock_eon_client" in args

if __name__ == "__main__":
    test_uat_cycle07_eon_integration()
