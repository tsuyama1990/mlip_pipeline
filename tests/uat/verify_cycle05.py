import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    HybridPotentialType,
)


def main() -> None:
    print("Starting UAT for Cycle 05: Dynamics Engine & Active Learning Loop")  # noqa: T201

    work_dir = Path("./test_uat_cycle05")
    if work_dir.exists():
        import shutil
        shutil.rmtree(work_dir)

    # Create config with LAMMPS Dynamics
    config = GlobalConfig(
        orchestrator=OrchestratorConfig(
            max_cycles=2, # Need cycle 1 for OTF
            work_dir=work_dir,
            execution_mode=ExecutionMode.MOCK,
            cleanup_on_exit=False, # Inspect files
            max_candidates=10,
        ),
        generator=GeneratorConfig(
            type=GeneratorType.MOCK,
            mock_count=2, # Small number
        ),
        oracle=OracleConfig(type=OracleType.MOCK),
        trainer=TrainerConfig(
            type=TrainerType.MOCK,
            mock_potential_content="MOCK POTENTIAL",
        ),
        dynamics=DynamicsConfig(
            type=DynamicsType.LAMMPS, # Test real driver logic
            hybrid_potential=HybridPotentialType.ZBL,
            steps=100,
            halt_on_uncertainty=True,
            max_gamma_threshold=5.0,
        ),
        validator=ValidatorConfig(type=ValidatorType.MOCK),
    )

    # Patch subprocess.run to simulate LAMMPS execution
    # Patch only for LAMMPSDriver to avoid affecting other components if they use subprocess
    # But here other components are Mock.

    with patch("subprocess.run") as mock_run:
        # Define side effect to create dump file and simulate halt
        def side_effect(cmd, cwd=None, **kwargs):
            # Check if this is LAMMPS command (list of strings)
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "lmp" in cmd_str or "lammps" in cmd_str:
                # print(f"Mocking LAMMPS execution in {cwd}") # noqa: T201

                # Create dummy dump file
                dump_file = cwd / "traj.dump"

                # Create dummy content manually to ensure custom column c_pace[1]
                content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z c_pace[1]
1 1 0.0 0.0 0.0 6.0
2 1 2.0 0.0 0.0 6.0
"""
                with open(dump_file, "w") as f:
                    f.write(content)

                # Return halt code 100
                return MagicMock(returncode=100)

            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        # Initialize Orchestrator
        orchestrator = Orchestrator(config)

        # Run
        orchestrator.run()

    # Verification
    # Check if halt logic was triggered
    # The orchestrator logs "Halt triggered".
    # We can check if cycle 1 generated OTF candidates.

    state = orchestrator.state_manager.state
    assert state.current_cycle == 2

    # Check if MD run directories exist
    md_runs = list(work_dir.glob("md_run_*"))
    if not md_runs:
        print("FAILURE: No MD run directories found.") # noqa: T201
        sys.exit(1)

    print(f"Found {len(md_runs)} MD runs.") # noqa: T201

    # Verify input script contains ZBL
    input_script = md_runs[0] / "in.md"
    content = input_script.read_text()
    if "pair_style hybrid/overlay" not in content:
        print("FAILURE: ZBL hybrid overlay not found in input script.") # noqa: T201
        sys.exit(1)

    if "fix halt_check" not in content:
        print("FAILURE: Fix halt not found in input script.") # noqa: T201
        sys.exit(1)

    print("UAT Cycle 05 completed successfully")  # noqa: T201


if __name__ == "__main__":
    main()
