import shutil
import subprocess
import sys
import uuid
from pathlib import Path

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def setup_project(work_dir: Path):
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    # Create checkpoint
    from mlip_autopipec.config.models import (
        CheckpointState,
        SystemConfig,
        TrainingConfig,
        TrainingRunMetrics,
        WorkflowConfig,
    )

    metrics = [
        TrainingRunMetrics(generation=1, num_structures=100, rmse_forces=0.1, rmse_energy_per_atom=0.01),
        TrainingRunMetrics(generation=2, num_structures=130, rmse_forces=0.08, rmse_energy_per_atom=0.008),
        TrainingRunMetrics(generation=3, num_structures=155, rmse_forces=0.05, rmse_energy_per_atom=0.005),
    ]

    state = CheckpointState(
        run_uuid=uuid.uuid4(),
        system_config=SystemConfig(
            project_name="UAT Cycle 08 Project",
            run_uuid=uuid.uuid4(),
            workflow_config=WorkflowConfig(checkpoint_filename="checkpoint.json"),
            training_config=TrainingConfig(data_source_db=Path("project.db")),
        ),
        active_learning_generation=3,
        training_history=metrics,
        pending_job_ids=[uuid.uuid4() for _ in range(15)]
    )

    (work_dir / "checkpoint.json").write_text(state.model_dump_json())

    # Create DB
    from ase import Atoms
    from ase.db import connect
    db_path = work_dir / "project.db"
    with connect(db_path) as db:
        for _ in range(100):
            db.write(Atoms('Si'), data={'config_type': 'initial'})
        for _ in range(30):
            db.write(Atoms('Si'), data={'config_type': 'active_learning_gen1'})
        for _ in range(25):
            db.write(Atoms('Si'), data={'config_type': 'active_learning_gen2'})

def run_command(cmd, cwd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, cwd=cwd, capture_output=True, text=True)
    return result

def main():
    print("Starting UAT for Cycle 08: Monitoring and Usability")
    work_dir = Path("uat_cycle_08_workspace")

    try:
        # Scenario 1: Generating and Viewing the Status Dashboard
        print("\n--- UAT-C8-001: Generating and Viewing the Status Dashboard ---")
        setup_project(work_dir)

        # We need to mock webbrowser open so it doesn't fail in headless environment,
        # but since we run subprocess, we can't easily mock it unless we set an env var or argument.
        # The command has --no-open.

        # Test default (tries to open) - might fail or just print.
        # Actually in `app.py`, `webbrowser.open` is called. If no browser, it might just fail or do nothing.
        # To avoid side effects, we'll use --no-open for the automated part.

        # We will test generation first.
        res = run_command(["mlip-auto", "status", ".", "--no-open"], cwd=work_dir)

        if res.returncode == 0 and "Dashboard generated at" in res.stdout:
            print(f"{GREEN}✅ Dashboard generation successful.{RESET}")
        else:
            print(f"{RED}❌ Dashboard generation failed.{RESET}")
            print(res.stdout)
            print(res.stderr)
            sys.exit(1)

        dashboard_path = work_dir / "dashboard.html"
        if dashboard_path.exists():
             print(f"{GREEN}✅ dashboard.html exists.{RESET}")
        else:
             print(f"{RED}❌ dashboard.html not found.{RESET}")
             sys.exit(1)

        # Scenario 2: Interpreting Dashboard for Workflow Insights
        print("\n--- UAT-C8-002: Interpreting Dashboard for Workflow Insights ---")
        content = dashboard_path.read_text()

        checks = [
            ("UAT Cycle 08 Project", "Project Name"),
            ("Current Generation", "Generation Label"),
            ("3", "Generation Value"),
            ("Force RMSE vs. Generation", "RMSE Plot"),
            ("Dataset Composition", "Composition Plot"),
            ("155", "Completed Calculations"), # 100+30+25
        ]

        all_passed = True
        for text, desc in checks:
            if text in content:
                print(f"{GREEN}✅ Found {desc} ('{text}'){RESET}")
            else:
                print(f"{RED}❌ Missing {desc} ('{text}'){RESET}")
                all_passed = False

        if all_passed:
            print(f"\n{GREEN}🎉 Cycle 08 UAT Passed!{RESET}")
        else:
            print(f"\n{RED}Cycle 08 UAT Failed.{RESET}")
            sys.exit(1)

    finally:
        if work_dir.exists():
            shutil.rmtree(work_dir)

if __name__ == "__main__":
    main()
