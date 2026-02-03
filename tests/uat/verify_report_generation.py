"""
UAT Script: Verify Report Generation
"""

import shutil
import sys
from pathlib import Path

from mlip_autopipec.config.config_model import ValidationConfig
from mlip_autopipec.validation.runner import ValidationRunner


def run_uat() -> None:
    print("Starting UAT: Report Generation")  # noqa: T201

    # Setup work directory
    work_dir = Path("uat_output")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()

    # Dummy potential
    potential_path = work_dir / "dummy_potential.yace"
    potential_path.touch()

    # Config
    config = ValidationConfig(
        run_validation=True, check_phonons=True, check_elastic=True
    )

    runner = ValidationRunner(config)

    print("Running Validation...")  # noqa: T201
    result = runner.validate(potential_path, work_dir)

    print(f"Validation Finished. Passed: {result.passed}")  # noqa: T201
    print(f"Report Path: {result.report_path}")  # noqa: T201

    if result.report_path and result.report_path.exists():
        print("SUCCESS: Report generated.")  # noqa: T201
    else:
        print("FAILURE: Report not generated.")  # noqa: T201
        sys.exit(1)


if __name__ == "__main__":
    run_uat()
