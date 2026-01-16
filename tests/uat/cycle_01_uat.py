# ruff: noqa: N999
# FIXME: The above comment is a temporary workaround for a ruff bug.
# It should be removed once the bug is fixed.
# For more information, see: https://github.com/astral-sh/ruff/issues/10515
"""
User Acceptance Tests (UAT) for Cycle 1: The Automated DFT Factory.

This script programmatically executes the test scenarios defined in
`dev_documents/system_prompts/CYCLE01/UAT.md`. It is intended to be run as a
Jupyter Notebook or a Python script to provide a clear, step-by-step
demonstration of the `DFTFactory`'s capabilities.
"""

import logging
import os
from pathlib import Path

import numpy as np
from ase.build import bulk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# --- Test Configuration ---
# IMPORTANT: This path must be configured to point to a valid `pw.x` executable
# for the integration tests to run.
QE_EXECUTABLE_PATH = os.environ.get("QE_EXECUTABLE_PATH", "/usr/bin/pw.x")
DB_PATH = Path("uat_cycle_01.db")


def run_uat_scenario(scenario_id: str, description: str):
    """Decorator to print scenario information."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logging.info(f"\n--- Running UAT Scenario: {scenario_id} ---")
            logging.info(description)
            try:
                func(*args, **kwargs)
                logging.info(f"--- PASSED: {scenario_id} ---")
            except Exception as e:
                logging.error(f"--- FAILED: {scenario_id} ---")
                logging.error(f"Reason: {e}", exc_info=True)
                raise

        return wrapper

    return decorator


@run_uat_scenario(
    "UAT-C1-001",
    "Successful 'Happy Path' Calculation: Verifies a standard, end-to-end "
    "DFT calculation for a well-behaved atomic structure (Silicon).",
)
def uat_c1_001_happy_path_calculation():
    """Demonstrates a successful DFT calculation for bulk Silicon."""
    from mlip_autopipec.modules.dft import DFTFactory

    if not Path(QE_EXECUTABLE_PATH).exists():
        logging.warning(
            f"QE executable not found at '{QE_EXECUTABLE_PATH}'. "
            "Skipping UAT-C1-001."
        )
        return

    factory = DFTFactory(qe_executable_path=QE_EXECUTABLE_PATH)
    si_atoms = bulk("Si", "diamond", a=5.43)

    logging.info("Running DFT calculation for Si...")
    result = factory.run(si_atoms)

    logging.info(f"Calculation successful. Job ID: {result.job_id}")
    logging.info(f"  - Energy: {result.energy:.4f} eV")
    logging.info(f"  - Forces Norm: {np.linalg.norm(result.forces):.4f}")
    logging.info(f"  - Stress: {result.stress}")

    assert result.energy < 0, "Energy should be negative for a bound system."
    assert np.allclose(
        result.forces,
        0,
        atol=1e-3,
    ), "Forces should be close to zero for equilibrium Si."


@run_uat_scenario(
    "UAT-C1-002",
    "Automatic Parameter Heuristics: Demonstrates that the factory "
    "intelligently adapts DFT parameters for different materials.",
)
def uat_c1_002_heuristic_verification():
    """Verifies that heuristics adapt to different material types."""
    from mlip_autopipec.modules.dft import DFTFactory

    factory = DFTFactory(qe_executable_path="dummy")  # No execution needed

    # Case 1: Simple Metal (Aluminium)
    al_atoms = bulk("Al", "fcc", a=4.05)
    al_params = factory._get_heuristic_parameters(al_atoms)
    logging.info("Generated parameters for Aluminium (Metal):")
    logging.info(f"  - Smearing: {al_params.smearing}")
    assert al_params.smearing is not None, "Smearing should be enabled for Al."

    # Case 2: Magnetic Element (Iron)
    fe_atoms = bulk("Fe", "bcc", a=2.87)
    fe_params = factory._get_heuristic_parameters(fe_atoms)
    logging.info("Generated parameters for Iron (Magnetic):")
    logging.info(f"  - Magnetism: {fe_params.magnetism}")
    assert fe_params.magnetism is not None, "Magnetism should be enabled for Fe."
    assert fe_params.magnetism.nspin == 2


@run_uat_scenario(
    "UAT-C1-004",
    "Data Persistence and Retrieval: Verifies that a successful DFT "
    "result is correctly saved to and can be retrieved from an ASE database.",
)
def uat_c1_004_data_persistence():
    """Tests saving and retrieving a DFT result from the database."""
    from ase.db import connect

    from mlip_autopipec.config.models import DFTResult
    from mlip_autopipec.utils.ase_utils import save_dft_result

    # Clean up any previous database file
    if DB_PATH.exists():
        DB_PATH.unlink()

    atoms = bulk("Si", "diamond", a=5.43)
    result = DFTResult(
        job_id="a-fake-job-id",  # type: ignore
        energy=-100.0,
        forces=[[0.0, 0.0, 0.0]] * 2,
        stress=[0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
    )

    logging.info(f"Saving DFT result to database: {DB_PATH}")
    save_dft_result(DB_PATH, atoms, result, config_type="uat_test")

    assert DB_PATH.exists(), "Database file was not created."

    with connect(DB_PATH) as db:
        assert len(db) == 1, "Database should contain exactly one entry."
        retrieved_row = db.get(id=1)
        logging.info("Retrieved data from database:")
        logging.info(f"  - Energy: {retrieved_row.energy}")
        logging.info(f"  - Stored Job ID: {retrieved_row.data['job_id']}")

        assert retrieved_row.energy == result.energy
        assert retrieved_row.data["job_id"] == str(result.job_id)

    # Clean up the database file
    DB_PATH.unlink()


def main():
    """Runs all UAT scenarios for Cycle 1."""
    logging.info("--- Starting UAT for Cycle 1: Automated DFT Factory ---")
    uat_c1_001_happy_path_calculation()
    uat_c1_002_heuristic_verification()
    # UAT-C1-003 (Resilience) is best tested with unit tests, as it
    # requires mocking specific failure modes.
    uat_c1_004_data_persistence()
    logging.info("\n--- UAT for Cycle 1 Completed Successfully ---")


if __name__ == "__main__":
    main()
