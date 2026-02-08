"""
Tutorial 02: Oracle Component Test (Cycle 03)

This script demonstrates the usage of the QEOracle component,
including self-healing and periodic embedding.

Note: Since Quantum Espresso (pw.x) is not installed in this environment,
we mock the ASE Calculator to simulate DFT calculations.
"""

import logging
import sys
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk, molecule
from ase.calculators.calculator import Calculator

# Add src to path if running from root
sys.path.append("src")

from mlip_autopipec.components.oracle.embedding import embed_cluster
from mlip_autopipec.components.oracle.qe import QEOracle
from mlip_autopipec.domain_models.config import QEOracleConfig
from mlip_autopipec.domain_models.enums import OracleType
from mlip_autopipec.domain_models.structure import Structure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Tutorial02")


def run_basic_dft_scenario() -> None:
    logger.info("\n--- Scenario 03-01: Basic DFT Calculation (Mocked) ---")

    # 1. Create Config
    config = QEOracleConfig(name=OracleType.QE, kspacing=0.1, ecutwfc=30.0, ecutrho=150.0)

    # 2. Create Structure (Bulk Silicon)
    atoms = bulk("Si", "diamond", a=5.43)
    structure = Structure.from_ase(atoms)

    # 3. Mock Espresso
    with patch("mlip_autopipec.components.oracle.qe.Espresso") as MockEspresso:
        # Setup mock behavior
        mock_calc = MagicMock(spec=Calculator)
        mock_calc.get_potential_energy.return_value = -15.0  # eV
        mock_calc.get_forces.return_value = np.zeros((len(atoms), 3))
        mock_calc.get_stress.return_value = np.zeros(6)
        mock_calc.parameters = {}
        MockEspresso.return_value = mock_calc

        # 4. Run Oracle
        oracle = QEOracle(config)
        logger.info("Running Oracle.compute()...")
        results = list(oracle.compute([structure]))

        # 5. Verify
        if results:
            s = results[0]
            logger.info(f"Success! Energy: {s.energy} eV")
            if s.forces is not None:
                logger.info(f"Forces shape: {s.forces.shape}")
        else:
            logger.error("Oracle returned no results.")


def run_healing_scenario() -> None:
    logger.info("\n--- Scenario 03-02: Self-Healing Mechanism (Mocked) ---")

    # 1. Create Config (High mixing beta to trigger failure)
    config = QEOracleConfig(
        name=OracleType.QE,
        mixing_beta=0.9,  # High
    )

    # 2. Create Structure
    atoms = bulk("Si", "diamond", a=5.43)
    structure = Structure.from_ase(atoms)

    # 3. Mock Espresso with Failure then Success
    with patch("mlip_autopipec.components.oracle.qe.Espresso") as MockEspresso:
        mock_calc = MagicMock(spec=Calculator)
        mock_calc.parameters = {"mixing_beta": 0.9}

        # Side effect: First call raises Exception, subsequent calls return value
        from itertools import chain, repeat

        mock_calc.get_potential_energy.side_effect = chain(
            [Exception("Convergence failed")], repeat(-15.0)
        )
        mock_calc.get_forces.return_value = np.zeros((len(atoms), 3))
        mock_calc.get_stress.return_value = np.zeros(6)

        MockEspresso.return_value = mock_calc

        # 4. Run Oracle
        oracle = QEOracle(config)
        logger.info("Running Oracle.compute() with potential failure...")
        results = list(oracle.compute([structure]))

        # 5. Verify Healing
        if results:
            s = results[0]
            logger.info(f"Success after healing! Energy: {s.energy} eV")
            # Check if parameters were updated (healing logic reduces mixing_beta to 0.3)
            # In the mock, we can check mock_calc.parameters if updated in place
            # But get_potential_energy might have been called on the SAME mock instance
            # The Healer modifies the calculator instance.
            final_beta = mock_calc.parameters["mixing_beta"]
            logger.info(f"Final mixing_beta: {final_beta}")
            if final_beta == 0.3:
                logger.info("Verified: mixing_beta was reduced to 0.3")
            else:
                logger.warning(f"Healing verification failed? beta={final_beta}")
        else:
            logger.error("Oracle failed to heal.")


def run_embedding_scenario() -> None:
    logger.info("\n--- Scenario 03-03: Periodic Embedding Visualization ---")

    # 1. Create Cluster (H2O molecule)
    cluster = molecule("H2O")
    logger.info(f"Original Cluster: {len(cluster)} atoms, PBC={cluster.pbc}")

    # 2. Embed in Vacuum
    vacuum = 5.0
    embedded = embed_cluster(cluster, vacuum)

    logger.info(f"Embedded Structure: {len(embedded)} atoms, PBC={embedded.pbc}")
    cell = embedded.get_cell()
    logger.info(f"Cell dimensions: {cell.lengths()}")

    # Verify
    if np.all(embedded.pbc) and np.all(cell.lengths() > 5.0):
        logger.info("Verified: Cluster successfully embedded in periodic box.")
    else:
        logger.error("Embedding verification failed.")


if __name__ == "__main__":
    run_basic_dft_scenario()
    run_healing_scenario()
    run_embedding_scenario()
