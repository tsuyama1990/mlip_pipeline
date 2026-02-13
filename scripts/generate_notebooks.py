import nbformat as nbf
from pathlib import Path

def create_notebook_1():
    nb = nbf.v4.new_notebook()

    nb.cells.append(nbf.v4.new_markdown_cell("""
# Tutorial 01: From Scratch to Active Learning - Training Potentials for Interfaces

**Goal**: Demonstrate the "Aha! Moment" where the system automatically learns and improves.

This tutorial guides you through:
1.  **Setup**: Defining the system (MgO substrate, FePt cluster).
2.  **Phase A (Bulk)**: Training simple bulk potentials.
3.  **Phase B (Interface)**: Active Learning for interface configurations.
4.  **Validation**: Inspecting the results.

**Modes**:
*   **CI Mode**: Uses mock components and tiny systems for fast verification.
*   **Real Mode**: Uses actual DFT (Quantum Espresso), Pacemaker, and LAMMPS.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
import os
import sys
from pathlib import Path

# Add src to path if running from tutorials directory
if Path("../src").exists():
    sys.path.append(str(Path("../src").resolve()))

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import (
    GlobalConfig,
    OrchestratorConfig,
    GeneratorConfig,
    OracleConfig,
    TrainerConfig,
    DynamicsConfig,
    ValidatorConfig,
    SystemConfig,
    ActiveLearningConfig
)
from mlip_autopipec.domain_models.enums import (
    ExecutionMode,
    GeneratorType,
    OracleType,
    TrainerType,
    DynamicsType,
    ValidatorType,
    DFTCode
)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Detect Mode
IS_CI_MODE = os.environ.get("CI", "false").lower() == "true"
print(f"Running in CI Mode: {IS_CI_MODE}")

WORK_DIR = Path("outputs/01_MgO_FePt")
WORK_DIR.mkdir(parents=True, exist_ok=True)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Step 1: Define Configuration

if IS_CI_MODE:
    # MOCK CONFIGURATION (Fast, no external dependencies)
    config = GlobalConfig(
        orchestrator=OrchestratorConfig(
            max_cycles=1,
            work_dir=WORK_DIR,
            execution_mode=ExecutionMode.MOCK,
            cleanup_on_exit=False
        ),
        generator=GeneratorConfig(
            type=GeneratorType.MOCK,
            mock_count=2
        ),
        oracle=OracleConfig(
            type=OracleType.MOCK
        ),
        trainer=TrainerConfig(
            type=TrainerType.MOCK,
            mock_potential_content="MOCK_POTENTIAL_CONTENT_YACE"
        ),
        dynamics=DynamicsConfig(
            type=DynamicsType.MOCK,
            halt_on_uncertainty=True
        ),
        validator=ValidatorConfig(
            type=ValidatorType.MOCK
        )
    )
else:
    # REAL CONFIGURATION (Requires QE, Pacemaker, LAMMPS)
    # Note: This configuration assumes external binaries are in PATH.
    config = GlobalConfig(
        orchestrator=OrchestratorConfig(
            max_cycles=5,
            work_dir=WORK_DIR,
            execution_mode=ExecutionMode.PRODUCTION,
            cleanup_on_exit=False
        ),
        generator=GeneratorConfig(
            type=GeneratorType.RANDOM, # or M3GNET if available
        ),
        oracle=OracleConfig(
            type=OracleType.DFT,
            dft_code=DFTCode.QUANTUM_ESPRESSO,
            command="mpirun -np 4 pw.x"
        ),
        trainer=TrainerConfig(
            type=TrainerType.PACEMAKER,
            max_epochs=100
        ),
        dynamics=DynamicsConfig(
            type=DynamicsType.LAMMPS,
            halt_on_uncertainty=True
        ),
        validator=ValidatorConfig(
            type=ValidatorType.PHYSICS
        )
    )

print("Configuration defined.")
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Step 2: Initialize Orchestrator
orchestrator = Orchestrator(config)
print("Orchestrator initialized.")
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Step 3: Run the Workflow
# This triggers the Active Learning Loop:
# Explore -> Detect Uncertainty -> Oracle (DFT) -> Train -> Validate

try:
    orchestrator.run()
    print("Workflow completed successfully.")
except Exception as e:
    print(f"Workflow failed: {e}")
    # In CI mode, we might want to raise to fail the test, but let's just print for now
    if IS_CI_MODE:
        raise
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Step 4: Inspect Results
# Check the state of the workflow

state = orchestrator.state_manager.state
print(f"Final Cycle: {state.current_cycle}")
print(f"Active Potential: {state.active_potential_path}")
print(f"Dataset Path: {state.dataset_path}")

# Copy the active potential to a fixed location for Tutorial 02
if state.active_potential_path and state.active_potential_path.exists():
    destination = WORK_DIR / "active_potential.yace"
    import shutil
    shutil.copy(state.active_potential_path, destination)
    print(f"Copied active potential to {destination}")
else:
    print("No active potential found to copy.")
"""))

    return nb

def create_notebook_2():
    nb = nbf.v4.new_notebook()

    nb.cells.append(nbf.v4.new_markdown_cell("""
# Tutorial 02: Simulating Growth - Deposition MD and Long-Timescale Ordering (aKMC)

**Goal**: Demonstrate the system's ability to handle complex, multi-scale physics using the potential trained in Tutorial 01.

This tutorial guides you through:
1.  **Setup**: Loading the potential.
2.  **Phase A (Deposition)**: LAMMPS MD simulation of Fe/Pt deposition.
3.  **Phase B (Ordering)**: EON aKMC simulation of L10 ordering.
4.  **Analysis**: Visualizing results.
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
import os
import sys
from pathlib import Path
import shutil

# Add src to path
if Path("../src").exists():
    sys.path.append(str(Path("../src").resolve()))

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import (
    GlobalConfig,
    DynamicsConfig,
    EONConfig,
    GeneratorConfig,
    OrchestratorConfig,
    OracleConfig,
    TrainerConfig,
    ValidatorConfig
)
from mlip_autopipec.domain_models.enums import DynamicsType, ExecutionMode
from mlip_autopipec.dynamics.interface import MockDynamics
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Detect Mode
IS_CI_MODE = os.environ.get("CI", "false").lower() == "true"
WORK_DIR = Path("outputs/02_Deposition")
WORK_DIR.mkdir(parents=True, exist_ok=True)
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Step 1: Load Potential
# In a real scenario, we would load 'outputs/01_MgO_FePt/active_potential.yace'
# For this tutorial, we will use a Mock potential in CI mode.

potential_path = Path("outputs/01_MgO_FePt/active_potential.yace")

if IS_CI_MODE:
    # Create a dummy potential file if it doesn't exist
    if not potential_path.exists():
        potential_path.parent.mkdir(parents=True, exist_ok=True)
        potential_path.write_text("MOCK_POTENTIAL_CONTENT")

print(f"Using potential: {potential_path}")
active_potential = Potential(path=potential_path, format="yace")
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Step 2: Phase A - Deposition MD (LAMMPS)

if IS_CI_MODE:
    # Use Mock Dynamics
    dyn_config = DynamicsConfig(
        type=DynamicsType.MOCK,
        steps=100,
        mock_frames=5
    )
    # We need a dummy structure to start
    # Create a simple structure (e.g. 2 atoms)
    from ase import Atoms
    atoms = Atoms('CO', positions=[[0, 0, 0], [0, 0, 1.2]], cell=[10, 10, 10], pbc=True)
    initial_structure = Structure(atoms=atoms, provenance="manual_setup")

    # Instantiate MockDynamics directly for demonstration if not going through Orchestrator
    # Or use Orchestrator logic. Here we use the component directly to show flexibility.

    dynamics = MockDynamics(dyn_config)
    print("Running Mock Deposition MD...")
    trajectory = list(dynamics.simulate(active_potential, initial_structure))
    print(f"Deposition complete. Generated {len(trajectory)} frames.")
    final_structure = trajectory[-1]

else:
    # Real LAMMPS Deposition
    print("Running Real LAMMPS Deposition...")
    from mlip_autopipec.dynamics.lammps_driver import LAMMPSDriver

    dyn_config = DynamicsConfig(
        type=DynamicsType.LAMMPS,
        steps=5000,
        temperature=600.0,
        lammps_command="lmp",
        halt_on_uncertainty=True
    )

    try:
        driver = LAMMPSDriver(WORK_DIR, dyn_config)
        # Note: In a real deposition simulation, we would insert atoms iteratively.
        # Here we run MD on the initial structure for demonstration.
        trajectory = list(driver.simulate(active_potential, initial_structure))
        final_structure = trajectory[-1]
        print(f"MD complete. Final energy: {final_structure.energy}")
    except Exception as e:
        print(f"Real execution failed (likely missing 'lmp'): {e}")
        print("Falling back to initial structure.")
        final_structure = initial_structure
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Step 3: Phase B - Ordering aKMC (EON)

if IS_CI_MODE:
    # Mock EON
    print("Running Mock aKMC Ordering...")
    ordered_structure = final_structure.model_copy(deep=True)
    # Simulate ordering effect
    if hasattr(ordered_structure.atoms, "set_chemical_symbols"):
         # Swap some atoms to simulate ordering if possible, or just keep as is
         pass
    print("aKMC complete.")

else:
    # Real EON execution
    print("Running Real EON aKMC...")
    from mlip_autopipec.dynamics.eon_driver import EONDriver

    eon_config = EONConfig(
        temperature=500.0,
        client_path="eonclient"
    )
    dyn_config = DynamicsConfig(
        type=DynamicsType.EON,
        eon=eon_config
    )

    try:
        driver = EONDriver(WORK_DIR, dyn_config)
        # EONDriver simulate yields structures (saddle points, minima)
        trajectory = list(driver.simulate(active_potential, final_structure))
        if trajectory:
            ordered_structure = trajectory[-1]
            print(f"aKMC complete. Found {len(trajectory)} states.")
        else:
             print("aKMC found no new states.")
             ordered_structure = final_structure
    except Exception as e:
        print(f"Real execution failed (likely missing 'eonclient'): {e}")
        print("Falling back to MD structure.")
        ordered_structure = final_structure
"""))

    nb.cells.append(nbf.v4.new_code_cell("""
# Step 4: Analysis & Visualization
# Visualize the final structure

try:
    from ase.visualize import view
    # view(ordered_structure.to_ase()) # Interactive view (not visible in CI)
    print("Structure ready for visualization.")
    print(ordered_structure.to_ase())
except ImportError:
    print("ASE GUI not available.")
"""))

    return nb

def main():
    tutorials_dir = Path("tutorials")
    tutorials_dir.mkdir(exist_ok=True)

    nb1 = create_notebook_1()
    with open(tutorials_dir / "01_MgO_FePt_Training.ipynb", "w") as f:
        nbf.write(nb1, f)
    print("Created tutorials/01_MgO_FePt_Training.ipynb")

    nb2 = create_notebook_2()
    with open(tutorials_dir / "02_Deposition_and_Ordering.ipynb", "w") as f:
        nbf.write(nb2, f)
    print("Created tutorials/02_Deposition_and_Ordering.ipynb")

if __name__ == "__main__":
    main()
