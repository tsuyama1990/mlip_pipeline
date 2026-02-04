# Final User Acceptance Test (UAT) & Tutorial Plan

## 1. Tutorial Strategy: "The Executable Scientific Paper"

The goal of the User Acceptance Test is not just to verify code coverage, but to demonstrate that **PYACEMAKER** can solve a genuine Grand Challenge in materials science: **The Hetero-epitaxial Growth and Ordering of Fe/Pt Nanoparticles on MgO Substrates.**

We will deliver a set of Jupyter Notebooks in the `tutorials/` directory. These notebooks serve a dual purpose:
1.  **For Users**: They act as a step-by-step guide ("The Aha! Moment") to using the system.
2.  **For QA/CI**: They act as the ultimate system integration test.

### The "Mock vs Real" Strategy
Scientific simulations are computationally expensive and time-consuming (hours to days). To ensure our Continuous Integration (CI) and automated QA processes remain agile, every notebook must implement a **Dual Mode** execution strategy controlled by an environment variable `IS_CI_MODE`.

*   **Real Mode** (Default):
    *   Runs full-scale DFT calculations (via Quantum Espresso) and long MD simulations.
    *   Requires a GPU and significant CPU resources.
    *   Target: Users with HPC access.
*   **Mock/CI Mode** (`IS_CI_MODE=true`):
    *   **Skip Heavy Compute**: Instead of running `pace_train` or `pw.x`, the notebook will load pre-calculated artifacts (e.g., `precomputed_potential.yace`, `reference_dft_data.xyz`) from a `test_assets/` directory.
    *   **Tiny Systems**: If calculation is unavoidable (e.g., MD deposition), use a minimized system (e.g., 2x2 surface, 10 atoms, 100 steps) that finishes in seconds.
    *   **Mocked Binaries**: If external binaries (LAMMPS, EON) are missing, the Python wrapper should catch the `FileNotFoundError` and print a "Simulation Skipped (Demo Mode)" message rather than crashing, while still plotting the *expected* result from a saved file.

## 2. Notebook Plan

The UAT is broken down into 4 logical scientific phases.

### Notebook 01: The Foundation - Pre-Training Bulk Systems
**Filename**: `tutorials/01_MgO_FePt_Training.ipynb`
**Objective**: Demonstrate the "Zero-Config" training pipeline for simple bulk materials.
*   **Scenario**:
    1.  User defines `config.yaml` for MgO (oxide) and FePt (alloy).
    2.  System generates initial structures (distorted NaCl for MgO, L10/A1 for FePt).
    3.  System runs Active Learning Cycle (Cycle 01-04 features).
*   **Key Verification**:
    *   Does the `StructureGenerator` create valid crystals?
    *   Does the `PacemakerWrapper` produce a `.yace` file?
    *   **Metric**: The resulting potential must predict the lattice constant of MgO within 1% of the DFT value.

### Notebook 02: The Interface - Learning Adhesion
**Filename**: `tutorials/02_Interface_Learning.ipynb`
**Objective**: Show how to handle multi-component interfaces, the hardest part of potential development.
*   **Scenario**:
    1.  Load potentials from NB01.
    2.  Construct a "slab + cluster" geometry (FePt cluster on MgO surface).
    3.  Run targeted Active Learning to capture the interfacial forces (adhesion).
*   **Key Verification**:
    *   **Physics Check**: The potential must predict a *negative* adhesion energy (attractive force) between the metal and the oxide.
    *   **Robustness**: Verify that atoms do not "fuse" (nuclear overlap) at the interface (checking the Hybrid Potential implementation).

### Notebook 03: The Dynamics - Deposition MD
**Filename**: `tutorials/03_Deposition_MD.ipynb`
**Objective**: Demonstrate the connection to LAMMPS and the "Uncertainty Watchdog".
*   **Scenario**:
    1.  Setup a LAMMPS simulation using `fix deposit`.
    2.  Deposit Fe and Pt atoms continuously onto the hot substrate (600K).
    3.  **The "Halt" Event**: Artificially trigger (or naturally observe) an uncertainty spike. Show how the system pauses, retrains, and resumes.
*   **Key Verification**:
    *   **Visual**: Generate a snapshot showing an island forming on the surface (not sinking in).
    *   **Stability**: The simulation runs to completion (e.g., 1000 atoms) without segmentation faults (Cycle 05 feature).

### Notebook 04: The Long Game - Ordering via aKMC
**Filename**: `tutorials/04_Ordering_aKMC.ipynb`
**Objective**: Demonstrate the "Scale-Up" capability connecting MD to kMC (EON).
*   **Scenario**:
    1.  Take the disordered, as-deposited cluster from NB03.
    2.  Hand it over to the EON wrapper.
    3.  Run Adaptive KMC to find lower-energy ordered states (L10 ordering) that are inaccessible to MD time-scales.
*   **Key Verification**:
    *   **Visual**: Show the transition from a random mix (A1) to an ordered layered structure (L10).
    *   **Metric**: Calculate the "Order Parameter" (number of Fe-Pt bonds vs Fe-Fe bonds) and show it increasing over kMC steps.

## 3. Validation Steps (For QA Agent)

When running the full suite (e.g., `pytest --nbval tutorials/`), the QA Agent must verify:

1.  **File Existence**:
    *   `tutorials/*.ipynb` exist.
    *   `tutorials/assets/precomputed_*.yace` exist (for Mock Mode).
2.  **Execution Success**:
    *   All cells execute with exit code 0.
    *   No `Traceback` or `Segmentation Fault` in the output.
3.  **Physical Sanity Checks (Assertions in Code)**:
    *   `assert abs(predicted_lattice_mgo - 4.21) < 0.05` (NB01)
    *   `assert adhesion_energy < 0.0` (NB02)
    *   `assert final_potential_energy < initial_potential_energy` (NB04 - Ordering implies energy minimization)
4.  **Visual Output**:
    *   Notebooks must contain inline plots (Matplotlib/ASE view) showing atoms.
    *   The "Halt & Resume" graph in NB03 should show a drop in Uncertainty ($\gamma$) after retraining.

This plan ensures that **PYACEMAKER** is not just a collection of scripts, but a coherent scientific instrument verified against real-world physics problems.
