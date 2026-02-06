# Final UAT & Tutorial Master Plan

## 1. Tutorial Strategy

The User Acceptance Testing (UAT) for PYACEMAKER is designed not just to verify code functionality, but to serve as a comprehensive educational journey for the user. We will deliver a set of "Executable Scientific Papers" in the form of Jupyter Notebooks.

### 1.1. The "Divide & Conquer" Philosophy
Real-world materials science is complex. We will not attempt to train a "Universal Potential" in one go. Instead, the tutorials will guide the user through a logical "Divide & Conquer" workflow:
1.  **Component Training:** Train potentials for the substrate (MgO) and the deposit (FePt) separately.
2.  **Interface Training:** Train the interaction between the two.
3.  **Application:** Use the combined knowledge for the final simulation.

This approach teaches the user best practices for MLIP construction: building complexity layer by layer.

### 1.2. The "Mock vs Real" Dual Mode
To ensure the tutorials are robust and testable in a Continuous Integration (CI) environment, every notebook will implement a strict "Dual Mode" execution strategy.

*   **CI Mode (`IS_CI_MODE = True`)**:
    *   **Trigger:** Activated when the environment variable `CI=true` is set.
    *   **Data:** Uses pre-calculated data or generates tiny datasets (e.g., 2 atoms, 1 k-point).
    *   **Execution:** Runs for minimal steps (e.g., 10 MD steps).
    *   **Goal:** Verify the *code path* works without crashing.
    *   **Runtime:** < 5 minutes total.

*   **Real Mode (`IS_CI_MODE = False`)**:
    *   **Trigger:** Default behavior for users.
    *   **Data:** Performs full DFT calculations and extensive sampling.
    *   **Execution:** Runs for nanoseconds of MD and thousands of kMC steps.
    *   **Goal:** Produce scientifically valid results.
    *   **Runtime:** Hours to Days.

### 1.3. Visualization First
Since users cannot see atoms, visualization is the primary debugging tool. Every key step (structure generation, interface creation, MD snapshots) must be visualized inline using `ase.visualize.plot` or similar static plotters (matplotlib) to ensure the notebook is self-documenting even when viewed on GitHub.

## 2. Notebook Plan

We will deliver four sequential notebooks located in the `tutorials/` directory.

### NB01: Foundations - Pre-Training Components
*   **Filename:** `tutorials/01_MgO_FePt_Training.ipynb`
*   **Objective:** Establish the baseline potentials for the pure phases.
*   **Key Steps:**
    1.  Initialize `StructureGenerator`.
    2.  Generate bulk MgO (NaCl structure) and distort it (EOS).
    3.  Generate bulk FePt (L10 structure) and random alloys.
    4.  Run the `Oracle` (or Mock Oracle in CI) to get forces.
    5.  Train two separate basic potentials or one combined potential.
    6.  **Validation:** Plot the Equation of State (Energy vs Volume) for MgO and check the bulk modulus.

### NB02: The Interface - Adhesion & Surface
*   **Filename:** `tutorials/02_Interface_Learning.ipynb`
*   **Objective:** Teach the potential how Fe and Pt atoms interact with the MgO surface.
*   **Key Steps:**
    1.  Load the trained potential from NB01.
    2.  Create an MgO(001) slab.
    3.  Place Fe and Pt clusters on top of the slab.
    4.  Perform "Active Learning" specifically on these interface structures.
    5.  **Validation:** Calculate and plot the Adhesion Energy curve ($E_{adh} = E_{total} - E_{slab} - E_{cluster}$) as the cluster moves away from the surface. Ensure no "ghost forces" exist in the vacuum.

### NB03: Dynamics - Deposition Simulation (MD)
*   **Filename:** `tutorials/03_Deposition_MD.ipynb`
*   **Objective:** Simulate the physical vapor deposition (PVD) process.
*   **Key Steps:**
    1.  Setup `DynamicsEngine` with LAMMPS.
    2.  Configure `fix deposit` to add Fe/Pt atoms every N steps.
    3.  Enable the **Uncertainty Watchdog** (`fix halt`).
    4.  Run the simulation.
    5.  **Demonstrate Self-Healing:** Ideally, force a high-uncertainty event (e.g., by increasing deposition energy) and show how the system pauses, learns, and resumes (or simulates this flow in CI).
    6.  **Visualization:** Show a movie or snapshots of island nucleation on the surface.

### NB04: Evolution - Long-Term Ordering (aKMC)
*   **Filename:** `tutorials/04_Ordering_aKMC.ipynb`
*   **Objective:** Bridge the time-scale gap. Convert the disordered deposit from NB03 into an ordered L10 crystal.
*   **Key Steps:**
    1.  Take the final frame from NB03.
    2.  Setup `EonClient` for Adaptive KMC.
    3.  Define the `pace_driver.py` script.
    4.  Run aKMC to find saddle points and traverse energy barriers.
    5.  **Validation:** Calculate the "Order Parameter" (number of Fe-Pt bonds vs Fe-Fe/Pt-Pt bonds). Show it increasing over KMC steps, indicating chemical ordering.

## 3. Validation Steps

The Quality Assurance (QA) agent will verify the tutorials using the following checklist:

### 3.1. Dependency Check
*   [ ] Does `pip install .` install all python dependencies?
*   [ ] Do the notebooks handle missing external binaries (LAMMPS/QE/EON) gracefully?
    *   *Expectation:* If `lmp_serial` is not found, the notebook should print a warning and switch to a "Mock/Replay" mode or skip the cell, rather than crashing with `FileNotFoundError`.

### 3.2. CI Mode Execution
*   [ ] Can all 4 notebooks run sequentially in under 10 minutes on a standard runner?
*   [ ] Do the assertions pass in CI mode?
    *   `assert potential_energy < 0`
    *   `assert final_structure is not None`

### 3.3. Scientific Validity (Real Mode)
*   [ ] **Core Repulsion:** Does the potential prevent atoms from overlapping ($r < 1.0 \AA$)?
*   [ ] **Adhesion:** Do Fe/Pt atoms stick to MgO (negative binding energy)?
*   [ ] **Ordering:** Does the aKMC simulation actually lower the potential energy of the cluster over time?

### 3.4. Artifact Generation
*   [ ] Are `.yace` potential files created in the `potentials/` directory?
*   [ ] Are `log.lammps` and `report.json` files generated?
*   [ ] Are plots (PNG/SVG) visible in the rendered notebook?
