# Final User Acceptance Testing (UAT) Plan

## 1. Tutorial Strategy

The core strategy for User Acceptance Testing and onboarding is to provide **Executable Scientific Papers** in the form of Jupyter Notebooks. These notebooks serve a dual purpose: they guide new users through the system's capabilities (Tutorials) and act as rigorous integration tests for the development team (UAT).

### 1.1. The "Mock vs. Real" Dual Execution Mode
To ensure these tutorials can run in a Continuous Integration (CI) environment (GitHub Actions) without requiring expensive DFT software or days of compute time, every notebook must implement a `IS_CI_MODE` toggle.

-   **Mock Mode (CI/Fast):**
    -   Triggered by environment variable `CI=true`.
    -   **DFT:** Uses a pre-calculated database (lookup) or a fast effective medium potential (e.g., EMT/LJ) instead of calling Quantum Espresso.
    -   **MD/kMC:** Runs for minimal steps (e.g., 100 steps) on tiny supercells (e.g., 2 atoms).
    -   **Goal:** Verify that the Python API, file handling, and logic flow are unbroken. Execution time < 5 minutes.

-   **Real Mode (User/Production):**
    -   Triggered when `CI` is unset or `false`.
    -   **DFT:** Calls actual Quantum Espresso binaries.
    -   **MD/kMC:** Runs full production-scale simulations (e.g., 10^6 steps, 500 atoms).
    -   **Goal:** Reproduce the scientific results described in the `USER_TEST_SCENARIO.md` (Fe/Pt on MgO).

### 1.2. The Scientific Narrative: "Fe/Pt Nanoparticles on MgO"
The tutorials will follow the story of a researcher trying to synthesize L10-ordered FePt magnetic nanoparticles on an MgO(001) substrate. This scenario covers:
1.  **Generation:** Creating diverse structures (bulk, surface, interface).
2.  **Learning:** Training a potential that handles multiple elements (Fe, Pt, Mg, O).
3.  **Application:** Simulating deposition (MD) and long-term ordering (kMC).

## 2. Notebook Plan

We will deliver two comprehensive notebooks in the `tutorials/` directory.

### 2.1. `tutorials/01_MgO_FePt_Training.ipynb`
**Title:** From Zero to Potential: Active Learning of Fe-Pt/MgO
**Objective:** Demonstrate the "Divide and Conquer" training strategy.
**Key Steps:**
1.  **Initialization:** Define the `GlobalConfig` for a multi-element system.
2.  **Phase A (Substrate):** Generate and learn MgO bulk and (001) surface structures.
3.  **Phase B (Alloy):** Generate and learn Fe-Pt bulk alloys (L10, disordered fcc) and clusters.
4.  **Phase C (Interface):** Place Fe/Pt clusters on the MgO slab to learn adhesion energies.
5.  **Validation:** Run the `Validator` suite to check the lattice constants and formation energies of the learned potential against reference values.

### 2.2. `tutorials/02_Deposition_and_Ordering.ipynb`
**Title:** Bridging Time Scales: Deposition (MD) and Ordering (aKMC)
**Objective:** Demonstrate the Hybrid Simulation capability (MD + kMC) using the potential trained in Notebook 01.
**Key Steps:**
1.  **Setup:** Load the `potential.yace` from the previous notebook.
2.  **MD Phase (Deposition):**
    -   Use `LAMMPS` to simulate the deposition of 100 Fe and Pt atoms onto the MgO substrate at 600K.
    -   Visualize the formation of a disordered nanoparticle.
    -   **Check:** Ensure no atoms penetrate the substrate (Physics Robustness).
3.  **kMC Phase (Ordering):**
    -   Take the final MD snapshot and pass it to `EON`.
    -   Run Adaptive Kinetic Monte Carlo to simulate the ordering transformation (L10 phase formation) which occurs over seconds/hours (unreachable by MD).
    -   **Check:** Observe the increase in Fe-Pt bond order parameters.
4.  **Visualization:** Generate interactive plots (using `ase.visualize` or `nglview`) showing the final ordered structure.

## 3. Validation Steps

### 3.1. CI/Automated Checks
The QA agent (or CI pipeline) will execute the notebooks using `papermill` or `pytest --nbval` with `CI=true`.
**Success Criteria:**
-   All cells execute without error.
-   Outputs (graphs, files) are generated.
-   Assertions in the code (e.g., `assert final_energy < initial_energy`) pass.

### 3.2. Manual/Scientific QA
A domain expert (or the sophisticated QA agent) will run the notebooks in **Real Mode**.
**Success Criteria:**
-   **Adhesion Energy:** The potential should predict correct adhesion trends (Pt should wet MgO less than Fe).
-   **Stability:** The MD simulation must be stable for >1ns.
-   **Ordering:** The kMC simulation must show a trend towards chemical ordering (lower energy) compared to the initial random alloy.

### 3.3. Specific Scientific Validations (The "Time-Scale" Check)
-   **Interface Construction:** Verify that the training set construction logic correctly handles the vacuum padding and periodicity mismatch between MgO and FePt.
-   **Hybrid Potential:** Confirm that `pair_style hybrid/overlay` is correctly active in the LAMMPS input, preventing nuclear fusion at the impact point of deposition.
