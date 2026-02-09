# FINAL UAT: User Acceptance Testing and Tutorial Strategy

## 1. Tutorial Strategy

The user acceptance testing (UAT) for PyAceMaker is designed to be an educational experience, transforming the raw technical specifications into "executable scientific papers." The core philosophy is **"Divide and Conquer"**, guiding the user through the construction of a complex potential (Fe/Pt on MgO) in manageable, logical steps.

### The "Time-Scale Problem"
A critical aspect of this UAT is demonstrating how the system bridges the gap between Molecular Dynamics (MD) and Kinetic Monte Carlo (kMC). The tutorial will explicitly show:
1.  **MD (Fast Dynamics):** Simulating the deposition of atoms, where kinetic energy and thermal diffusion dominate.
2.  **kMC (Slow Dynamics):** Taking the resulting structure and finding the long-term equilibrium (L10 ordering) which would be impossible to reach with MD alone.

### "Mock Mode" vs "Real Mode"
To ensure the tutorials are verifiable in a Continuous Integration (CI) environment without requiring expensive HPC resources or long wait times, every notebook will implement a dual-mode execution strategy.

*   **Real Mode (User):**
    *   **Scale:** Large slabs (e.g., 500+ atoms), realistic vacuum (15Å+).
    *   **Compute:** Runs full DFT (QE/VASP) and long MD/kMC simulations.
    *   **Time:** Hours to Days.
    *   **Trigger:** Default when running on a user's machine with `ASE_ESPRESSO_COMMAND` set.

*   **Mock Mode (CI/Demo):**
    *   **Scale:** Tiny unit cells (e.g., 2x2x1 MgO), minimal vacuum.
    *   **Compute:**
        *   **DFT:** Uses a pre-calculated "Lookup Table" calculator or a fast semi-empirical method (e.g., EMT/LJ) to mock DFT forces/energies instantly.
        *   **MD/kMC:** Runs for minimal steps (e.g., 10 steps) to verify the *workflow* (file generation, command execution) without waiting for physics.
    *   **Time:** < 5 minutes per notebook.
    *   **Trigger:** Activated when environment variable `IS_CI_MODE="true"` is detected.

## 2. Notebook Plan

The UAT will be delivered as a series of 4 sequential Jupyter Notebooks in the `tutorials/` directory.

### `tutorials/01_MgO_FePt_Training.ipynb`: The Foundation
*   **Goal:** Generate and train separate potentials for the substrate (MgO) and the deposit (FePt).
*   **Narrative:** "Before we can simulate the interface, we must understand the bulk."
*   **Key Steps:**
    1.  Define `config.yaml` for MgO (Ionic) and FePt (Metallic).
    2.  Run `StructureGenerator` to create initial random structures and strained bulk.
    3.  Run `Oracle` (DFT) to get ground truth labels (or Mock labels).
    4.  Run `Trainer` to fit initial ACE potentials.
    5.  **Validation:** Check EOS curves for MgO and FePt.

### `tutorials/02_Interface_Learning.ipynb`: The Interface
*   **Goal:** Learn the interaction between the deposit and the substrate.
*   **Narrative:** "What happens when Fe meets MgO?"
*   **Key Steps:**
    1.  Construct "Slab + Cluster" geometries (Fe/Pt clusters on MgO surface).
    2.  Perform Active Learning (AL) on these interface structures.
    3.  Demonstrate `ActiveSetSelector` filtering redundant configurations.
    4.  **Validation:** Calculate the Adhesion Energy ($E_{adh}$) and verify it matches literature/DFT.

### `tutorials/03_Deposition_MD.ipynb`: The Dynamic Event
*   **Goal:** Simulate the physical deposition process using LAMMPS.
*   **Narrative:** "Let's rain atoms."
*   **Key Steps:**
    1.  Load the combined potential (MgO + FePt + Interface).
    2.  Setup LAMMPS `fix deposit` to insert Fe and Pt atoms alternately.
    3.  Run NVT MD at 600K.
    4.  **Observation:** Watch atoms land, diffuse, and form islands.
    5.  **Safety Check:** Verify no atoms penetrate the surface (Core Repulsion/Hybrid Potential check).

### `tutorials/04_Ordering_aKMC.ipynb`: The Long-Term Evolution
*   **Goal:** Bridge the timescale gap and observe L10 ordering.
*   **Narrative:** "MD is too slow. Let's fast-forward with kMC."
*   **Key Steps:**
    1.  Take the final frame from Notebook 03 (the disordered island).
    2.  Setup `EON` input files using the `EONWrapper`.
    3.  Run aKMC to explore saddle points and find lower energy basins.
    4.  **Analysis:** Calculate the "Order Parameter" (number of Fe-Pt bonds vs like-bonds).
    5.  **Visualisation:** Show the transition from "Random Alloy" to "Ordered Structure".

## 3. Validation Steps

The Quality Assurance (QA) agent or automated CI pipeline should look for the following artifacts to confirm success:

### 1. File Artifacts
*   `data/potential_mgo.yace` and `data/potential_fept.yace` exist.
*   `tutorials/outputs/deposition.dump` (LAMMPS trajectory) exists and is not empty.
*   `tutorials/outputs/kmc_result.con` (EON final structure) exists.

### 2. Physical Sanity Checks (Implemented as `assert` in notebooks)
*   **Stability:** `potential_energy < 0` for all equilibrated structures.
*   **Geometry:** `min_distance > 1.8 Å` (No nuclear fusion).
*   **Conservation:** Total energy drift in NVE MD is < 1e-4 eV/atom/ps.

### 3. Visual Verification
*   **Plots:**
    *   EOS Curves (Energy vs Volume) are smooth and convex.
    *   Parity Plots (DFT Force vs ACE Force) show tight correlation ($R^2 > 0.95$).
*   **Images:**
    *   Snapshot of Fe/Pt island on MgO surface.
    *   Color-coded view distinguishing Fe (Red) and Pt (Blue).

### 4. Error Handling Verification
*   If `lammps` or `eon` binaries are missing, the notebooks must **NOT** crash. They should print a warning ("Skipping execution...") and proceed to show pre-rendered results or exit gracefully.
