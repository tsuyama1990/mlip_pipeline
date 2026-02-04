# FINAL UAT: Master Test Plan

## 1. Tutorial Strategy

Our strategy is to turn the complex scientific workflow of "Hetero-epitaxial Growth & Ordering of Fe/Pt on MgO" into a series of digestible, executable "scientific papers" in the form of Jupyter Notebooks.

### Key Principles
1.  **Divide & Conquer Training**: Instead of training a massive potential at once, we guide the user to train sub-systems (MgO, FePt) separately before tackling the complex interface.
2.  **Hybrid Simulation**: We explicitly demonstrate how to link different simulation engines (LAMMPS for fast deposition, EON/kMC for slow ordering) using a common potential.
3.  **Mock vs Real Mode**: To ensure the tutorials are verifiable in a CI/CD environment, every notebook must implement a "Mock Mode".
    -   **CI Mode**: Runs on tiny supercells (e.g., 2x2x1), with minimal atoms (N=5) and short steps (T=100), or loads pre-calculated results.
    -   **User Mode**: Runs the full-scale scientific simulation.

## 2. Notebook Plan

We will deliver the following notebooks in the `tutorials/` directory.

### NB01: Pre-Training Fundamentals
*   **Goal**: Generate and train separate potentials for the substrate (MgO) and the deposit (FePt).
*   **Scenario**:
    1.  Use `StructureGenerator` to create bulk and surface configurations for MgO.
    2.  Use `StructureGenerator` to create bulk L10 phases and random alloys for FePt.
    3.  Run the Active Learning Loop (Cycle 1-4 features) to train two separate potentials.
*   **Outcome**: Two `.yace` files (`mgo.yace`, `fept.yace`) that are stable for their respective pure phases.

### NB02: Interface Learning & Adhesion
*   **Goal**: Learn the interaction between the deposit and the substrate.
*   **Scenario**:
    1.  Place Fe and Pt clusters on an MgO(001) slab.
    2.  Run Active Learning specifically on these interface structures.
    3.  Perform DFT calculations on these heterogeneous systems.
*   **Outcome**: A robust "Interface Potential" that predicts the correct adhesion energy and prevents atoms from passing through the substrate (Core-Repulsion check).

### NB03: Dynamic Deposition (MD)
*   **Goal**: Simulate the physical deposition process using LAMMPS.
*   **Scenario**:
    1.  Load the trained potential.
    2.  Setup LAMMPS with `fix deposit` to drop Fe and Pt atoms alternately onto the hot MgO substrate (600K).
    3.  **Visualization**: Show atoms landing, diffusing, and forming small islands (Nucleation).
*   **Constraint**: Must check for "flying ice cube" artifacts or unphysical fusion (atoms merging).

### NB04: Ordering & Ripening (aKMC)
*   **Goal**: Observe the chemical ordering (L10 formation) that is too slow for MD.
*   **Scenario**:
    1.  Take the final snapshot from NB03 (disordered island).
    2.  Pass it to the `EON` kMC engine.
    3.  Run Adaptive kMC to find lower-energy basins (ordering).
    4.  **Analysis**: Calculate the Short-Range Order (SRO) parameter to quantify the transition from random alloy to L10 ordered phase.

## 3. Validation Steps

The QA Agent (and the user) should verify the following success criteria when running the notebooks.

### General Checks
-   **Execution Time**: In CI mode, the entire suite must run in under 15 minutes.
-   **Dependencies**: If `lammps` or `eon` binaries are missing, the notebooks must degrade gracefully (print a warning and skip the cell, or load a cached image) rather than crashing.

### Scientific Checks
-   **Energy Conservation**: In NVE runs during training, total energy drift should be minimal.
-   **Adhesion**: Fe/Pt atoms must stick to the MgO surface, not bounce off or penetrate.
-   **Ordering**: NB04 must show an increase in the number of Fe-Pt bonds compared to a random distribution (approaching L10 structure).
-   **Visuals**: Generated PNGs must show distinct atomic layers (MgO slab) and cluster formation.
