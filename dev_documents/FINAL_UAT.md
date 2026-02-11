# Final User Acceptance Test (UAT) Plan

## 1. Tutorial Strategy: "The Executable Paper"

The User Acceptance Testing strategy revolves around providing "executable scientific papers" in the form of Jupyter Notebooks. These notebooks serve a dual purpose: they act as a comprehensive tutorial for new users and as a rigorous end-to-end test suite for the system.

### The "Mock vs Real" Dual Mode
To ensure the system can be tested in a Continuous Integration (CI) environment without access to heavy computational resources or DFT licenses, all tutorials must implement a "Dual Mode" execution strategy.

-   **CI Mode (`IS_CI_MODE=True`):**
    -   **Trigger:** Environment variable `CI=true` or user flag.
    -   **Behavior:**
        -   Uses `MockOracle` instead of real DFT (returns random but consistent forces/energies).
        -   Uses tiny supercells (e.g., 2 atoms) instead of production slabs.
        -   Runs minimal training epochs (e.g., 1 epoch).
        -   Runs very short MD/kMC trajectories (e.g., 10 steps).
        -   Skips heavy external binary calls if not found, printing a "Mock Success" message.
    -   **Goal:** Verify that the *code logic* (Orchestrator -> Component -> File I/O) is broken, completing the entire workflow in < 5 minutes.

-   **Real Mode (`IS_CI_MODE=False`):**
    -   **Trigger:** Default behavior when running on a workstation.
    -   **Behavior:**
        -   Uses real `QuantumEspresso` or `VASP` via ASE.
        -   Uses physically meaningful system sizes.
        -   Runs full training to convergence.
        -   Runs production-length MD/kMC.
    -   **Goal:** Verify the *scientific validity* of the results (e.g., correct lattice constants, stable deposition).

## 2. Notebook Plan

We will deliver two core notebooks that cover the "Fe/Pt Deposition on MgO" Grand Challenge scenario.

### Notebook 1: `tutorials/01_MgO_FePt_Training.ipynb`
**Title:** "Divide & Conquer: Active Learning for MgO and FePt"
**Objective:** Demonstrate the `StructureGenerator`, `Oracle`, and `Trainer` components by creating separate potentials for the substrate and the adatoms.

**Sections:**
1.  **Introduction**: Explanation of the "Divide and Conquer" strategy.
2.  **Part A: MgO Substrate**:
    -   Initialize `StructureGenerator` for MgO.
    -   Run "Cold Start" using M3GNet (or random if M3GNet unavailable).
    -   Run the Active Learning Loop (Explore -> Label -> Train) for 3 generations.
    -   *Validation*: Check MgO lattice constant and bulk modulus.
3.  **Part B: FePt Alloy**:
    -   Initialize `StructureGenerator` for FePt binary system.
    -   Focus exploration on L10 phase and surface configurations.
    -   Run Active Learning Loop.
    -   *Validation*: Check Fe-Pt mixing energy.
4.  **Part C: Interface Learning (Adhesion)**:
    -   Construct an MgO slab with Fe/Pt clusters on top.
    -   Run a targeted learning cycle to capture adhesion forces.
5.  **Conclusion**: Save the final `potential.yace` for use in the next tutorial.

**Success Criteria:**
-   Notebook runs to completion without errors.
-   `potential.yace` is generated.
-   Parity plots show convergence (RMSE decreases).

### Notebook 2: `tutorials/02_Deposition_and_Ordering.ipynb`
**Title:** "bridging Time Scales: Deposition (MD) and Ordering (aKMC)"
**Objective:** Demonstrate the `DynamicsEngine` (LAMMPS & EON) and the `HybridPotential` capability.

**Sections:**
1.  **Setup**: Load the `potential.yace` from Notebook 1.
2.  **Phase 1: Dynamic Deposition (MD)**:
    -   Setup LAMMPS `fix deposit`.
    -   Inject Fe and Pt atoms alternately onto the hot MgO substrate (600K).
    -   **Visualisation**: Use `ase.visualize.plot` to show the growth of the island.
    -   **Check**: Verify atoms do not penetrate the substrate (ZBL/Hybrid potential test).
3.  **Phase 2: Long-Term Ordering (aKMC)**:
    -   Take the final disordered cluster from MD.
    -   Initialize the EON client (or mock if EON not installed).
    -   Run aKMC to find lower-energy ordered states (L10 ordering).
    -   **Visualisation**: Compare initial (disordered) vs final (ordered) structure.
4.  **Analysis**:
    -   Calculate the "Order Parameter" (number of Fe-Pt bonds).
    -   Plot the energy evolution over time.

**Success Criteria:**
-   Deposition simulation finishes without "Lost Atoms" error (proving Hybrid Potential stability).
-   kMC step demonstrates energy minimization.
-   Visualisations clearly show the substrate and the cluster.

## 3. Validation Steps

The QA Agent (or human reviewer) should perform the following checks when running these notebooks:

1.  **Dependency Handling**:
    -   Uninstall `lammps` or `eon` and run the notebook. It should NOT crash but gracefully degrade (print warning and skip cell or use mock data).
    -   Install dependencies and run. It should execute the real physics.

2.  **File Artifacts**:
    -   After running NB01, check that `data/training_data.pckl.gzip` exists and is non-empty.
    -   After running NB02, check that `dump.lammps` and `client_log.txt` (EON log) are generated.

3.  **Scientific Sanity (Real Mode only)**:
    -   MgO Lattice Constant should be approx 4.21 Å.
    -   Fe-Fe minimal distance should be > 2.0 Å (no fusion).
    -   Total energy should be negative.

4.  **Time Constraints**:
    -   In CI Mode, the total execution time for both notebooks should be under 10 minutes.
