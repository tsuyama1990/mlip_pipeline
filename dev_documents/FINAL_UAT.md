# Final UAT Master Plan

## 1. Tutorial Strategy

The user acceptance testing (UAT) for PYACEMAKER is designed around "Executable Scientific Papers". Instead of dry manual test cases, we provide a series of Jupyter Notebooks that guide the user through a realistic research project: **Hetero-epitaxial Growth of Fe/Pt on MgO(001)**. This scenario was chosen because it exercises every component of the system:
-   **Multi-species**: Handling 3 elements (Fe, Pt, Mg, O) requires robust dataset management.
-   **Interfaces**: The interaction between the metal cluster and the oxide substrate tests the "Periodic Embedding" and "Force Masking" logic.
-   **Rare Events**: The ordering of Fe/Pt into the L10 phase is a slow process, ideal for showcasing the **Adaptive Kinetic Monte Carlo (aKMC)** integration.

### The "Mock vs Real" Strategy
To ensure these tutorials are accessible to all users (including those without HPC access) and verifiable in CI environments, every notebook will implement a dual-mode execution strategy:

-   **CI Mode (`IS_CI_MODE = True`)**:
    -   Uses "Mock" components or pre-calculated data.
    -   Runs on tiny supercells (e.g., 2x2x1 MgO).
    -   Deposits only ~5 atoms.
    -   Mocks long-running steps (e.g., "Load pre-trained potential" instead of training for hours).
    -   **Goal**: Verify the *code flow* and API correctness in < 5 minutes.

-   **Real Mode (`IS_CI_MODE = False`)**:
    -   Uses real DFT (Quantum Espresso), real Training (Pacemaker), and real MD (LAMMPS).
    -   Runs on large supercells (e.g., 6x6x2 MgO).
    -   Deposits 500+ atoms.
    -   **Goal**: Produce publication-quality scientific results.

## 2. Notebook Plan

The tutorials are split into logical scientific phases, mirroring the "Divide & Conquer" strategy essential for complex MLIP construction.

### `tutorials/01_MgO_FePt_Training.ipynb` (Phase 1: Component Training)
**Goal**: Create the "Base" potentials for the substrate and the deposit separately.
-   **Scenario**:
    1.  **MgO Bulk & Surface**: Generate MgO structures with oxygen vacancies. Run Active Learning to train a robust MgO potential.
    2.  **Fe-Pt Alloy**: Generate bulk L10, disordered fcc, and liquid structures. Train a metallic FePt potential.
-   **Key Features**: `StructureGenerator` (Bulk/Surface modes), `Trainer` (Active Set selection).
-   **Validation**:
    -   Check Lattice Constants of MgO vs Experiment.
    -   Check Formation Energy of L10-FePt.

### `tutorials/02_Interface_Learning.ipynb` (Phase 2: The Interface)
**Goal**: Teach the potential how Fe/Pt atoms interact with the MgO surface.
-   **Scenario**:
    1.  **Placement**: Place Fe and Pt clusters on the MgO(001) slab.
    2.  **Active Learning**: Run the "Orchestrator" loop. The system should detect high uncertainty at the interface.
    3.  **Embedding**: Visualize how the system cuts out the interface cluster for DFT (Periodic Embedding).
-   **Key Features**: `Oracle` (Embedding), `Orchestrator` (Loop).
-   **Validation**:
    -   Check Adhesion Energy ($E_{adhesion}$) of Fe on O-site vs Mg-site.
    -   Confirm no "hole" drilling (atoms sinking into the surface).

### `tutorials/03_Deposition_MD.ipynb` (Phase 3: Dynamic Deposition)
**Goal**: Simulate the physical vapor deposition (PVD) process.
-   **Scenario**:
    1.  **Setup**: Initialize a large MgO slab at 600K.
    2.  **Deposition**: Use LAMMPS `fix deposit` to drop Fe and Pt atoms.
    3.  **Observation**: Watch atoms diffuse and form islands.
-   **Key Features**: `Dynamics Engine` (LAMMPS integration, Hybrid Potential `pace + zbl`).
-   **Validation**:
    -   **Visual**: Islands should be 3D (Volmer-Weber growth) or 2D (Layer-by-layer), depending on surface energy.
    -   **Stability**: Monitor $T$ and $P$. No segmentation faults allowed!

### `tutorials/04_Ordering_aKMC.ipynb` (Phase 4: Long-Term Ordering)
**Goal**: Overcome the time-scale limitation to observe chemical ordering.
-   **Scenario**:
    1.  **Input**: Take the disordered cluster from the end of NB03.
    2.  **aKMC**: Run EON to explore diffusion barriers and find lower energy states.
    3.  **Result**: The cluster transforms into the chemically ordered L10 phase.
-   **Key Features**: `Dynamics Engine` (EON integration), `Orchestrator` (OTF loop for transition states).
-   **Validation**:
    -   **Order Parameter**: Calculate the number of Fe-Pt bonds vs Fe-Fe bonds. It should increase.

## 3. Validation Steps for QA

The QA Agent (or human reviewer) should perform the following checks when running the notebooks:

1.  **Environment Check**:
    -   Ensure `uv sync` installs all python dependencies.
    -   Ensure external binaries (`lmp`, `pw.x`, `pace_train`) are either in `$PATH` or the notebook handles their absence gracefully (skipping "Real Mode" cells).

2.  **Visual Verification**:
    -   **NB01**: Parity plots (DFT Energy vs Predicted Energy) should show a tight correlation ($R^2 > 0.99$).
    -   **NB03**: The trajectory movie (or snapshots) must show atoms *on top* of the surface, not flying away or merging into the substrate.

3.  **Physics Sanity Checks (Assertions)**:
    -   **Cohesion**: The potential energy of the cluster must be negative (stable).
    -   **Repulsion**: The minimum distance between any two atoms must be $> 1.8 \AA$ (approx). If $< 1.0 \AA$, the ZBL baseline failed.

4.  **Error Handling**:
    -   Trigger a "Fake Halt" (manually set a high uncertainty) in the MD loop and verify the Orchestrator catches it and prints "High Uncertainty Detected".

5.  **Artifact Generation**:
    -   Confirm that `potentials/` contains `.yace` files.
    -   Confirm that `dev_documents/validation_report.html` (if generated) opens and displays graphs.
