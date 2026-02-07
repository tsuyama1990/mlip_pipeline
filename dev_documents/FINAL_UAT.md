# Final User Acceptance Testing (UAT) Plan

## 1. Tutorial Strategy

The goal of the UAT is to demonstrate the **PYACEMAKER** system's capability to solve real-world materials science problems, specifically the "Grand Challenge": Hetero-epitaxial Growth of FePt magnetic nanoparticles on an MgO substrate.

To achieve this, we will implement a "Scientific Workflow" approach, breaking the complex problem into logical, executable Jupyter Notebooks. These tutorials serve dual purposes:
1.  **User Education**: Teaching users how to use the system step-by-step.
2.  **System Verification**: Acting as an automated test suite for the entire pipeline.

### The "Mock Mode" Strategy (CI/CD Friendly)
Scientific simulations are computationally expensive. To ensure the tutorials can run in a CI environment (GitHub Actions) without requiring a supercomputer or hours of runtime, we will implement a `IS_CI_MODE` flag.

-   **Real Mode (`IS_CI_MODE = False`)**:
    -   Uses full-scale system sizes (e.g., 500+ atoms).
    -   Runs long MD/aKMC simulations (ns to seconds).
    -   Performs actual DFT calculations via Quantum Espresso.
-   **Mock Mode (`IS_CI_MODE = True`)**:
    -   Uses tiny supercells (e.g., 2x2x1 MgO, < 10 atoms).
    -   Runs very short MD (100 steps).
    -   **Mocks external binaries**: If `pw.x` or `lmp_serial` are missing, the notebook will use pre-calculated data or simple analytical potentials (LJ) to proceed without crashing.
    -   Skips heavy validation steps but verifies the *workflow logic*.

## 2. Notebook Plan

We will deliver 4 sequential notebooks in the `tutorials/` directory.

### NB01: Pre-Training (The Foundation)
**Goal**: Generate and train separate potentials for the bulk materials (MgO substrate and FePt alloy).
-   **Scenario**:
    -   Define `MgO` (rocksalt) and `FePt` (L10 ordered alloy).
    -   Use `StructureGenerator` to create random/rattled structures.
    -   Run a short Active Learning loop to fit `MgO.yace` and `FePt.yace`.
-   **Key Feature Verified**: `StructureGenerator`, `Oracle` (DFT), `Trainer` (Pacemaker).

### NB02: Interface Learning (The Glue)
**Goal**: Train the interaction between the deposit and the substrate.
-   **Scenario**:
    -   Create a slab of MgO(001).
    -   Place Fe and Pt clusters on top.
    -   Run Active Learning to capture adhesion energy and interface forces.
    -   **Merge** the potentials: $V_{total} = V_{MgO} + V_{FePt} + V_{interface}$.
-   **Key Feature Verified**: Complex structure generation, Multi-component training.

### NB03: Deposition MD (The Process)
**Goal**: Simulate the physical deposition process.
-   **Scenario**:
    -   Load the trained potential.
    -   Use LAMMPS `fix deposit` to drop Fe/Pt atoms onto the heated MgO substrate (600K).
    -   Observe surface diffusion and island nucleation.
    -   **Critical Check**: Ensure atoms do not sink *into* the substrate (Core Repulsion verify).
-   **Key Feature Verified**: `DynamicsEngine` (MD), Hybrid Potential (`pair_style hybrid/overlay`).

### NB04: Ordering aKMC (The Long-Term Evolution)
**Goal**: Observe the chemical ordering (L10 phase formation) which occurs on a longer timescale.
-   **Scenario**:
    -   Take the final state from NB03 (disordered island).
    -   Pass the structure to **EON** (Adaptive Kinetic Monte Carlo).
    -   Run saddle point searches to find lower-energy configurations (ordering).
-   **Key Feature Verified**: `DynamicsEngine` (aKMC), Python-EON integration.

## 3. Validation Steps

The QA Agent (or human reviewer) should verify the following when running the notebooks:

### 3.1. General Checks
-   **No Crashing**: All cells must execute without error (handling missing dependencies gracefully).
-   **Visualization**: The notebook must produce inline images (PNG/SVG) of the atomic structures (e.g., using `ase.visualize.plot`).
-   **Logs**: The output must show clear logs from the Orchestrator (e.g., "Cycle 1: Exploration...", "Cycle 1: Training...").

### 3.2. Scientific Validity Checks
-   **Energy Conservation**: In MD (NVE), total energy should be conserved.
-   **Geometry**:
    -   MgO lattice parameter should be approx 4.21 Å.
    -   Fe-Pt bond lengths should be physical (~2.6-2.7 Å).
-   **Ordering**: The final structure in NB04 should show a higher number of Fe-Pt bonds (L10 ordering) compared to the initial random alloy.

### 3.3. File Artifacts
-   Verify that `.yace` potential files are generated in `potentials/`.
-   Verify that `validation_report.html` is generated after training.
