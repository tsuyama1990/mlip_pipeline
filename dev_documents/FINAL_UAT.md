# Final User Acceptance Test (UAT) Plan

## 1. Tutorial Strategy

The primary vehicle for User Acceptance Testing will be a set of executable Jupyter Notebooks located in the `tutorials/` directory. These notebooks serve a dual purpose: they act as the definitive "How-To" guide for new users and as the automated integration test suite for the CI/CD pipeline.

### "Mock Mode" vs "Real Mode"
To ensure the UAT is robust yet practical for continuous integration, we implement a strict "Mock Mode" strategy.

-   **Mock Mode (CI/CD)**:
    -   **Trigger**: Activated when the environment variable `CI=true` or `PYACEMAKER_MOCK=true` is set.
    -   **Behavior**:
        -   **DFT**: Instead of calling `pw.x`, the `MockOracle` returns pre-calculated energies and forces (lookup table) or simple empirical potentials (LJ/EAM) disguised as DFT results.
        -   **Training**: The `MockTrainer` skips the actual `pace_train` command (which takes minutes/hours) and simply copies a pre-trained `potential.yace` to the output directory.
        -   **Dynamics**: MD runs for only 10-50 steps on a tiny system (e.g., 2x2x2 unit cell).
        -   **kMC**: The `EONWrapper` immediately returns a "saddle point found" status using a hardcoded reaction path.
    -   **Goal**: Verify the *logic* and *data flow* of the entire pipeline in under 5 minutes.

-   **Real Mode (User/Production)**:
    -   **Trigger**: Default behavior when no flag is set.
    -   **Behavior**: Runs actual Quantum Espresso, Pacemaker, LAMMPS, and EON calculations.
    -   **Goal**: Scientifically validate the results (e.g., formation energies, ordering parameters).

## 2. Notebook Plan

We will deliver three core notebooks that simulate the "Fe/Pt on MgO" scientific workflow.

### NB01: The Foundation - "Zero-Config" Training
**Filename**: `tutorials/01_MgO_FePt_Training.ipynb`
**Scenario**: "I want to create a potential for MgO and FePt alloy from scratch."
**Steps**:
1.  **Initialize**: Load `config.yaml` for a bulk MgO system.
2.  **Orchestrate**: Call `Orchestrator.run()` to start the active learning loop.
3.  **Visualization**: Plot the Energy/Force parity graphs from the `Validator`.
4.  **Repeat**: Do the same for FePt binary alloy.
**Mock Logic**: In Mock Mode, this notebook loads a pre-existing `dataset.pckl` and skips the loop, showing the final parity plot immediately.

### NB02: The Experiment - Interface & Deposition (MD)
**Filename**: `tutorials/02_Deposition_and_Interface.ipynb`
**Scenario**: "I want to simulate Fe and Pt atoms landing on an MgO substrate."
**Steps**:
1.  **Interface Training**: Define a slab geometry (MgO 001). Place Fe/Pt clusters on top. Run a short Active Learning cycle to learn the *adhesion* (interface) interaction.
2.  **Hybrid Setup**: Show how the `DynamicsEngine` automatically mixes the MgO potential, FePt potential, and the Interface correction.
3.  **Deposition**: Run LAMMPS with `fix deposit`. Visualize atoms raining down and diffusing on the surface.
**Success Criteria**: Fe/Pt atoms must *stick* to the surface but not *sink* into the MgO (Core Repulsion check).
**Mock Logic**: Loads a trajectory file (`dump.lammps`) from a previous successful run and animates it.

### NB03: The Long Game - Ordering (aKMC)
**Filename**: `tutorials/03_Ordering_aKMC.ipynb`
**Scenario**: "I want to see the disordered FePt cluster order into L10 phase over time."
**Steps**:
1.  **Handoff**: Take the final configuration from NB02.
2.  **EON Setup**: configure the `EONWrapper` to look for exchange events (swapping Fe and Pt).
3.  **Execution**: Run the `Orchestrator` in "kMC Mode".
4.  **Analysis**: Calculate the "Ordering Parameter" (number of Fe-Pt bonds vs Fe-Fe/Pt-Pt) and plot it against time.
**Success Criteria**: The ordering parameter should increase, indicating L10 formation.
**Mock Logic**: Returns a fake time-series data of the ordering parameter increasing.

## 3. Validation Steps (QA Instructions)

The QA Agent (or human reviewer) must verify the following:

### 3.1. Installation & Environment
-   `uv sync` must install all dependencies without conflict.
-   The command `pyacemaker --help` must run without error.

### 3.2. Notebook Execution
-   **Run All**: Execute `pytest --nbval tutorials/` (using the `nbval` plugin).
-   **Pass Criteria**: All cells must execute without raising exceptions.
-   **Visuals**: Parity plots and MD snapshots (PNG) must be generated and visible in the notebook output.

### 3.3. Scientific Sanity Checks (Real Mode Only)
-   **Energy Conservation**: In NVE MD, total energy drift should be < 1 meV/atom/ps.
-   **Symmetry**: The trained MgO potential should predict a cubic lattice parameter within 1% of the DFT input (approx 4.21 Å).
-   **Adhesion**: The Fe/MgO interface energy should be positive (stable interface) but not infinitely attractive.
-   **Ordering**: The aKMC simulation must show a preference for Fe-Pt neighbors (L10) over random mixing at lower temperatures.

### 3.4. Code Quality
-   Run `ruff check .` and `mypy .` (strict mode).
-   Ensure no "bare excepts" (`except:`) are used in the core logic.
-   Ensure all public methods have docstrings (Google/NumPy style).
