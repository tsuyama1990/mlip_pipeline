# Final User Acceptance Test (UAT) Plan

## 1. Tutorial Strategy

The User Acceptance Testing strategy for PYACEMAKER is centered around "Executable Tutorials" in the form of Jupyter Notebooks. These notebooks serve a dual purpose: they act as the primary documentation for new users (The "Aha!" moment) and as the integration test suite for the Quality Assurance (QA) team.

### 1.1 "Mock Mode" vs "Real Mode"

To ensure that UAT can be performed even in environments without heavy computational resources or specific licenses (like VASP), the system supports two modes:

*   **Mock Mode (CI/Demonstration)**:
    *   **Description**: All external physics calls (DFT, LAMMPS, Pacemaker) are intercepted.
    *   **Behavior**: The system reads pre-calculated output files from a `tests/data` cache instead of running the actual binary.
    *   **Use Case**: CI pipelines, Quickstart verification on laptops without QE/LAMMPS.
    *   **Verification**: Ensures the Python logic (Orchestrator, File I/O, State Machine) works perfectly.

*   **Real Mode (Production)**:
    *   **Description**: The system executes actual binaries (`pw.x`, `lmp_serial`, `pace_train`).
    *   **Use Case**: Final release validation, Advanced tutorials for users with HPC access.
    *   **Verification**: Proves the physics and data integrity are correct.

## 2. Notebook Plan

We will deliver three key notebooks in the `tutorials/` directory.

### Tutorial 01: The "Zero-Config" Experience (Silicon)
*   **Filename**: `tutorials/01_quickstart_silicon.ipynb`
*   **Target Audience**: Absolute beginners.
*   **Goal**: Demonstrate the "One Command" promise.
*   **Scenario**:
    1.  User initializes a project for Bulk Silicon.
    2.  User views the generated `config.yaml` (simple default).
    3.  User runs the Orchestrator.
    4.  System runs 1 cycle of Active Learning (Exploration -> DFT -> Train).
    5.  User sees a "Validation Report" link appearing.
*   **Constraint**: Must run in "Mock Mode" for instant gratification (under 2 minutes).

### Tutorial 02: Advanced Oxide System (TiO2)
*   **Filename**: `tutorials/02_advanced_tio2.ipynb`
*   **Target Audience**: Power users / Researchers.
*   **Goal**: Demonstrate robustness and the "Hybrid Potential" feature.
*   **Scenario**:
    1.  User configures a Titania (TiO2) system.
    2.  User adjusts the `AdaptivePolicy` to prioritize "High Temperature" exploration (Simulation of melt).
    3.  User observes how the system handles a "Halt" event (high uncertainty) and recovers.
    4.  User analyzes the generated `potential.yace` vs a standard potential.
*   **Constraint**: Requires "Real Mode" or extensive cached data.

### Tutorial 03: Post-Analysis & Validation
*   **Filename**: `tutorials/03_validation_suite.ipynb`
*   **Target Audience**: Quality Assurance.
*   **Goal**: Deep dive into the Validator module.
*   **Scenario**:
    1.  Load a pre-trained potential.
    2.  Run Phonon Dispersion calculation.
    3.  Run Elastic Constant calculation.
    4.  Visualize the results interactively within the notebook.

## 3. Validation Steps for QA Agent

When the QA agent (or human tester) runs these notebooks, they should verify the following:

1.  **No Exceptions**: Cells must execute sequentially without Python errors.
2.  **Output Existence**:
    *   `active_learning/iter_XXX/` directories are created.
    *   `potentials/generation_XXX.yace` exists.
    *   `validation_report.html` is generated.
3.  **Mock Integrity**: In Mock Mode, verify that no `pw.x` or `lmp` processes are actually spawned in the OS process list (checking logs).
4.  **Visuals**: Ensure that plots (matplotlib/plotly) are rendered inline in the notebook.
