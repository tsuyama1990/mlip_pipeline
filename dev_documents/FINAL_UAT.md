# Final User Acceptance Testing (UAT) Plan

## 1. Tutorial Strategy

The primary vehicle for User Acceptance Testing and user onboarding will be a set of "Executable Scientific Papers" in the form of Jupyter Notebooks. These notebooks serve a dual purpose: verifying the system functionality (QA) and teaching users how to solve complex materials science problems (Tutorial).

### 1.1. The "Mock vs Real" Dual-Mode Strategy
Scientific simulations often require HPC resources and days of runtime, which is incompatible with Continuous Integration (CI) and quick user trials. To solve this, all tutorials must implement a dual-mode execution strategy controlled by an environment variable `IS_CI_MODE`.

*   **Real Mode (`IS_CI_MODE=False`)**:
    *   **Behavior**: Runs full DFT calculations (Quantum Espresso), trains on large datasets (thousands of structures), and performs long MD/kMC runs.
    *   **Target**: Users with HPC access and API keys/installed binaries.
    *   **Outcome**: Publication-quality plots and physically meaningful results.

*   **Mock Mode (`IS_CI_MODE=True`)**:
    *   **Behavior**:
        *   Skips actual DFT calls; uses pre-calculated "Toy Data" or simple analytical potentials (e.g., LJ) masked as DFT results.
        *   Reduces training epochs to 1 (proof of pipeline functionality).
        *   Uses tiny supercells (e.g., 2x2x2) and short MD runs (100 steps).
        *   Mocks long-running external processes (e.g., EON) by loading pre-generated output files.
    *   **Target**: CI pipelines (GitHub Actions) and users verifying installation.
    *   **Outcome**: Verifies that the *code* runs without crashing, files are created, and the API is correct.

## 2. Notebook Plan

We will deliver two comprehensive notebooks that cover the entire "Fe/Pt on MgO" Grand Challenge.

### 2.1. `tutorials/01_MgO_FePt_Training.ipynb`
**Title**: "Divide & Conquer: Active Learning of Heterogeneous Systems"
**Goal**: Demonstrate the `Orchestrator`, `Structure Generator`, `Oracle`, and `Trainer` components.

*   **Scenario**:
    1.  **Bulk & Surface Training (MgO)**:
        *   Initialize a `StructureGenerator` for MgO.
        *   Run Active Learning loop: Generate -> Label (DFT/Mock) -> Train.
        *   Validate phonons (ensure stability).
    2.  **Alloy Training (FePt)**:
        *   Initialize a separate generator for FePt (L10 ordering target).
        *   Run Active Learning loop.
    3.  **Interface Learning**:
        *   Create Fe/Pt clusters on MgO slabs.
        *   Run Active Learning to capture adhesion forces (preventing "floating" or "sinking" atoms).
*   **Key Validations**:
    *   Check that `config.yaml` is correctly parsed.
    *   Verify `pace_train` execution and `.yace` file generation.
    *   Verify Parity Plots (Energy/Force accuracy).

### 2.2. `tutorials/02_Deposition_and_Ordering.ipynb`
**Title**: "From Nanoseconds to Seconds: Hybrid MD & aKMC Simulation"
**Goal**: Demonstrate the `Dynamics Engine` (LAMMPS & EON Integration) and Scale-up capabilities.

*   **Scenario**:
    1.  **Hybrid MD Setup**:
        *   Load the potential trained in NB01.
        *   Configure `pair_style hybrid/overlay pace zbl` to ensure safety.
    2.  **Deposition (MD)**:
        *   Use `fix deposit` to drop Fe/Pt atoms onto MgO at 600K.
        *   **UAT Check**: Monitor "Extrapolation Grade" ($\gamma$). The simulation *should* halt if atoms hit high-uncertainty configurations, triggering the "Halt & Diagnose" handler (demonstrating robustness).
    3.  **Ordering (aKMC)**:
        *   Take the final disordered cluster from MD.
        *   Hand off to EON (Adaptive Kinetic Monte Carlo).
        *   Observe the transition from disordered fcc to ordered L10 phase (mocked in CI mode).
*   **Key Validations**:
    *   Visual check: Atoms must land on the surface, not fly away or fuse.
    *   Ordering Parameter: Calculate Fe-Pt vs Fe-Fe bonds.
    *   Physics Check: `pair_style hybrid` is active (check log file).

## 3. Validation Steps

The QA Agent (or human reviewer) must perform the following checks:

1.  **Environment Setup**:
    *   Run `uv sync --dev` and ensure no conflicts.
    *   Check `mlip-runner --help` works.

2.  **Mock Mode Execution**:
    *   Run `export IS_CI_MODE=true`
    *   Execute both notebooks: `pytest --nbval tutorials/` (or run manually).
    *   **Success**: All cells execute without error. Total runtime < 5 minutes.

3.  **Artifact Verification**:
    *   Check `potentials/` directory for generated `.yace` files.
    *   Check `active_learning/` directory for run logs.
    *   Check that `report.html` was generated.

4.  **Physics Sanity (Manual/Code Check)**:
    *   In NB02, verify that the `ZBL` potential is overlayed.
    *   Verify that the "Halt" mechanism is logged if triggered.
