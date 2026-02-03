# Final User Acceptance Testing (UAT) Master Plan

## 1. Tutorial Strategy

The primary goal of the User Acceptance Testing (UAT) is to demonstrate that PYACEMAKER can solve a "Grand Challenge" problem—specifically, the Hetero-epitaxial Growth and Ordering of Fe/Pt on MgO(001)—while remaining accessible to non-experts.

To achieve this, we will adopt a "Dual-Mode" strategy for all tutorials:

### 1.1 The "Mock Mode" (CI/CD & Fast Verification)
-   **Trigger**: Activated when the environment variable `CI=true` or `PYACEMAKER_MOCK_MODE=1` is set.
-   **Behavior**:
    -   Uses tiny supercells (e.g., $2 \times 2 \times 1$ MgO slab).
    -   Reduces training epochs to 1 (or loads pre-calculated weights).
    -   Mocks external long-running processes (DFT, EON) by loading cached outputs from a `tests/data` directory.
    -   **Goal**: Ensure the *code logic* (Orchestrator -> Component -> File I/O) works end-to-end in under 5 minutes without requiring an HPC scheduler.

### 1.2 The "Real Mode" (User Experience)
-   **Trigger**: Default behavior when running on a workstation with `pw.x` (Quantum Espresso) and `lammps` installed.
-   **Behavior**:
    -   Executes actual DFT calculations.
    -   Runs long MD simulations (100ps+).
    -   Performs real Active Learning loops.
    -   **Goal**: Produce scientifically valid results (e.g., correct lattice constants, stable deposition) that match literature values.

## 2. Notebook Plan

We will deliver a set of Jupyter Notebooks in the `tutorials/` directory. These notebooks serve as both the UAT test cases and the user documentation.

### 2.1 `tutorials/01_quickstart_silicon.ipynb` (The "Aha!" Moment)
**Objective**: Prove that the system works for a simple, single-element system.
**Scenario**:
-   **Input**: "Silicon (Diamond structure)".
-   **Action**:
    1.  Generate random distorted structures.
    2.  Run a mock DFT (or real DFT) to get forces.
    3.  Train a simple ACE potential.
    4.  Validate against elastic constants.
-   **Success Criteria**: The code runs to completion, and the final potential predicts the bulk modulus of Silicon within 10% of the reference.

### 2.2 `tutorials/02_advanced_tio2.ipynb` (The Active Learning Demo)
**Objective**: Demonstrate the "Self-Healing" capabilities (Halt & Diagnose).
**Scenario**:
-   **Input**: "TiO2 (Rutile)".
-   **Action**:
    1.  Start MD with a poor initial potential.
    2.  **Expectation**: The simulation halts due to high uncertainty ($\gamma > 5.0$).
    3.  **Automatic Recovery**: The notebook shows the Orchestrator catching the halt, running DFT, and updating the potential.
    4.  **Resume**: The simulation continues past the previous failure point.
-   **Success Criteria**: The notebook logs explicitly show "Halt triggered" followed by "Resuming simulation".

### 2.3 `tutorials/03_validation_suite.ipynb` (Quality Assurance)
**Objective**: Deep dive into the validation metrics.
**Scenario**:
-   **Input**: A pre-trained Copper potential.
-   **Action**:
    1.  Calculate Phonon dispersion curves (using Phonopy interface).
    2.  Calculate Elastic Constants ($C_{11}, C_{12}, C_{44}$).
    3.  Calculate Equation of State (EOS).
-   **Success Criteria**: Generate professional HTML reports and plots (Phonon band structure) directly in the notebook.

### 2.4 `tutorials/04_grand_challenge_fept.ipynb` (The Scientific Showcase)
**Objective**: Simulate the Fe/Pt deposition on MgO (The main User Scenario).
**Scenario**:
-   **Phase 1: Divide & Conquer Training**:
    -   Train MgO bulk/surface potential.
    -   Train FePt alloy potential.
    -   Train Interface (Fe/Pt clusters on MgO).
-   **Phase 2: Hybrid Deposition (MD)**:
    -   Use `pair_style hybrid/overlay` (ACE + ZBL).
    -   Simulate Fe and Pt atoms raining onto the substrate at 600K.
    -   Observe island nucleation.
-   **Phase 3: Long-Term Ordering (aKMC)**:
    -   Take the final MD structure.
    -   Bridge to EON (or Mock EON) to find the L10 ordered phase.
-   **Success Criteria**:
    -   Visualisation of atoms adhering to the surface (not flying away or sinking in).
    -   Identification of L10 ordering (alternating Fe/Pt layers) in the final structure.

## 3. Validation Steps (QA Checklist)

The Quality Assurance agent (or human reviewer) must verify the following:

### 3.1 Installation & Dependency Check
-   [ ] `uv sync` installs all dependencies without conflict.
-   [ ] `import mlip_autopipec` works without errors.
-   [ ] The system correctly identifies missing external tools (QE, LAMMPS) and warns the user instead of crashing (graceful degradation).

### 3.2 Feature Verification
-   [ ] **Config Parsing**: Can the system read a complex `config.yaml`?
-   [ ] **Mock Oracle**: Does the Mock Oracle generate consistent "fake" forces?
-   [ ] **Restart Capability**: If a job is interrupted, can it resume from `loop_state.json`?
-   [ ] **Artifact Generation**: Does the run produce a `production_potential.zip` containing the potential, manifest, and report?

### 3.3 Scientific Validity (Real Mode Only)
-   [ ] **MgO Stability**: The substrate must remain crystalline at 300K.
-   [ ] **No Fusion**: Minimum interatomic distance must be $> 1.5 \AA$ at all times (ZBL check).
-   [ ] **Convergence**: The total energy error on the test set must be $< 2$ meV/atom.

### 3.4 Documentation
-   [ ] The `README.md` clearly explains how to run the tutorials.
-   [ ] The notebooks contain markdown cells explaining the "Physics" behind each step, not just code.
