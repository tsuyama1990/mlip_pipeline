# Cycle 08 Specification: Validator & Reporting

## 1. Summary
This cycle implements the "Gatekeeper" component: the `Validator`. While previous cycles focus on learning, this cycle focuses on quality assurance. A potential might have low RMSE (fitting error) but still predict unphysical behavior (e.g., imaginary phonon modes implying structural instability, or negative bulk modulus). The Validator runs a suite of physical tests—Phonon Dispersion, Elastic Constants, and Equation of State (EOS)—to certify the potential. It also acts as the "Reporter," aggregating metrics from all cycles (learning curves, uncertainty histograms) into a human-readable HTML dashboard, providing transparency to the user.

## 2. System Architecture

### 2.1. File Structure
files to be created/modified in this cycle are bolded.

```text
src/mlip_autopipec/
├── components/
│   ├── validator/
│   │   ├── __init__.py
│   │   ├── base.py                 # [CREATE] Abstract Base Class
│   │   ├── physics_calcs.py        # [CREATE] Phonon/Elastic/EOS logic
│   │   └── reporter.py             # [CREATE] HTML Generator
├── domain_models/
│   ├── config.py                   # [MODIFY] Add ValidationConfig
└── core/
    └── orchestrator.py             # [MODIFY] Integrate Validator into validate()
```

### 2.2. Component Interaction
1.  **`Orchestrator`** calls `validator.validate(potential, context=state)`.
2.  **`Validator`**:
    *   **`EOSCalc`**: Compresses/expands cell by $\pm 10\%$. Fits Birch-Murnaghan EOS. Checks if Bulk Modulus $B_0 > 0$.
    *   **`ElasticCalc`**: Applies small strains. Calculates stiffness tensor $C_{ij}$. Checks Born stability criteria.
    *   **`PhononCalc`** (via Phonopy): Calculates force constants. Checks for imaginary frequencies ($\omega^2 < 0$) at high symmetry points.
3.  **`Reporter`**:
    *   Reads `workflow_state.json` and `metrics.json`.
    *   Plots "RMSE vs Cycle".
    *   Plots "Parity Plot".
    *   Embeds EOS curves and Phonon band structures (images).
    *   Generates `report.html`.

## 3. Design Architecture

### 3.1. Domain Models

#### `config.py`
*   `ValidationConfig`:
    *   `phonon_supercell`: List[int] (e.g., [3, 3, 3])
    *   `elastic_strain_magnitude`: float (0.01)
    *   `eos_vol_range`: float (0.10)

### 3.2. Core Logic

#### `physics_calcs.py`
*   **Responsibility**: Run physical simulations using ASE calculators.
*   **`check_stability(atoms, calc)`**: Returns `True` if Born criteria satisfied.
*   **`check_phonons(atoms, calc)`**: Wrapper around `phonopy`. Returns `True` if stable.

#### `reporter.py`
*   **Responsibility**: Data visualization.
*   **Technology**: `matplotlib` for static plots, `pandas` for tables, `jinja2` (or simple string formatting) for HTML.

## 4. Implementation Approach

### Step 1: EOS Calculator
*   Implement `calculate_eos`.
*   Use `ase.eos.EquationOfState`.

### Step 2: Elastic Calculator
*   Implement `calculate_elastic_constants`.
*   Use `ase.elasticity`.
*   Implement checks for Cubic system ($C_{11}-C_{12} > 0$, etc.).

### Step 3: Phonon Calculator (Optional/Mock)
*   Phonopy is a heavy dependency.
*   Implement a "Mock Phonon" check first (always passes or fails based on config).
*   If `phonopy` is installed, use it.

### Step 4: Reporter
*   Implement `generate_report(state)`.
*   Create a simple HTML template.

### Step 5: Orchestrator Integration
*   Call validation after training.
*   If validation fails, mark state as `CONDITIONAL` or `FAILED` but do not necessarily stop the loop (unless critical).

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_physics_calcs.py`**:
    *   Test Born criteria function with known stable/unstable matrices.
    *   Test EOS fitting with perfect Lennard-Jones data.

### 5.2. Integration Testing
*   **`test_validator_full.py`**:
    *   Provide a "Good" potential (e.g., LJ for Argon).
    *   Assert Validation passes.
    *   Provide a "Bad" potential (e.g., repulsive only).
    *   Assert Validation fails (EOS check should fail, no minimum).
    *   Verify `report.html` is generated.
