# Cycle 08 Specification: Validation & Full Automation

## 1. Summary

Cycle 08 is the final stage of development, focusing on "Quality Assurance" and "Reporting". A Machine Learning Potential (MLIP) is useless if it predicts incorrect physics or is unstable, even if the training error (RMSE) is low.

The `Validator` component runs a battery of physical tests:
1.  **Phonon Dispersion**: Checks for dynamical stability (imaginary frequencies).
2.  **Elastic Constants**: Checks for mechanical stability (Born criteria) and stiffness (Bulk/Shear modulus).
3.  **Equation of State (EOS)**: Checks the energy-volume curve for smoothness and physically reasonable bulk modulus.
4.  **Melting Point**: Estimates $T_m$ to ensure the potential behaves correctly at high temperatures.

Additionally, this cycle implements the `ReportGenerator` to compile all metrics (Training curves, Validation results) into a human-readable HTML report. Finally, we integrate everything into the `Orchestrator` for the complete End-to-End workflow.

## 2. System Architecture

This cycle completes the `components/validator` package and finalizes the `core`.

### File Structure
Files to be created/modified in this cycle are marked in **bold**.

```
src/mlip_autopipec/
├── components/
│   ├── validator/
│   │   ├── __init__.py
│   │   ├── **base.py**             # Enhanced Abstract Base Class
│   │   ├── **phonon.py**           # Phonopy Interface
│   │   ├── **elastic.py**          # Elastic Tensor Calculator
│   │   ├── **eos.py**              # Birch-Murnaghan Fit
│   │   └── **report.py**           # HTML Report Generator
│   └── factory.py                  # Register Validator
├── domain_models/
│   └── **results.py**              # ValidationMetrics Pydantic Model
├── core/
│   └── **orchestrator.py**         # Final Polish (Calls Validator)
└── tests/
    └── **test_validator.py**
```

## 3. Design Architecture

### 3.1. Validator Configuration (`domain_models/config.py`)
Update `ValidatorConfig`:
*   `phonon_check`: bool.
*   `elastic_check`: bool.
*   `eos_check`: bool.
*   `melting_check`: bool.
*   `supercell_matrix`: List[int] (e.g., [2, 2, 2] for Phonons).

### 3.2. Validation Metrics (`domain_models/results.py`)
*   `ValidationMetrics`:
    *   `phonon_stable`: bool.
    *   `elastic_stable`: bool.
    *   `bulk_modulus`: float (GPa).
    *   `shear_modulus`: float (GPa).
    *   `eos_rmse`: float.
    *   `failed_structures`: List[Structure].

### 3.3. Phonon Calculator (`components/validator/phonon.py`)
*   Uses `phonopy` (if installed) or a simplified finite displacement method.
*   Computes the dynamical matrix and eigenvalues.
*   If any eigenvalue < -0.1 THz (imaginary), sets `phonon_stable = False`.

### 3.4. Report Generator (`components/validator/report.py`)
*   Reads `metrics.json` from each cycle directory.
*   Uses `pandas` and `matplotlib` to plot:
    *   Training Error vs Cycle.
    *   Data Count vs Cycle.
    *   Validation metrics.
*   Outputs `report.html`.

## 4. Implementation Approach

1.  **Implement `EOSCalc`**:
    *   Use `ase.eos`.
    *   Generate 10 volumes around equilibrium.
    *   Calculate Energy.
    *   Fit Birch-Murnaghan.
2.  **Implement `ElasticCalc`**:
    *   Apply strain tensors ($\pm 1\%$).
    *   Calculate Stress.
    *   Fit stiffness tensor.
3.  **Implement `PhononCalc`**:
    *   Create supercell.
    *   Displace atoms.
    *   Calculate Forces (using Potential).
    *   Compute frequencies.
4.  **Orchestrator Integration**:
    *   After Training (Cycle 04), run Validation.
    *   If Validation Fails (e.g., Unstable Phonon), feedback to Generator (Cycle 02) to sample that mode? (Advanced: Just Report for now).
    *   Generate Report at the end.

## 5. Test Strategy

### 5.1. Unit Testing
*   **EOS**:
    *   Input: Potential (mocked LJ).
    *   Action: `calculate_eos(Al_fcc)`.
    *   Assert: `bulk_modulus` approx 70 GPa.
*   **Elastic**:
    *   Input: Potential (mocked).
    *   Action: `calculate_elastic(Si_diamond)`.
    *   Assert: $C_{11}, C_{12}, C_{44}$ are positive.

### 5.2. Integration Testing
*   **Full Pipeline**:
    *   Run a short Orchestrator job (2 cycles).
    *   Assert: `report.html` is generated.
    *   Assert: `validation_metrics.json` exists in `potentials/`.
