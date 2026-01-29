# Cycle 05 Specification: Validation Framework

## 1. Summary

Cycle 05 implements the **Validator** module, which acts as the Quality Assurance (QA) gate for the generated potentials. It goes beyond simple test-set errors (RMSE) and evaluates the physical validity of the potential. This includes calculating Phonon dispersion curves (to check for dynamic stability/imaginary frequencies), Elastic constants (mechanical stability), and Equation of State (EOS) curves. It generates a comprehensive HTML report summarizing these metrics.

## 2. System Architecture

### File Structure

Files to be created/modified are **bold**.

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── config.py                     # Update: Add ValidationConfig
├── modules/
│   └── **validator/**
│       ├── **__init__.py**
│       ├── **runner.py**             # Main Validator class
│       ├── **phonon.py**             # Phonopy integration
│       ├── **elasticity.py**         # Elastic constant checks
│       ├── **eos.py**                # EOS fitting
│       └── **reporting.py**          # HTML Report generator
└── orchestration/
    └── phases/
        ├── **__init__.py**
        └── **validation.py**         # ValidationPhase implementation
```

## 3. Design Architecture

### Domain Models

#### `config.py`
*   **`ValidationConfig`**:
    *   `run_phonon`: bool
    *   `run_elastic`: bool
    *   `run_eos`: bool
    *   `thresholds`: Dict (max RMSE, min elastic modulus)

### Components (`modules/validator/`)

#### `phonon.py`
*   **`PhononValidator`**:
    *   Uses `phonopy` via Python API.
    *   **`compute_band_structure(structure, potential)`**: Calculates phonon bands.
    *   **`check_stability()`**: Returns False if imaginary frequencies exist.

#### `elasticity.py`
*   **`ElasticityValidator`**:
    *   **`compute_elastic_constants(structure, potential)`**: Applies strains and fits stress-strain curves.
    *   **`check_born_criteria()`**: Verifies mechanical stability (e.g., C11 - C12 > 0).

#### `reporting.py`
*   **`ReportGenerator`**:
    *   Uses `jinja2` to render HTML templates.
    *   Aggregates plots (matplotlib/plotly) and metrics into `validation_report.html`.

### Orchestration (`orchestration/phases/validation.py`)

#### `ValidationPhase`
*   Runs after Training.
*   Executes enabled validators.
*   Decides **PASS/FAIL** based on thresholds.
*   If FAIL, marks the potential as rejected (or conditional) and logs reasons.

## 4. Implementation Approach

1.  **Update Config**: Add `ValidationConfig`.
2.  **Implement Physical Validators**:
    *   `PhononValidator`: Need to convert ASE atoms to Phonopy atoms.
    *   `ElasticityValidator`: Implement finite difference method for stiffness matrix.
    *   `EOSValidator`: Implement Birch-Murnaghan fit.
3.  **Implement Reporting**:
    *   Create a simple HTML template.
    *   Save plots as images or JSON for embedding.
4.  **Implement Validation Phase**:
    *   Execute validators in parallel if possible.
    *   Implement "Gatekeeper" logic (Pass/Fail).

## 5. Test Strategy

### Unit Testing
*   **`test_elasticity.py`**:
    *   Test with a known potential (e.g., LJ) on a simple lattice. Verify C11, C12 values match analytical results.
*   **`test_phonon.py`**:
    *   Mock Phonopy to return fake frequencies. Verify stability check logic.

### Integration Testing
*   **`test_validation_phase.py`**:
    *   Run the phase on a dummy potential.
    *   Verify `validation_report.html` is generated.
    *   Verify `WorkflowState` reflects the validation result.
