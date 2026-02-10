# Cycle 08: Validator & Quality Assurance

## 1. Summary

Cycle 08 is the final stage of the development process, focusing on the "Validator" component. The goal is to ensure that the generated machine-learned potential is not only accurate in reproducing the training data (low RMSE) but also physically robust and predictive for unknown configurations.

The `StandardValidator` will run a comprehensive suite of physical tests:
1.  **Phonon Stability**: Calculate the phonon dispersion relations to check for imaginary frequencies (instabilities).
2.  **Elastic Constants**: Calculate the elastic tensor ($C_{ij}$) and verify Born stability criteria.
3.  **Equation of State (EOS)**: Fit the Energy-Volume curve to the Birch-Murnaghan equation to extract the bulk modulus ($B_0$) and equilibrium volume ($V_0$).
4.  **Reporting**: Compile all results into an interactive HTML report (`validation_report.html`) for the user.

## 2. System Architecture

The following file structure will be modified/created. Files in **bold** are the primary deliverables for this cycle.

```
.
├── config.yaml
├── src/
│   └── mlip_autopipec/
│       ├── core/
│       │   ├── config.py         # Update ValidatorConfig
│       ├── components/
│       │   ├── base.py
│       │   └── **validator.py**  # Standard Validator
│       └── utils/
│           ├── **phonons.py**    # Phonopy Wrapper
│           ├── **elasticity.py** # Elastic Constant Calc
│           ├── **eos.py**        # Birch-Murnaghan Fit
│           └── **reporting.py**  # HTML Generator
└── tests/
    ├── **test_validator.py**
    ├── **test_phonons.py**
    ├── **test_elasticity.py**
    └── **test_eos.py**
```

## 3. Design Architecture

### Standard Validator (`components/validator.py`)
This class implements the `BaseValidator` interface.
*   `validate(potential)`:
    1.  Run `PhononCalc`.
    2.  Run `ElasticCalc`.
    3.  Run `EOSCalc`.
    4.  Run `TestSetCalc` (RMSE on held-out data).
    5.  Check against thresholds defined in `ValidatorConfig`.
    6.  Generate Report.
    7.  Return `ValidationResult(passed=True/False, reason=...)`.

### Physical Tests
*   `PhononCalc` (`utils/phonons.py`):
    *   Uses `phonopy` (optional dependency) or a simple finite displacement method if phonopy is absent.
    *   Checks for $\omega^2 < -10^{-5}$ (imaginary modes).
*   `ElasticCalc` (`utils/elasticity.py`):
    *   Applies strains ($\pm 0.01$) to the unit cell.
    *   Fits the stress-strain curve to get $C_{ij}$.
    *   Checks Born criteria (e.g., $C_{11} - C_{12} > 0$).
*   `EOSCalc` (`utils/eos.py`):
    *   Scales volume by $\pm 10\%$.
    *   Fits $E(V)$ to Birch-Murnaghan.
    *   Compares $B_0$ with reference (if available) or checks for convexity ($B_0 > 0$).

### Reporting (`utils/reporting.py`)
Generates an HTML file using `jinja2` templates and `matplotlib` plots (encoded as base64 strings).
*   Sections: Summary Table, EOS Plot, Phonon Band Structure, Parity Plots (Energy/Force).

## 4. Implementation Approach

1.  **Utilities**: Implement `utils/eos.py` (simplest), then `elasticity.py`, then `phonons.py`.
2.  **Validator**: Implement `components/validator.py` to orchestrate these tests.
3.  **Reporting**: Create a simple HTML template and the `Reporting` class.
4.  **Configuration**: Update `ValidatorConfig` with tolerances (e.g., `max_energy_rmse=0.002`).
5.  **Integration**: Update `Orchestrator` to call `validator.validate()` at the end of each cycle or upon request.

## 5. Test Strategy

### Unit Testing
*   **EOS Fit**: Provide ideal $E(V)$ data (e.g., $E = 0.5 k (V-V_0)^2$) and assert that $B_0$ is calculated correctly.
*   **Elasticity**: Provide a cubic crystal with known stiffness. Verify $C_{11}, C_{12}, C_{44}$.

### Integration Testing (System Level)
*   **Full Validation**:
    *   Train a potential (mock or real) on a perfect crystal.
    *   Run `validator.validate()`.
    *   Assert `passed=True`.
    *   Assert `validation_report.html` exists.
*   **Failure Case**:
    *   Train a potential on corrupted data (or use a bad mock).
    *   Run `validator.validate()`.
    *   Assert `passed=False` and `reason` contains "Instability detected".
