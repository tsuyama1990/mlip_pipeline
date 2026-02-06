# Cycle 08 Specification: Validation & Production Readiness

## 1. Summary

Cycle 08 is the final cycle, focusing on **Quality Assurance**. Before a potential is marked as "Production Ready," it must pass a battery of physical tests. It is not enough to have a low RMSE on the training set; the potential must reproduce fundamental physical properties like phonon stability, elastic constants, and the equation of state.

We will implement a **Validator** module that runs these checks automatically and generates a human-readable **HTML Report**.

## 2. System Architecture

Files to create/modify are **bolded**.

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── **validation.py**       # ValidationResult models
└── infrastructure/
    └── **validator.py**        # Implementation of physical checks
```

## 3. Design Architecture

### 3.1. Validation Data Models
*   `ValidationMetric`: Name, Value, Threshold, Pass/Fail.
*   `ValidationReport`: List[ValidationMetric], Plots (paths).

### 3.2. Validator Logic
*   **EOS Check**: Calculate E vs V curve. Fit Birch-Murnaghan. Check Bulk Modulus ($B_0$).
*   **Elastic Check**: Deform cell, compute stress. Fit Cij. Check Born stability criteria.
*   **Phonon Check**: (Optional, requires Phonopy) Calculate force constants. Check for imaginary frequencies.

## 4. Implementation Approach

1.  **EOS**: Implement `check_eos(potential, structure)`. Use ASE's `EquationOfState`.
2.  **Elasticity**: Implement `check_elasticity(potential, structure)`. Apply $\pm 1\%$ strain in Voigt notation.
3.  **Reporting**: Use `jinja2` to render an HTML template containing the plots and pass/fail tables.
4.  **Orchestrator Integration**: Call `Validator.validate()` at the end of each cycle. If it fails, log a warning (or Halt the whole pipeline if strict mode).

## 5. Test Strategy

### 5.1. Unit Testing
*   **Elasticity**: Create a dummy potential (LJ) that we know the answer for. Run the check. Assert $C_{11}$, $C_{12}$ are within expected range.

### 5.2. Integration Testing
*   **Report Gen**: Run the validator on a dummy potential. Assert `report.html` is generated.
*   **Pipeline**: Run the full "Fe/Pt" scenario. Assert that the final potential has a validation report attached.
