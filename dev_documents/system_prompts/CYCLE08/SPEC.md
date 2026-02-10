# Cycle 08 Specification: Validator & Quality Assurance

## 1. Summary

In this final cycle, we implement the **Quality Assurance (QA)** module. The `StandardValidator` runs a battery of physical tests to ensure that the trained potential is not only numerically accurate (low RMSE) but also physically meaningful. This includes checking for dynamical stability (Phonons), mechanical stability (Elastic Constants), and thermodynamic behavior (Equation of State). The results are compiled into a comprehensive HTML report.

## 2. System Architecture

The following file structure will be created. **Bold** files are the targets for this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── validation/
│   │   ├── **phonons.py**          # Wrapper for Phonopy
│   │   ├── **elasticity.py**       # Elastic Constant Calculation
│   │   ├── **eos.py**              # Birch-Murnaghan EOS Fit
│   │   ├── **report.py**           # HTML Report Generator
│   │   └── **base.py**             # (Update ABC with Validation logic)
tests/
└── **test_validator.py**           # Tests for Physics Calculations
```

## 3. Design Architecture

### 3.1. Validator Components (`components/validation/`)
*   **`StandardValidator`**:
    *   `validate(potential) -> ValidationReport`:
        1.  **Run Phonons**: `PhononCalc.run(structure, potential)`. Checks for imaginary frequencies ($\omega^2 < 0$).
        2.  **Run Elasticity**: `ElasticCalc.run(structure, potential)`. Checks Born stability criteria ($C_{ij} > 0$, etc.).
        3.  **Run EOS**: `EOSCalc.run(structure, potential)`. Fits energy vs volume curve.
        4.  **Aggregate**: Compile results into a `ValidationReport` object (PASS/FAIL/WARN).
        5.  **Report**: Generate `validation_report.html` using `pandas` and `matplotlib`.

### 3.2. Physics Calculators
*   **`PhononCalc`**:
    *   Uses `phonopy` (if installed) or a simple finite displacement method within ASE (`ase.phonons.Phonons`).
    *   Calculates band structure and DOS.
    *   Returns `stable: bool` (True if no imaginary modes at high symmetry points).
*   **`ElasticCalc`**:
    *   Applies strains ($\pm 1\%, \pm 2\%$) to the unit cell.
    *   Fits stress-strain curves to get $C_{11}, C_{12}, C_{44}$.
    *   Calculates Bulk Modulus ($B$) and Shear Modulus ($G$).
*   **`EOSCalc`**:
    *   Scales volume from $0.8 V_0$ to $1.2 V_0$.
    *   Fits Birch-Murnaghan equation.
    *   Returns $V_0, B_0, B'_0$.

## 4. Implementation Approach

1.  **Validator Config**: Add `ValidatorConfig` to `config.py`.
2.  **Calculators**: Implement `ElasticCalc` and `EOSCalc` using ASE (no external heavy dependencies). Implement `PhononCalc` wrapping `phonopy` (optional dependency).
3.  **Reporting**: Create `ReportGenerator` class to render HTML with embedded plots (base64 images).
4.  **Integration**: Add `validator` step to `Orchestrator`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_elasticity.py`**:
    *   Test calculation on a known potential (e.g., LJ argon). Verify $C_{11}$ is reasonable.
*   **`test_eos.py`**:
    *   Test EOS fit on a simple Lennard-Jones solid. Verify $B_0$ matches analytical value.

### 5.2. Integration Testing
*   **`test_validator.py`**:
    *   **Mock Potential**: Use a simple effective potential (e.g., EMT).
    *   **Verify Report**: Check that `validation_report.html` is generated and contains the words "PASS" or "FAIL".
