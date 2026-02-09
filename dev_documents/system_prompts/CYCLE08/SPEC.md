# Cycle 08 Specification: Validator & Reporting

## 1. Summary
The "Validator" is the final gatekeeper of the MLIP production line. It ensures that the generated potential is not only accurate on the training set (low RMSE) but also physically robust and reliable. This cycle implements a suite of physical tests—including phonon dispersion stability, elastic constant calculations, and Equation of State (EOS) fitting—and compiles the results into a comprehensive HTML report. Only potentials that pass all "Critical" tests are promoted to production status.

## 2. System Architecture

Files in **bold** are to be created or modified.

```ascii
src/mlip_autopipec/
├── components/
│   ├── validator/
│   │   ├── **__init__.py**
│   │   ├── **base.py**             # BaseValidator (ABC)
│   │   ├── **standard_validator.py** # Main Implementation
│   │   ├── **phonon.py**           # Phonon Calculator
│   │   ├── **elastic.py**          # Elastic Tensor Calculator
│   │   ├── **eos.py**              # Birch-Murnaghan Fit
│   │   └── **report_generator.py** # HTML Report Builder
│   └── ...
└── core/
    └── **orchestrator.py**         # Update to use validator
```

## 3. Design Architecture

### 3.1 Standard Validator (`standard_validator.py`)

This class coordinates the execution of sub-tests.

**Configuration (`ValidatorConfig`):**
*   `phonon_displacement`: float (e.g., 0.01 Å).
*   `eos_strain_range`: float (e.g., 0.15).
*   `elastic_strain`: float (e.g., 0.01).

**Workflow:**
1.  **Initialize**: Load the potential and reference structures (equilibrium bulk).
2.  **Run Tests**:
    *   `PhononCalculator.check_stability()`: Returns boolean + max imaginary freq.
    *   `ElasticCalculator.compute_moduli()`: Returns Bulk ($B$) and Shear ($G$) moduli.
    *   `EOSCalculator.fit_curve()`: Returns $V_0, E_0, B_0, B'_0$.
3.  **Evaluate**: Compare results against reference values (DFT or Experiment) or physical constraints (Born stability).
4.  **Report**: Generate `validation_report.html`.

### 3.2 Phonon Calculator (`phonon.py`)

Wraps `phonopy` via ASE interface.

**Logic:**
1.  Create supercell (e.g., 2x2x2).
2.  Apply finite displacements.
3.  Compute forces using ACE potential.
4.  Calculate force constants and phonon band structure.
5.  **Check**: If any frequency at any q-point is significantly imaginary (< -0.1 THz), fail the test.

### 3.3 Elastic Calculator (`elastic.py`)

Computes the $6 \times 6$ elastic stiffness tensor ($C_{ij}$).

**Logic:**
1.  Apply small strains to the unit cell.
2.  Compute stress tensor.
3.  Fit stress-strain curve to obtain $C_{ij}$.
4.  **Check**: Verify Born stability criteria (matrix positive definiteness).

## 4. Implementation Approach

1.  **Implement Calculators**: Create `phonon.py`, `elastic.py`, `eos.py`. Each should take an `ase.Atoms` and a `Calculator` (ACE) and return structured results.
2.  **Implement Report Generator**: Use `jinja2` or simple string formatting to create an HTML page with plots (embedded base64 PNGs from `matplotlib`).
3.  **Implement Validator**: Wire the calculators together. Define pass/fail logic.
4.  **Integration**: Update `Orchestrator` to call `Validator.validate()`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **EOS Fit**: Provide perfect E-V data (from known B-M equation). Verify fitted $B_0$ matches input.
*   **Elastic Logic**: Provide stress-strain data for a known cubic crystal. Verify calculated $C_{11}, C_{12}, C_{44}$ are correct.

### 5.2 Integration Testing
*   **Full Validation Run (Mock)**:
    *   Use a dummy potential (e.g., LJ/EMT) as the "ACE" potential.
    *   Run `Validator.validate()`.
    *   Verify:
        *   Phonons are calculated (no crash).
        *   Elastic constants are reasonable.
        *   `validation_report.html` is generated.
        *   The report contains plots (check file size > 0).
