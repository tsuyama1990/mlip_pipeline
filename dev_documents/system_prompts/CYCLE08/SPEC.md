# Cycle 08 Specification: Validator & Reporting

## 1. Summary

Cycle 08 implements the **Validator** and **Report Generator** modules. These components are critical for ensuring the scientific quality and transparency of the generated potentials. While previous cycles focused on numerical accuracy (RMSE), this cycle evaluates the *physical validity* of the potential:
1.  **Phonon Stability**: Ensuring no imaginary frequencies exist (dynamical stability).
2.  **Elastic Constants**: Verifying mechanical stability (Born criteria) and reasonable bulk/shear moduli.
3.  **Equation of State (EOS)**: Checking the energy-volume curve for smoothness and correct equilibrium volume.
4.  **Reporting**: Aggregating all metrics into a comprehensive HTML report for user review.

This cycle acts as the "Quality Gate". If validation fails, the potential is flagged, and the active learning loop may be adjusted (e.g., adding strain-distorted structures).

## 2. System Architecture

We expand the `components/validator` module and introduce reporting tools.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```
.
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   └── **config.py**      # Add ValidatorConfig
│       ├── components/
│       │   ├── **validator/**
│       │   │   ├── **__init__.py**
│       │   │   ├── **base.py**        # BaseValidator (Abstract)
│       │   │   ├── **phonon.py**      # Phonon Dispersion Calculator
│       │   │   ├── **elastic.py**     # Elastic Constants Calculator
│       │   │   ├── **eos.py**         # Birch-Murnaghan Fit
│       │   │   └── **report_gen.py**  # HTML Report Builder
│       │   └── **factory.py**         # Update for Validator
│       └── utils/
│           └── **plotting.py**        # Matplotlib helpers
```

### Key Components
1.  **PhononCalc (`src/mlip_autopipec/components/validator/phonon.py`)**: Wrapper for `phonopy` (or `ase.phonons`). Calculates force constants and band structures. Checks for imaginary frequencies ($\omega^2 < 0$).
2.  **ElasticCalc (`src/mlip_autopipec/components/validator/elastic.py`)**: Applies small strains to the unit cell, computes stress, and fits the elastic stiffness tensor ($C_{ij}$). Checks Born stability criteria.
3.  **EOSCalc (`src/mlip_autopipec/components/validator/eos.py`)**: Computes energy vs. volume for a range of compressions/expansions. Fits to Birch-Murnaghan EOS.
4.  **ReportGenerator (`src/mlip_autopipec/components/validator/report_gen.py`)**: Reads all validation results and training metrics (from previous cycles) to generate a self-contained HTML file with interactive plots.

## 3. Design Architecture

### 3.1. Domain Models
*   **ValidatorConfig**:
    *   `phonon`: Dict (supercell, mesh, path).
    *   `elastic`: Dict (strain_max, n_steps).
    *   `eos`: Dict (vol_range, n_points).
*   **ValidationResult**:
    *   `phonon_stable`: Boolean.
    *   `elastic_stable`: Boolean.
    *   `bulk_modulus`: Float (GPa).
    *   `shear_modulus`: Float (GPa).
    *   `equilibrium_volume`: Float ($\AA^3$).
    *   `plots`: List[Path] (paths to generated images).

### 3.2. Phonon Stability Logic
*   **Method**: Finite Displacement.
*   **Criteria**: If any frequency at any q-point is less than a small negative threshold (e.g., -0.1 THz), the structure is dynamically unstable. Note: Gamma point acoustic modes are always 0, so small numerical noise must be ignored.

### 3.3. Elastic Constants Logic
*   **Method**: Stress-strain method. Apply $\epsilon = \pm 0.01$ in Voigt notation (1-6).
*   **Criteria**: Born stability conditions depend on crystal symmetry (e.g., Cubic: $C_{11}-C_{12} > 0$, $C_{11}+2C_{12} > 0$, $C_{44} > 0$).

## 4. Implementation Approach

1.  **Dependencies**: `phonopy` (optional, fallback to simple ASE phonons), `matplotlib`, `jinja2`.
2.  **EOS**: Implement `EOSCalc`. Use `ase.eos`.
3.  **Elastic**: Implement `ElasticCalc`. Use `ase.elasticity` or `pymatgen.analysis.elasticity`.
4.  **Phonon**: Implement `PhononCalc`. Use `phonopy` via API if installed. If not, skip or use a mock.
5.  **Reporting**: Create a Jinja2 template (`report_template.html`). Implement `ReportGenerator` to render it.

## 5. Test Strategy

### 5.1. Unit Testing
*   **EOS**: Verify fitting logic on dummy E-V data.
*   **Elastic**: Verify matrix inversion and Voigt-Reuss-Hill averaging on a known stiffness tensor.
*   **Report**: Verify HTML generation with dummy data.

### 5.2. Integration Testing (Mock Validation)
*   **Goal**: Verify the pipeline runs validation and generates a report.
*   **Procedure**:
    1.  Run the Orchestrator with `validator.type: standard`.
    2.  Use a Mock Potential (simple LJ or EMT) that is known to be stable for FCC metals.
    3.  Verify that `validation_report.html` is created.
    4.  Verify that stability checks pass.
*   **Fail Case**: Use an unstable potential (e.g., repulsive only) and verify that validation fails and the error is logged.
