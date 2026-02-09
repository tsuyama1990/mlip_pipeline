# Cycle 08: Validator & Quality Assurance Specification

## 1. Summary

Cycle 08 completes the PyAceMaker system by implementing the **Validator** module. This component acts as the final "Quality Gate" before a potential is marked as production-ready. It goes beyond simple numerical error metrics (RMSE) on a test set and evaluates the physical properties of the generated potential. Specifically, it calculates phonon dispersion relations to ensure dynamic stability, elastic constants to verify mechanical stiffness, and equations of state (EOS) to check thermodynamic behavior under pressure. The results are compiled into a comprehensive HTML report, allowing users to visually inspect the quality of the potential.

## 2. System Architecture

This cycle focuses on the `components/validator` package and integration with `phonopy` and `ase.eos`.

### File Structure

The following file structure will be created. **Bold** files are to be implemented in this cycle.

*   **`src/`**
    *   **`mlip_autopipec/`**
        *   **`components/`**
            *   **`validator/`**
                *   **`__init__.py`**
                *   **`base_validator.py`** (Abstract Base Class)
                *   **`standard_validator.py`** (Main Implementation)
                *   **`phonon_calc.py`** (Phonopy Wrapper)
                *   **`elastic_calc.py`** (Elastic Constant Calculator)
                *   **`eos_calc.py`** (Equation of State Calculator)
                *   **`report_generator.py`** (HTML Report Builder)
        *   **`domain_models/`**
            *   **`results.py`** (ValidationMetrics)

## 3. Design Architecture

### 3.1 Components

#### `BaseValidator`
Defines the standard interface for validation suites.
*   **`validate(potential: PotentialArtifact) -> ValidationResult`**:
    *   Input: A trained potential.
    *   Output: A `ValidationResult` containing pass/fail status and detailed metrics.

#### `StandardValidator`
The default validation suite.
*   **`__init__(config: ValidatorConfig)`**: Sets up thresholds (e.g., max imaginary frequency).
*   **`run_suite(potential: PotentialArtifact) -> ValidationResult`**:
    1.  **Test Set**: Calculates RMSE on held-out data.
    2.  **Phonons**: Calls `PhononCalc`. Checks for imaginary modes.
    3.  **Elastic**: Calls `ElasticCalc`. Checks Born stability criteria.
    4.  **EOS**: Calls `EOSCalc`. Checks bulk modulus ($B_0$).
    5.  **Report**: Generates HTML summary.

#### `PhononCalc`
Wrapper around `phonopy`.
*   **`calculate(structure: Structure, potential: PotentialArtifact) -> PhononResult`**:
    *   Calculates force constants using finite displacement.
    *   Computes band structure.
    *   Returns max imaginary frequency (should be ~0).

#### `ElasticCalc`
Calculates elastic tensor ($C_{ij}$).
*   **`calculate(structure: Structure, potential: PotentialArtifact) -> ElasticResult`**:
    *   Applies small strains ($\pm \epsilon$).
    *   Fits stress-strain curves.
    *   Returns $C_{11}, C_{12}, C_{44}$, etc.

#### `ReportGenerator`
Generates visual reports.
*   **`generate_html(metrics: ValidationMetrics, plots: dict[str, Path]) -> Path`**:
    *   Uses Jinja2 templates or simple string formatting.
    *   Embeds plots (PNG/SVG) of phonon bands and EOS curves.

### 3.2 Domain Models

*   **`ValidationMetrics`**:
    *   `phonon_stable: bool`
    *   `elastic_stable: bool`
    *   `bulk_modulus: float`
    *   `eos_rmse: float`
    *   `test_set_rmse_E: float`
    *   `test_set_rmse_F: float`

## 4. Implementation Approach

1.  **Phonopy Integration**: Implement `phonon_calc.py`. Use `phonopy` via its Python API if available, or CLI. Ensure supercells are large enough (e.g., 2x2x2).
2.  **Elastic Logic**: Implement `elastic_calc.py` using `ase.constraints.StrainFilter` or manual deformation.
3.  **EOS Logic**: Implement `eos_calc.py` using `ase.eos.EquationOfState`.
4.  **Validator Logic**: Implement `StandardValidator.run_suite`. Orchestrate the calls and aggregate results.
5.  **Reporting**: Implement `report_generator.py`. Use `matplotlib` to generate plots and save them to a `report/` directory.
6.  **Configuration**: Update `config.py` with `ValidatorConfig`.
7.  **Factory**: Register `StandardValidator` in `ComponentFactory`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **`test_eos_calc.py`**:
    *   Create a dummy list of volumes and energies (Birch-Murnaghan equation).
    *   Call `fit(volumes, energies)`.
    *   Assert that $B_0$ and $V_0$ are close to expected values.
*   **`test_elastic_calc.py`**:
    *   Mock the stress calculation (return stress = C * strain).
    *   Call `calculate`.
    *   Assert that the fitted C matrix matches the input C.

### 5.2 Integration Testing (Mocked)
*   **Full Validation Run**:
    *   Mock `PhononCalc` to return stable bands.
    *   Mock `ElasticCalc` to return valid constants.
    *   Call `validator.validate(potential)`.
    *   Assert that `ValidationResult.passed` is True.
    *   Assert that `validation_report.html` is created.

### 5.3 System Testing
*   **Real Validation**:
    *   Requires `phonopy` installed.
    *   Run validation on a known good potential (e.g., Al_zbl.yace if available, or simple LJ).
    *   Verify that phonon dispersion curves look reasonable (acoustic modes go to 0 at Gamma).
