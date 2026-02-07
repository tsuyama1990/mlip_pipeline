# Cycle 06 Specification: Validation, Robustness & Final Polish

## 1. Summary
Cycle 06 is the final quality assurance phase. We implement the `Validator` module, which subjects the trained potential to a battery of physical tests: Phonon dispersion stability (via Phonopy), Elastic constant calculation, and Equation of State (EOS) fitting. This cycle also focuses on "Production Readiness," including robust error handling, checkpointing, and the generation of user-friendly HTML reports. Finally, we implement the tutorial notebooks defined in the UAT plan.

## 2. System Architecture

### File Structure
Files to be created/modified are marked in **bold**.

```
mlip-pipeline/
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   ├── **validation.py**   # ValidationConfig, ValidationResult
│       ├── infrastructure/
│       │   ├── validator/
│       │   │   ├── **__init__.py**
│       │   │   ├── **phonons.py**  # Phonopy wrapper
│       │   │   ├── **elasticity.py** # Elastic constant calculator
│       │   │   ├── **eos.py**      # Birch-Murnaghan fit
│       │   │   └── **report.py**   # HTML Report Generator
│       └── orchestrator/
│           └── **checkpoint.py**   # State saving logic
└── tests/
    └── integration/
        └── **test_validation_pipeline.py**
```

## 3. Design Architecture

### Domain Models (`domain_models/`)

-   **`ValidationConfig`**:
    -   `phonon_supercell`: List[int] (Default: [2, 2, 2])
    -   `elastic_strain_mag`: float (Default: 0.01)
    -   `max_rmse_energy`: float (Default: 0.005 eV/atom)
    -   `max_rmse_force`: float (Default: 0.1 eV/Å)

-   **`ValidationResult`**:
    -   `passed`: bool
    -   `metrics`: Dict[str, float] (RMSEs, Bulk Modulus)
    -   `phonon_stable`: bool
    -   `report_path`: Path

### Infrastructure (`infrastructure/`)

-   **`Validator` (BaseValidator)**:
    -   `validate(potential: Path, test_dataset: Dataset) -> ValidationResult`:
        -   Runs `pace_test` (or internal logic) for RMSE.
        -   Calls `PhononCalculator`.
        -   Calls `ElasticityCalculator`.
        -   Generates `validation_report.html`.

-   **`PhononCalculator`**:
    -   `check_stability(structure: Structure, potential: Path) -> bool`:
        -   Displaces atoms -> Calc Forces -> Get Force Constants.
        -   Checks for imaginary frequencies at Gamma point (or full path if `phonopy` installed).

## 4. Implementation Approach

1.  **Phonopy Integration**: Make `phonopy` an optional dependency. If missing, skip phonon tests but warn the user.
2.  **Elasticity**: Implement simple finite-difference method to compute $C_{11}, C_{12}, C_{44}$ for cubic systems.
3.  **Reporting**: Use `jinja2` or simple string formatting to create an HTML report with embedded plots (base64 encoded PNGs from `matplotlib`).
4.  **Checkpointing**: Add `save_state()` and `load_state()` to `Orchestrator` to allow resuming interrupted runs.

## 5. Test Strategy

### Unit Testing (`tests/unit/`)
-   **`test_elasticity.py`**:
    -   Test the Voigt-Reuss-Hill averaging logic.
    -   Test `get_elastic_tensor` with a known Lennard-Jones potential (analytical solution).
-   **`test_report.py`**:
    -   Verify HTML generation (presence of key tags).

### Integration Testing (`tests/integration/`)
-   **`test_validation_pipeline.py`**:
    -   Create a dummy potential (Mock).
    -   Run `validate`.
    -   Assert `report.html` is created.
    -   Assert `ValidationResult.passed` matches expected (Mock should pass).
