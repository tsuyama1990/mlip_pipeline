# Cycle 06 Specification: Optimization, UAT & Final Polish

## 1. Summary
Cycle 06 is the final quality assurance and refinement phase. Its primary goal is to ensure that the generated potential is not only numerically stable but also scientifically accurate. We will implement the **Validator** module to perform rigorous physical checks, such as phonon dispersion stability and Equation of State (EOS) curves.

Additionally, we will consolidate the entire workflow into a comprehensive User Acceptance Test (UAT) and Tutorial script (`tutorials/UAT_AND_TUTORIAL.py`). This script will serve as the primary entry point for new users and the final verification step for the system. It must run seamlessly in both "Mock Mode" (for rapid CI feedback) and "Real Mode" (for actual scientific production).

## 2. System Architecture

The following file structure will be created or modified. **Bold** files are new or significantly modified.

```text
src/pyacemaker/
├── validator/
│   ├── **__init__.py**
│   ├── **base.py**           # BaseValidator Interface
│   ├── **physics.py**        # Phonon & EOS Checks
│   └── **report.py**         # HTML/Markdown Report Generator
└── **tutorials/UAT_AND_TUTORIAL.py** # The Master Script
```

## 3. Design Architecture

### 3.1. Validator Module (`validator/physics.py`)
-   **`PhysicsValidator`**: Implements `BaseValidator`.
    -   **`check_eos(structure: StructureData, potential: Path) -> ValidationResult`**:
        -   Calculates the Energy vs Volume curve.
        -   Fits the Birch-Murnaghan equation of state.
        -   **Criterion**: The bulk modulus should be positive and within physical range (e.g., 10-500 GPa).
    -   **`check_phonons(structure: StructureData, potential: Path) -> ValidationResult`**:
        -   Uses Phonopy (if available) to calculate phonon dispersion.
        -   **Criterion**: No significant imaginary frequencies (unstable modes) for a known stable ground state.
    -   **`check_elastic(structure: StructureData, potential: Path) -> ValidationResult`**:
        -   Calculates elastic constants (C11, C12, C44).
        -   **Criterion**: Born stability criteria for cubic systems.

### 3.2. Validation Report (`validator/report.py`)
-   **`ReportGenerator`**:
    -   Compiles metrics from all steps (Active Learning, Training Loss, Validation Results).
    -   Generates a simple `report.html` or `SUMMARY.md` for the user.

### 3.3. Master Tutorial Script (`tutorials/UAT_AND_TUTORIAL.py`)
-   A single, self-contained `marimo` notebook (or Python script) that:
    1.  Sets up the environment.
    2.  Defines the `config.yaml` for the SN2 Reaction scenario.
    3.  Runs the full `Orchestrator` pipeline.
    4.  Visualizes the results (Energy barrier, parity plots).
    5.  Demonstrates how to use the final potential in a simple simulation.

## 4. Implementation Approach

1.  **Implement Validator**: Create `PhysicsValidator` in `validator/physics.py`.
2.  **Implement Reporting**: Create `ReportGenerator` in `validator/report.py`.
3.  **Create UAT Script**: Develop `tutorials/UAT_AND_TUTORIAL.py` using `marimo`.
    -   Ensure it detects `CI` environment variable to switch to Mock Mode automatically.
4.  **Final Polish**:
    -   Run `ruff` and `mypy` on the entire codebase.
    -   Add docstrings to all public methods.
    -   Verify `README.md` instructions.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Validator**: Feed a known bad potential (e.g., one that predicts negative bulk modulus) to `check_eos` and verify it fails validation.
-   **Reporting**: Verify `report.html` is generated and contains the expected sections.

### 5.2. Integration Testing (The Final UAT)
-   **Execution**: Run `tutorials/UAT_AND_TUTORIAL.py`.
-   **Verification**:
    -   The script should complete without error.
    -   The final potential should be produced.
    -   The validation report should be generated.
    -   The SN2 reaction barrier should be reasonably close to the reference value (or at least physically plausible in Real Mode).
