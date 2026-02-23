# Cycle 06 UAT: Scientific Validation and Tutorial Execution

## 1. Test Scenarios

### Scenario 01: Physics Validation (EOS & Phonons)
-   **Priority**: High
-   **Description**: Verify that the generated potential passes basic physical checks (Equation of State and Phonon stability).
-   **Execution**:
    1.  Provide a stable structure (e.g., ground state bulk) and a `potential.yace`.
    2.  Run the validation command.
    3.  Check output for "EOS Fit Success: Bulk Modulus = X GPa".
    4.  Verify that X > 0.
    5.  Check output for "Phonon Check Success" (or "Mock Phonon Check Passed").

### Scenario 02: End-to-End Tutorial Execution (The Master UAT)
-   **Priority**: Critical
-   **Description**: Execute the complete `tutorials/UAT_AND_TUTORIAL.py` script to simulate the entire user journey.
-   **Execution**:
    1.  Ensure a clean environment.
    2.  Run `marimo run tutorials/UAT_AND_TUTORIAL.py` (or export to script and run).
    3.  Verify that it completes without error.
    4.  Verify that the final potential `final_potential.yace` is created in the output directory.
    5.  Verify that a `report.html` is generated.

## 2. Behavior Definitions (Gherkin)

### Feature: Scientific Validation and Tutorial

**Scenario: Validate potential physics**
  GIVEN a trained `final_potential.yace`
  AND a known stable crystal structure
  WHEN the Validator executes `check_eos` and `check_phonons`
  THEN it should calculate a positive Bulk Modulus
  AND it should not detect unstable phonon modes (imaginary frequencies)
  AND it should generate a validation report

**Scenario: Execute Master Tutorial Script**
  GIVEN the `tutorials/UAT_AND_TUTORIAL.py` file
  AND the `pyacemaker` package installed
  WHEN I execute the tutorial script
  THEN it should initialize the Orchestrator
  AND it should run the full 7-step pipeline (in Mock or Real mode)
  AND it should visualize the training progress
  AND it should successfully complete and exit with code 0
