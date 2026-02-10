# Cycle 08 User Acceptance Test (UAT)

## 1. Test Scenarios

### **UAT-08: Physics Validation & Reporting**
*   **Goal**: Ensure that the system automatically validates the trained potential against physical criteria (phonons, elasticity) and generates a human-readable report.
*   **Priority**: Medium (Quality Gate)
*   **Success Criteria**:
    *   The Validator correctly calculates elastic constants ($C_{11}, C_{12}, C_{44}$) for a test structure.
    *   The Validator flags unstable potentials (imaginary phonons or $C_{ij} < 0$).
    *   A `validation_report.html` is generated with summary plots.

## 2. Behavior Definitions (Gherkin)

### Scenario: Stable Potential Validation
**GIVEN** a trained potential (or proxy like EMT) that is physically stable
**WHEN** the `StandardValidator` runs on the equilibrium structure
**THEN** the Elastic Constants should be positive
**AND** the Phonon Frequencies should be real (no imaginary modes)
**AND** the final report should say "PASS"

### Scenario: Unstable Potential Detection
**GIVEN** a potential that predicts negative elastic moduli (mocked or bad fit)
**WHEN** the `StandardValidator` runs
**THEN** the report should say "FAIL"
**AND** the reason should be clearly stated (e.g., "Born Stability Criteria Violated")

### Scenario: Report Generation
**GIVEN** validation results (pass or fail)
**WHEN** the `ReportGenerator` runs
**THEN** an HTML file `validation_report.html` should be created
**AND** it should contain a table of calculated properties vs reference values (if provided)
**AND** it should contain an EOS plot (Energy vs Volume)
