# Cycle 05 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario ID: UAT-05-01 (Gatekeeper Function)
**Priority**: High
**Description**: Verify that an unstable potential is blocked from deployment.
**Steps**:
1.  Artificially create a potential file (or use a bad checkpoint).
2.  Run `mlip-auto validate --potential bad.yace`.
3.  Expect the output to say "Validation FAILED".
4.  Expect the report to highlight "Imaginary Frequencies Detected".

### Scenario ID: UAT-05-02 (Validation Report)
**Priority**: Medium
**Description**: specific check for the visual report.
**Steps**:
1.  Run validation on a decent potential.
2.  Open the generated `validation_report.html`.
3.  Check for the presence of:
    -   Phonon dispersion curve.
    -   Energy-Volume curve.
    -   Parity plots.
4.  Verify the layout is readable.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Quality Assurance

  Scenario: Validate a new potential
    GIVEN a newly trained potential
    WHEN I run the validation suite
    THEN it should calculate phonon dispersion
    AND it should calculate elastic constants
    AND if the structure is dynamically unstable
    THEN the validation result should be FAIL
    AND a report should be generated
```
