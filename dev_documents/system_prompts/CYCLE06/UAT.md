# Cycle 06 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: EOS Validation
*   **ID**: UAT-06-01
*   **Priority**: High
*   **Description**: Verify that the system calculates the Bulk Modulus and rejects non-physical potentials.
*   **Success Criteria**:
    *   Input: A potential that predicts attractive forces at $r=0$.
    *   Output: `EOSValidator` returns FAIL (Bulk Modulus < 0 or curve not convex).

### Scenario 2: Elastic Stability Check
*   **ID**: UAT-06-02
*   **Priority**: High
*   **Description**: Verify that the system correctly identifies a mechanically unstable phase.
*   **Success Criteria**:
    *   Input: A structure violating Born criteria.
    *   Output: `ElasticValidator` returns FAIL with message "Born stability criteria violated".

### Scenario 3: Automated Gatekeeping
*   **ID**: UAT-06-03
*   **Priority**: Critical
*   **Description**: Verify that the Orchestrator respects the validation result.
*   **Success Criteria**:
    *   Run the full pipeline with a mock validator that always fails.
    *   The system should **not** promote the potential to `current.yace`.
    *   The system should log "Potential rejected by validation".

## 2. Behavior Definitions

```gherkin
Feature: Validation Suite

  As a researcher
  I want the system to verify physical properties automatically
  So that I don't waste time simulating with broken potentials

  Scenario: Detecting Imaginary Phonons
    GIVEN a potential that is overfitted
    WHEN I run the PhononValidator
    THEN it should detect imaginary frequencies at the X-point
    AND the validation status should be FAIL

  Scenario: Valid Potential Promotion
    GIVEN a potential that passes EOS, Elastic, and Phonon checks
    WHEN the validation suite completes
    THEN the status should be PASS
    AND the potential should be deployed for the next MD cycle
```
