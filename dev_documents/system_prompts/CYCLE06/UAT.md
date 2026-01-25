# Cycle 06: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Detect Unstable Phonons
**Priority**: High
**Goal**: Ensure the system blocks a potential with imaginary modes.
**Procedure**:
1.  Create a structure that is known to be unstable (e.g., high-pressure phase at zero pressure).
2.  Run `PhononValidator` using a dummy potential.
3.  Check result.
**Success Criteria**:
*   `is_stable` is False.
*   Report indicates "Imaginary frequencies detected".

### Scenario 2: Elastic Constants Verification
**Priority**: Medium
**Goal**: Verify calculated elastic moduli are accurate.
**Procedure**:
1.  Use a standard LJ potential on FCC Argon.
2.  Run `ElasticityValidator`.
3.  Compare $C_{11}, C_{12}, C_{44}$ with literature values.
**Success Criteria**:
*   Values are within 5% of theoretical targets.
*   Born stability check passes.

## 2. Behavior Definitions

```gherkin
Feature: Validation Module

  Scenario: Phonon Stability Check
    GIVEN a candidate potential
    WHEN the Phonon Validator runs
    THEN it should calculate the full phonon band structure
    AND it should fail if any mode has imaginary frequency (negative eigenvalues)

  Scenario: Elasticity Check
    GIVEN a candidate potential
    WHEN the Elasticity Validator runs
    THEN it should calculate stiffness tensor Cij
    AND it should verify Born stability criteria
```
