# Cycle 05 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 5.1: kMC Exploration
**Goal**: Verify that EON (kMC) can explore potential energy surfaces and find saddle points using the trained potential.
**Priority**: High (P1) - Enables long-timescale simulations.
**Steps**:
1.  Configure `config.yaml` with `dynamics.type: eon`.
2.  Start Orchestrator with a simple structure (e.g., Al vacancy).
3.  Let EON run for a few steps.
4.  Check `eon_client.log`.
**Success Criteria**:
*   The system initialises EON correctly.
*   The `pace_driver.py` is called successfully by EON.
*   EON reports finding saddle points and updating the state.
*   If uncertainty is high, the driver exits with code 100, and the Orchestrator catches it.

### Scenario 5.2: Validation Gatekeeper
**Goal**: Verify that the validator correctly identifies unstable potentials.
**Priority**: Critical (P0) - Quality Assurance.
**Steps**:
1.  **Case A (Unstable)**: Provide a potential trained on only 1 structure (randomly chosen).
    *   Run `Validator.validate()`.
    *   Expect: `passed=False`, reason="Imaginary Phonons" or "Born Criteria Violation".
2.  **Case B (Stable)**: Provide a well-trained potential (or a mocked stable one).
    *   Run `Validator.validate()`.
    *   Expect: `passed=True`.
**Success Criteria**:
*   The validator correctly distinguishes between "garbage" and "good" potentials.
*   The HTML report (`validation_report.html`) is generated with plots.

## 2. Behaviour Definitions (Gherkin)

```gherkin
Feature: kMC & Validation

  Scenario: Run kMC with Uncertainty Check
    Given a configured EON wrapper
    And a potential driver script
    When EON calls the driver script
    Then it should compute energy/forces
    And check the extrapolation grade (gamma)
    And exit with code 100 if gamma > threshold

  Scenario: Validate Potential Stability
    Given a trained potential
    When I run the validator
    Then it should calculate elastic constants (C11, C12, C44)
    And calculate phonon dispersion
    And return PASSED only if all stability criteria are met
```
