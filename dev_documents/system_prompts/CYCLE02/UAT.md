# Cycle 02: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 1: Basic DFT Calculation
**Priority**: High
**Goal**: Verify that the Oracle can calculate energy and forces for a simple structure (e.g., Silicon).
**Procedure**:
1.  Define a simple Si crystal structure in a Jupyter notebook.
2.  Initialize `DFTPhase` with a mock QE runner (or real if available).
3.  Run the calculation.
4.  Inspect the output.
**Success Criteria**:
*   Output contains `energy`, `forces` (N x 3 array), and `stress` (3 x 3 matrix).
*   Values are non-zero and physically reasonable (if using real QE).

### Scenario 2: Self-Healing on Convergence Failure
**Priority**: High
**Goal**: Verify that the system recovers from a "Convergence NOT achieved" error.
**Procedure**:
1.  Configure the Mock Runner to fail the first attempt with a convergence error.
2.  Run `DFTPhase`.
3.  Check logs.
**Success Criteria**:
*   Logs show "Convergence failed. Retrying with strategy..."
*   The second attempt (with modified parameters) succeeds.
*   The final result is returned correctly.

## 2. Behavior Definitions

```gherkin
Feature: DFT Oracle

  Scenario: Successful Calculation
    GIVEN a structure of Silicon
    WHEN the Oracle processes the structure
    THEN it should return an Energy value
    AND it should return Forces for all atoms
    AND it should return the Virial Stress tensor

  Scenario: Retry Logic
    GIVEN a structure that is hard to converge
    WHEN the Oracle fails on the first attempt
    THEN it should adjust "mixing_beta"
    AND retry the calculation automatically
    AND return the result if the second attempt succeeds
```
