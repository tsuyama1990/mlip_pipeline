# Cycle 02 UAT: The Oracle and the Recovery

## 1. Test Scenarios

### Scenario 2: DFT Calculation with Self-Healing
**Priority**: High
**Objective**: Verify that the Oracle can handle calculation failures and retry with different parameters.

**Prerequisites**:
*   This UAT technically requires Quantum Espresso (`pw.x`) to be installed for a "Real" run.
*   For the "Mock/CI" run, we will use the "Simulated QE" approach described in the Spec.

**Steps**:
1.  **Preparation**:
    *   Configure `oracle.type = "espresso"`.
    *   (Mock Mode): Inject a "Faulty Calculator" that fails the first time.
2.  **Execution**:
    *   Run the pipeline with a single structure.
3.  **Verification**:
    *   Check logs: "Calculation failed with default params. Retrying with Recovery Recipe #1...".
    *   Check logs: "Calculation succeeded."
    *   Verify the output structure has Energy and Forces attached.

## 2. Behavior Definitions

```gherkin
Feature: Oracle Robustness

  Scenario: Recovering from SCF convergence failure
    GIVEN an atomic structure that is difficult to converge
    AND an "EspressoOracle" configured with recovery strategies
    WHEN the Oracle attempts to calculate the energy
    AND the first calculation attempt fails with "ConvergenceError"
    THEN the Oracle should NOT raise an exception immediately
    BUT it should apply the "Conservative Mixing" strategy (beta=0.3)
    AND retry the calculation
    AND if the second attempt succeeds, it should return the labeled structure
```
