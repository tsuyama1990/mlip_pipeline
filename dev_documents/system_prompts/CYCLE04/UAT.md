# Cycle 04 UAT: The First Training

## 1. Test Scenarios

### Scenario 4: Training a Potential
**Priority**: Critical
**Objective**: Verify that the system can generate a valid input for Pacemaker and successfully "train" (invoke the binary).

**Steps**:
1.  **Preparation**:
    *   Create a dataset with 10 labeled structures (random positions, fake energies).
    *   Config: `trainer.type = "pacemaker"`.
2.  **Execution**:
    *   Run the Trainer module directly or via Orchestrator.
3.  **Verification**:
    *   Check that `input.yaml` was generated in the work directory.
    *   Check that `input.yaml` contains `b_ref` (ZBL) settings.
    *   Check that `potential.yace` exists.

## 2. Behavior Definitions

```gherkin
Feature: Delta Learning Configuration

  Scenario: Enforcing ZBL baseline
    GIVEN a configuration specifying "ZBL" baseline
    WHEN the Trainer prepares the "input.yaml" for Pacemaker
    THEN the YAML must contain a "b_ref" section
    AND the "type" must be "zbl"
    AND the "inner_cutoff" must match the configuration
```
