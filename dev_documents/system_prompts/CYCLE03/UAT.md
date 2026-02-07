# Cycle 03 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 3.1: "Mock Binary" Integration
**Goal**: Verify that the `DFTManager` and `PacemakerWrapper` can execute external commands correctly without requiring the full HPC software stack.
**Priority**: High (P1) - Enables CI/CD testing.
**Steps**:
1.  Create `mock_pw.py` (simulates `pw.x`: writes an output file with dummy energy/forces).
2.  Create `mock_pace.py` (simulates `pace_train`: writes a `potential.yace` file).
3.  Configure `config.yaml` with:
    *   `oracle.type: qe`
    *   `oracle.command: "python mock_pw.py"`
    *   `trainer.type: pacemaker`
    *   `trainer.command: "python mock_pace.py"`
4.  Run the Orchestrator loop for 1 cycle.
**Success Criteria**:
*   The loop completes without crashing.
*   `mock_pw.py` is executed for each structure in the dataset.
*   `mock_pace.py` is executed once per cycle.
*   The system correctly parses the "dummy" output from these scripts.

### Scenario 3.2: Self-Healing Logic (DFT)
**Goal**: Verify that the Oracle retries calculations if they fail.
**Priority**: Critical (P0) - Robustness.
**Steps**:
1.  Modify `mock_pw.py` to fail (exit code 1 or print "Error in routine c_bands") on the FIRST attempt, but succeed on the SECOND attempt.
2.  Run the Orchestrator.
3.  Check the logs.
**Success Criteria**:
*   The log should show:
    *   "DFT Calculation Failed. Retrying with reduced mixing beta..."
    *   "DFT Calculation Succeeded."
*   The structure should be labeled correctly despite the initial failure.

## 2. Behaviour Definitions (Gherkin)

```gherkin
Feature: Oracle & Trainer Integration

  Scenario: Run DFT with Self-Healing
    Given a configured DFTManager with "mock_pw.py" (which fails once)
    When I call compute() on a structure
    Then it should catch the failure
    And retry with modified parameters (e.g. mixing_beta=0.3)
    And succeed on the second attempt
    And return a labeled structure

  Scenario: Train Potential with Pacemaker
    Given a configured PacemakerWrapper with "mock_pace.py"
    And a valid dataset
    When I call train()
    Then it should execute the command line tool
    And return a valid Potential object
    And the potential path should point to the generated file
```
