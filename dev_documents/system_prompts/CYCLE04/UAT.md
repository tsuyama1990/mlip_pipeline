# Cycle 04 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 04-01: End-to-End Active Learning Cycle
- **Priority**: Critical
- **Description**: Verify that the system can autonomously improve a potential.
- **Steps**:
    1. Start with a poor potential.
    2. Configure the loop to run 2 iterations.
    3. Launch the system.
    4. **Expected Result**:
        - Iteration 1: MD halts -> DFT runs -> Potential v2 created.
        - Iteration 2: MD runs longer (or halts again) -> Potential v3 created.
        - Files `generation_001.yace` and `generation_002.yace` exist.

### Scenario 04-02: Periodic Embedding Accuracy
- **Priority**: High
- **Description**: Verify that the "cut out" structure physically matches the original environment.
- **Steps**:
    1. Take a large supercell with a vacancy.
    2. Run the embedding tool to extract the vacancy region.
    3. Visualise both structures.
    4. **Expected Result**: The extracted cell contains the vacancy and immediate neighbors, and the atomic distances exactly match the original large cell.

### Scenario 04-03: Resume Capability
- **Priority**: Medium
- **Description**: Verify that if the process is killed during DFT, it resumes from the DFT step, not the beginning.
- **Steps**:
    1. Start a cycle. Wait until it reaches "Calculating DFT".
    2. Kill the process (Ctrl+C).
    3. Restart the process.
    4. **Expected Result**: Logs show "Resuming from DFT phase", skipping the already completed MD exploration.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Active Learning Loop

  Scenario: Completing a refinement cycle
    GIVEN a running Orchestrator
    WHEN the Dynamics Engine reports a halt at step 5000
    THEN the system should extract the halted structure
    AND submit it to the Oracle
    AND retrain the potential with the new data
    AND restart the MD from step 5000

  Scenario: Handling redundant data
    GIVEN 100 candidate structures from a halted run
    WHEN the Selection phase runs
    THEN only the top N (e.g., 5 or 10) most distinct structures should be sent to DFT
    AND the rest should be discarded to save compute time
```
