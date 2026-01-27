# Cycle 04: User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 01: Resume Capability
*   **ID**: UAT-CY04-01
*   **Priority**: High
*   **Description**: Verify that the system can resume from a crash or interruption without losing progress.
*   **Steps**:
    1.  Start a workflow that is designed to halt (mocked MD).
    2.  Interrupt the process (Ctrl+C) during the "Calculation" phase.
    3.  Restart the workflow with the same config.
*   **Expected Result**:
    *   The system logs "Resuming from cycle X, phase Calculation".
    *   The system does NOT re-run the previous "Exploration" phase.

### Scenario 02: Periodic Embedding correctness
*   **ID**: UAT-CY04-02
*   **Priority**: High
*   **Description**: Verify that the structures extracted from MD are suitable for DFT.
*   **Pre-conditions**: A large MD dump file.
*   **Steps**:
    1.  Run the selection tool: `mlip-auto extract --dump dump.lammps --output extracted.xyz`.
    2.  Visualize `extracted.xyz`.
*   **Expected Result**:
    *   The structures are small (e.g., 50-100 atoms) compared to the dump (thousands).
    *   The structures are periodic (Cell is defined).
    *   The local environment of the central atom is preserved.

### Scenario 03: Full Closed Loop (Miniature)
*   **ID**: UAT-CY04-03
*   **Priority**: Critical
*   **Description**: Verify the connection of all components.
*   **Pre-conditions**: Mocked DFT (returns Energy = Potential Energy + random noise). Mocked Pacemaker (returns a valid potential file).
*   **Steps**:
    1.  Config: `max_cycles: 2`.
    2.  Run `mlip-auto run --config loop_config.yaml`.
*   **Expected Result**:
    1.  Cycle 1 starts.
    2.  MD runs -> Halts.
    3.  Structure extracted.
    4.  Oracle "calculates".
    5.  Trainer "trains".
    6.  Cycle 2 starts.
    7.  MD runs -> Completes (Mock behavior).
    8.  Workflow finishes successfully.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Active Learning Loop

  Scenario: Handling MD Halt
    GIVEN the workflow is in Exploration phase
    WHEN the MD simulation halts due to uncertainty
    THEN the system should identify the frame causing the halt
    AND extract a local cluster around the high-uncertainty atom
    AND transition the state to 'Selection'

  Scenario: Iterative Improvement
    GIVEN a workflow running for multiple cycles
    WHEN a cycle completes
    THEN the 'generation' counter of the potential should increment
    AND the next MD simulation should use the new potential
    AND the database size should reflect the new added structures
```
