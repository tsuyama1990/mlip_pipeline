# Cycle 06 User Acceptance Test (UAT)

## 1. Test Scenarios

### Scenario 1: Full Mock Active Learning Loop
**Priority**: P0 (Critical)
**Description**: Verify that the Orchestrator successfully runs a full active learning loop with mocked components.
**Steps**:
1.  Configure `config.yaml` with:
    -   `generator.strategy: random`
    -   `oracle.type: mock`
    -   `trainer.type: mock`
    -   `dynamics.type: mock_halt`
    -   `orchestrator.max_iterations: 3`
2.  Run `mlip-runner run config.yaml`.
3.  Inspect the output.
    -   Expected: "Iteration 1: Dynamics Halted."
    -   Expected: "Generating local candidates..."
    -   Expected: "Labeling 5 structures..."
    -   Expected: "Training new potential..."
    -   Expected: "Iteration 2: Dynamics Halted..."
    -   Expected: "Iteration 3: Dynamics Halted..."
4.  Verify directories `active_learning/iter_001`, `iter_002`, `iter_003` exist.

### Scenario 2: Candidate Generation on Halt
**Priority**: P1 (High)
**Description**: Verify that when a simulation halts, local candidates are generated around the halt structure.
**Steps**:
1.  Run the loop (as above).
2.  Inspect `iter_001/candidates/`.
3.  Verify that it contains multiple structure files (e.g., `candidate_001.xyz` to `candidate_020.xyz`).
4.  Verify that these structures are slightly different (perturbed) versions of the halt structure.

### Scenario 3: State Recovery (Resume)
**Priority**: P2 (Medium)
**Description**: Verify that if the process is killed, it can resume from the last saved state.
**Steps**:
1.  Run the loop and kill it (Ctrl+C) during Iteration 2.
2.  Inspect `workflow_state.json`. It should show `current_iteration: 2`.
3.  Run the loop again.
4.  Verify that it starts from Iteration 2 (skipping Iteration 1).

## 2. Behavior Definitions (Gherkin)

### Feature: Active Learning Loop

**Scenario**: Complete Cycle Execution
    **Given** a configured Orchestrator
    **When** the loop runs for 3 iterations
    **Then** 3 generations of potential files should be created
    **And** the dataset size should increase in each iteration

**Scenario**: Handling High Uncertainty
    **Given** a Dynamics simulation that halts due to high gamma
    **When** the Orchestrator processes the halt
    **Then** a swarm of local candidates should be generated
    **And** the Oracle should be invoked to label the most informative ones
    **And** the Trainer should update the potential to "fix" the hole

**Scenario**: Convergence Detection
    **Given** a Dynamics simulation that runs to completion (no halt)
    **When** the Orchestrator receives the success signal
    **Then** the loop should terminate early
    **And** a success message should be logged
