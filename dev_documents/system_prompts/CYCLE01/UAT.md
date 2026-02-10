# Cycle 01 User Acceptance Test (UAT)

## 1. Test Scenarios

### **UAT-01: Basic Pipeline Execution (Mock Mode)**
*   **Goal**: Verify the system can run a complete active learning loop using Mock components, ensuring the orchestration logic, state persistence, and logging infrastructure are functioning correctly.
*   **Priority**: High (Critical Path)
*   **Success Criteria**:
    *   The user can initialize a project with `mlip-auto init`.
    *   The user can run the loop with `mlip-auto run-loop`.
    *   The process completes successfully (exit code 0).
    *   A `workflow_state.json` file is created and updated.
    *   Logs show clearly: "Generator finished", "Oracle finished", "Trainer finished".

## 2. Behavior Definitions (Gherkin)

### Scenario: First-Time Initialization
**GIVEN** a clean directory
**WHEN** the user runs `mlip-auto init`
**THEN** a `config.yaml` file is created with default settings
**AND** the file contains valid YAML structure with `mock` components selected

### Scenario: Mock Execution Loop
**GIVEN** a valid `config.yaml` configured with:
    *   `orchestrator.max_iterations: 2`
    *   `generator.type: mock`
    *   `oracle.type: mock`
    *   `trainer.type: mock`
**WHEN** the user runs `mlip-auto run-loop`
**THEN** the application should run for exactly 2 iterations
**AND** the log output should contain:
    *   "Cycle 1/2: Exploring..."
    *   "Cycle 1/2: Labeling (Mock Oracle)..."
    *   "Cycle 1/2: Training (Mock Trainer)..."
    *   "Cycle 2/2: Exploring..."
**AND** a `workflow_state.json` file should exist in the work directory
**AND** the final state in `workflow_state.json` should reflect `iteration: 2`
