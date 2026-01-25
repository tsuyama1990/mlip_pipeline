# Cycle 06 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 06-01: Full Active Learning Cycle (Simulated)
- **Priority**: Critical
- **Description**: Verify the orchestrator navigates through all phases.
- **Steps**:
  1. Configure `WorkflowManager` with Mock components.
  2. Run `manager.run()`.
  3. Observe logs.
- **Expected Result**: Log sequence: "Starting MD" -> "Halt detected" -> "Selecting candidates" -> "Running DFT" -> "Training" -> "Cycle Complete".

### Scenario 06-02: State Recovery
- **Priority**: High
- **Description**: System resumes from where it left off.
- **Steps**:
  1. Interrupt the loop during "Calculation" phase (simulate Ctrl+C).
  2. Restart the process.
- **Expected Result**: System detects existing state and resumes at "Calculation", skipping "Exploration".

## 2. Behavior Definitions

```gherkin
Feature: Workflow Orchestration

  Scenario: Handle MD Halt
    GIVEN the system is in Exploration phase
    WHEN the MD simulation halts due to high uncertainty
    THEN the system should transition to Selection phase
    AND extract candidate structures from the dump file

  Scenario: Resume from Checkpoint
    GIVEN a saved state indicating "Training" phase
    WHEN the Orchestrator starts
    THEN it should load the dataset
    AND immediately begin the Training process
```
