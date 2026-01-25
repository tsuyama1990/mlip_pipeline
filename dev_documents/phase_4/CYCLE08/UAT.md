# Cycle 08 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 08-01: Phonon Stability Check
- **Priority**: High
- **Description**: Ensure the potential predicts stable phonons for a crystal.
- **Steps**:
  1. Train a potential for Al.
  2. Run `mlip-auto validate --phonon`.
- **Expected Result**: No imaginary modes (negative frequencies) in the dispersion curve.

### Scenario 08-02: End-to-End "Zero Config" Run
- **Priority**: Critical
- **Description**: The "Holy Grail" test.
- **Steps**:
  1. `mlip-auto run config_simple.yaml`
  2. Wait.
- **Expected Result**: System runs Exploration -> DFT -> Training loop without crashing.

### Scenario 08-03: CLI Usability
- **Priority**: Medium
- **Description**: Help messages and progress bars.
- **Steps**:
  1. `mlip-auto --help`
- **Expected Result**: Clear description of commands and arguments.

## 2. Behavior Definitions

```gherkin
Feature: System Validation

  Scenario: Validate Physical Properties
    GIVEN a trained potential
    WHEN the validator runs
    THEN it should compute Elastic Constants and Phonons
    AND report if the structure is dynamically stable

  Scenario: Production Execution
    GIVEN a complete configuration
    WHEN the user launches the workflow
    THEN the system should autonomously manage the active learning loop
```
