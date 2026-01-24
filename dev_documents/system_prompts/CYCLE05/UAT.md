# Cycle 05 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 05-01: Hybrid Potential Setup
- **Priority**: Critical
- **Description**: Verify LAMMPS input contains safe overlay.
- **Steps**:
  1. Generate inputs for a Ti-O system.
  2. Inspect `in.lammps`.
- **Expected Result**: `pair_style hybrid/overlay` exists. `pace` and `zbl` (or `lj/cut`) are defined.

### Scenario 05-02: Uncertainty Halt Detection
- **Priority**: Critical
- **Description**: System correctly identifies when MD stops due to high gamma.
- **Steps**:
  1. Use Mock LAMMPS that simulates a crash/halt at step 500.
  2. Run `LammpsRunner`.
- **Expected Result**: Return object indicates `halted=True` and `step=500`.

### Scenario 05-03: Resume Capability
- **Priority**: High
- **Description**: Ensure restart files are requested in input.
- **Steps**:
  1. Check generated input.
- **Expected Result**: `restart` command is present (e.g., every 1000 steps).

## 2. Behavior Definitions

```gherkin
Feature: MD Inference Engine

  Scenario: Generate Safe Input
    GIVEN a configuration with "use_zbl_baseline: True"
    WHEN the input writer generates the script
    THEN it should include "pair_style hybrid/overlay"
    AND "pair_coeff * * zbl"

  Scenario: Detect Watchdog Halt
    GIVEN an MD run that exceeds the uncertainty threshold
    WHEN the runner parses the output
    THEN it should raise a specific HaltSignal or return status="halted"
    AND provide the path to the trajectory dump
```
