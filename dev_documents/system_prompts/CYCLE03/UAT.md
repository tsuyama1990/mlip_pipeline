# Cycle 03 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 03-01: Hybrid Potential Generation
- **Priority**: Critical
- **Description**: Verify that the generated LAMMPS input correctly defines the hybrid potential.
- **Steps**:
    1. Configure the system for Silicon.
    2. Generate an MD task.
    3. Inspect the `in.lammps` file.
    4. **Expected Result**: The file contains `pair_style hybrid/overlay` and `pair_coeff` lines for both `pace` and `zbl`.

### Scenario 03-02: Uncertainty Halt Detection
- **Priority**: Critical
- **Description**: Verify that the system stops when the potential enters an extrapolation region.
- **Steps**:
    1. Use a "poor" potential (e.g., trained on only 1 structure).
    2. Start a high-temperature MD run (e.g., 3000K).
    3. **Expected Result**: The simulation stops before the requested number of steps. The system logs report "Halted due to high uncertainty (gamma > 5.0)".

### Scenario 03-03: Safe Crash Recovery
- **Priority**: High
- **Description**: Verify that if LAMMPS crashes (e.g., segmentation fault), the system handles it gracefully.
- **Steps**:
    1. Force a crash (e.g., by providing an invalid potential path).
    2. Run the MD engine.
    3. **Expected Result**: The Python process captures the error, logs it as a "Simulation Failure", and does not crash the entire workflow.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Dynamics and Uncertainty

  Scenario: Running MD with Hybrid Potential
    GIVEN a trained potential "current.yace"
    WHEN I launch an NVT simulation
    THEN the LAMMPS input should overlay ZBL potential
    AND the simulation should run without immediate "atoms too close" errors

  Scenario: Triggering Uncertainty Watchdog
    GIVEN a simulation with a gamma threshold of 5.0
    WHEN the maximum gamma value in the system reaches 5.1
    THEN the LAMMPS process should terminate
    AND the wrapper should report "Halted" status
    AND the structure at that frame should be saved
```
