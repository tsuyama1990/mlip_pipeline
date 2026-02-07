# Cycle 04 UAT: Molecular Dynamics & On-the-Fly Learning

## 1. Test Scenarios

### Scenario 04: "Running with Safety Wheels"
**Priority**: High
**Description**: Verify that the MD engine runs a simulation using the Hybrid Potential (ACE + ZBL) and correctly halts when uncertainty spikes. This ensures the "Active Learning" trigger works.

**Pre-conditions**:
-   `lammps` executable with USER-PACE package.
-   A valid `potential.yace` file.
-   An initial structure (e.g., Al supercell).

**Steps**:
1.  User creates a `config.yaml` with `dynamics.type: lammps` and `otf_threshold: 2.0` (low threshold to force halt).
2.  User runs `pyacemaker explore --potential potential.yace --structure structure.xyz --config config.yaml` (New CLI command).

**Expected Outcome**:
-   Simulation starts (LAMMPS header printed).
-   Simulation stops *before* reaching `steps` limit.
-   Console reports: "Simulation Halted due to Uncertainty (Gamma=2.5 > 2.0)".
-   `dump.lammps` contains the trajectory up to the halt point.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: MD Exploration with Uncertainty Watchdog

  Scenario: Halt on High Uncertainty
    Given a valid potential "potential.yace"
    And a Dynamics configuration with "otf_threshold: 0.1"
    When I run the exploration
    Then the simulation should start
    And the simulation should terminate with reason "UncertaintyHalt"
    And the final structure should have "max_gamma" greater than 0.1
```
