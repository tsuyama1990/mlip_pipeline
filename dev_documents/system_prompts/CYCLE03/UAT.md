# Cycle 03 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario ID: UAT-03-01 (Safety Mechanism)
**Priority**: Critical
**Description**: Verify that the "Hybrid Potential" logic prevents atoms from collapsing (nuclear fusion) when the ML potential is undefined.
**Steps**:
1.  Configure a run with `ZBL` baseline.
2.  Start MD with two atoms moving towards each other at high velocity.
3.  Simulate.
4.  Check the minimum distance in the trajectory.
5.  Expect it to never drop below ~0.5 Å (due to ZBL repulsion), even if the ML potential predicts attraction.

### Scenario ID: UAT-03-02 (Uncertainty Trigger)
**Priority**: High
**Description**: Verify that the simulation stops exactly when uncertainty is high.
**Steps**:
1.  Train a weak potential on minimal data (Cycle 02).
2.  Run MD on a high-temperature system using this potential.
3.  Set `uncertainty_threshold` to a strict value.
4.  Run.
5.  Expect the run to terminate with a "Halted" status.
6.  Inspect the output dump. The final frame should have high gamma values.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Dynamics Engine Safety

  Scenario: Halt on high uncertainty
    GIVEN a running MD simulation
    AND an uncertainty threshold of 5.0
    WHEN the maximum extrapolation grade of any atom exceeds 5.0
    THEN the simulation should terminate immediately
    AND the exit status should indicate "Halted"
    AND the last trajectory frame should be preserved for analysis
```
