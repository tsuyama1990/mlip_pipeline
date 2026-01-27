# Cycle 06 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario ID: UAT-06-01 (Adaptive Intelligence)
**Priority**: Medium
**Description**: Verify the system changes strategy.
**Steps**:
1.  Configure a job for an Insulator.
2.  Run the system.
3.  Check logs. Expect "Defect-Driven Policy" to be active (creating vacancies).
4.  Configure a job for a Metal.
5.  Check logs. Expect "High-MC Policy" to be active.

### Scenario ID: UAT-06-02 (Rare Event Discovery)
**Priority**: Low (Advanced)
**Description**: Verify kMC finds a transition.
**Steps**:
1.  Set up a system with a known barrier (e.g., vacancy migration).
2.  Force the policy to run kMC.
3.  Check if EON finds a saddle point.
4.  Check if the system learns from the saddle point structure (if uncertainty was high).

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Final System Capabilities

  Scenario: Intelligent Strategy Switching
    GIVEN the system is exploring a complex material
    WHEN the initial MD phase is complete
    AND the system detects low diffusion
    THEN the Adaptive Policy should select "kMC Exploration" for the next cycle
    AND the Orchestrator should execute EON instead of LAMMPS
```
