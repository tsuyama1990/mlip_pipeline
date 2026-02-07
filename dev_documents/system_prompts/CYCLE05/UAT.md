# Cycle 05 UAT: Advanced Exploration & kMC

## 1. Test Scenarios

### Scenario 05: "Intelligent Exploration"
**Priority**: Medium
**Description**: Verify that the system can intelligently switch exploration strategies (e.g., adding defects) and perform a kMC-like saddle point search. This tests `AdaptivePolicy` and `EONWrapper`.

**Pre-conditions**:
-   `eonclient` (or mock) installed.
-   Valid `potential.yace`.

**Steps**:
1.  User creates a `config.yaml` with `generator.policy_mode: adaptive`.
2.  User runs `pyacemaker explore --potential potential.yace` (Mock inputs: stiff material).
3.  User checks logs to see "Selected Strategy: STRAIN_SCAN".
4.  User runs `pyacemaker kmc --potential potential.yace` (New CLI command).

**Expected Outcome**:
-   Logs show strategy selection logic.
-   kMC run creates `processes/` directory.
-   `client.log` indicates successful saddle searches.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Adaptive Exploration and kMC

  Scenario: Select Defect Strategy for Alloys
    Given a configuration with "policy_mode: adaptive"
    And a material identified as "alloy"
    When the Policy Engine decides the next step
    Then the strategy should be "DEFECT_SAMPLING" or "HIGH_MC"

  Scenario: Run kMC Saddle Search
    Given a valid potential
    When I request a kMC run
    Then the EON client should be executed
    And a saddle point should be identified
```
