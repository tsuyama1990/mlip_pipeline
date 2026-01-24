# Cycle 07 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 07-01: kMC Config Generation
- **Priority**: Medium
- **Description**: Verify `config.ini` for EON is correct.
- **Steps**:
  1. Configure `EONWrapper`.
  2. Generate inputs.
- **Expected Result**: `job = process_search` and `potential = script` are set.

### Scenario 07-02: EON Driver Interface
- **Priority**: Critical
- **Description**: Verify the driver script correctly computes forces and uncertainty.
- **Steps**:
  1. Run `python pace_driver.py < input_coords`.
- **Expected Result**: Output contains Energy and Forces. Exit code is 0 (if uncertainty is low).

### Scenario 07-03: kMC Halt Detection
- **Priority**: Critical
- **Description**: Verify the system detects high uncertainty during saddle search.
- **Steps**:
  1. Run `pace_driver.py` with a high-uncertainty structure.
- **Expected Result**: Exit code is 100 (or designated signal).

## 2. Behavior Definitions

```gherkin
Feature: kMC Exploration

  Scenario: Run EON Process Search
    GIVEN a system configuration
    WHEN the EON client runs
    THEN it should explore saddle points using the ACE potential

  Scenario: Interrupt kMC on Uncertainty
    GIVEN a saddle point search entering unknown territory
    WHEN the extrapolation grade exceeds the limit
    THEN the driver should abort the client
    AND the Orchestrator should capture the failing structure
```
