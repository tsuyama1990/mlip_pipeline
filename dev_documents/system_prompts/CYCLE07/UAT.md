# Cycle 07 UAT: Advanced Dynamics (kMC / EON)

## 1. Test Scenarios

### Scenario 1: EON Integration (Mocked)
*   **ID**: S07-01
*   **Goal**: Verify that EON client is launched and can communicate with Pacemaker.
*   **Priority**: Critical.
*   **Steps**:
    1.  Create `config.ini` for EON.
    2.  Mock `driver.py` hook (just return dummy energy).
    3.  Run `eonclient` (or mocked subprocess).
    4.  Assert exit code is 0 (normal termination).

### Scenario 2: On-the-Fly Detection (Halt Event)
*   **ID**: S07-02
*   **Goal**: Verify that kMC halts when a saddle point has high uncertainty.
*   **Priority**: High.
*   **Steps**:
    1.  Mock `driver.py` hook to check $\gamma$ and return high value.
    2.  Run `eonclient` (or mocked subprocess).
    3.  Assert exit code is 100 (Halt).
    4.  Assert `bad_structure.cfg` is created.

### Scenario 3: Python Hook Functionality
*   **ID**: S07-03
*   **Goal**: Verify that `driver.py` correctly calculates energy using Pacemaker.
*   **Priority**: Medium.
*   **Steps**:
    1.  Feed `driver.py` a valid structure via stdin.
    2.  Use a dummy `.yace` potential.
    3.  Assert output contains valid energy and force values.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Kinetic Monte Carlo Simulation

  Scenario: Standard EON Run
    Given an EON configuration file
    And a valid potential driver script
    When the EON client executes
    Then the driver should compute energies for multiple structures
    And the simulation should proceed normally

  Scenario: Halting on Uncertain Saddle Point
    Given the driver detects high gamma (> 5.0) for a saddle point
    When the EON client calls the driver
    Then the driver should exit with status code 100
    And write the uncertain structure to "bad_structure.cfg"
```
