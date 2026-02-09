# Cycle 07 UAT: Advanced Dynamics (EON/kMC)

## 1. Test Scenarios

### Scenario 7.1: EON Input Configuration
*   **Goal:** Verify that the system generates valid EON inputs.
*   **Steps:**
    1.  Provide a valid `Structure` (reactant).
    2.  Configure `EONDynamics` with `temp=300K`.
    3.  Run `EONDynamics.prepare_files()`.
    4.  Inspect `config.ini` and `reactant.con`.
*   **Expected Behavior:**
    *   `config.ini` contains `[Potential]` section pointing to the script.
    *   `reactant.con` is valid EON format (contains box vectors and coords).

### Scenario 7.2: Bridge Execution
*   **Goal:** Verify that the `pace_driver.py` script computes forces correctly.
*   **Steps:**
    1.  Run the generated driver script manually with a known structure.
    2.  Use a mock potential that returns deterministic values.
    3.  Check standard output.
*   **Expected Behavior:**
    *   Output matches the expected energy and forces.
    *   Format is correct for EON (e.g., lines of floats).

### Scenario 7.3: Active Learning Halt (kMC)
*   **Goal:** Verify that uncertainty in a transition state triggers a halt.
*   **Steps:**
    1.  Configure the driver to mock a high-gamma event.
    2.  Run `eonclient` (or simulate its call).
    3.  Verify the driver exits with code 100.
    4.  Verify the Orchestrator logs "kMC Halted due to Uncertainty".
    5.  Verify `halted.con` is created.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Adaptive kMC

  Scenario: Run kMC simulation
    Given an equilibrated reactant structure
    When I run the EON dynamics engine
    Then the system should explore rare events (saddle points)
    And the timescale should exceed MD limits (e.g., > 1 microsecond)

  Scenario: Halt on uncertain transition state
    Given a transition state search (dimer method)
    When the potential uncertainty exceeds the threshold
    Then the kMC simulation should stop
    And the transition state structure should be saved for learning

  Scenario: Resume after learning
    Given a halted kMC run
    And a refined potential
    When I restart the simulation
    Then it should continue exploring from the reactant state
    And it should successfully find the transition state with the new potential
```
