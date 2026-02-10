# Cycle 07: Advanced Dynamics (aKMC & Deposition) - UAT

## 1. Test Scenarios

### Scenario 1: Deposition Script Generation
*   **ID**: UAT-07-001
*   **Objective**: Ensure LAMMPS is configured for atom deposition.
*   **Pre-conditions**: Cycle 05 complete.
*   **Steps**:
    1.  Configure `DynamicsConfig` with `deposition={"rate": 10, "species": ["Fe", "Pt"], "substrate_height": 5.0}`.
    2.  Generate input script.
    3.  Check `in.lammps`.
*   **Expected Result**:
    *   `region deposition block ...` is defined above the substrate.
    *   `fix 1 addatoms deposit 10 0 100 12345 region deposition near 1.0` is present.
    *   `group bottom region substrate_bottom` is frozen (`fix setforce 0 0 0`).

### Scenario 2: EON Driver Interface
*   **ID**: UAT-07-002
*   **Objective**: Verify that `pace_driver.py` correctly interfaces with EON.
*   **Pre-conditions**: A valid potential file.
*   **Steps**:
    1.  Create a file `pos.con` with atomic coordinates in EON format.
    2.  Run `python src/mlip_autopipec/utils/pace_driver.py < pos.con`.
*   **Expected Result**:
    *   Standard Output contains Energy (1st line).
    *   Standard Output contains Forces (subsequent lines).
    *   No python traceback errors.

### Scenario 3: Bridging MD to kMC
*   **ID**: UAT-07-003
*   **Objective**: Ensure the final state of MD can be used as the initial state for kMC.
*   **Pre-conditions**: MD run completed with `dump.lammps`.
*   **Steps**:
    1.  Extract the last frame of `dump.lammps`.
    2.  Convert it to `reactant.con` (EON format).
    3.  Initialize `EONWrapper` with this structure.
*   **Expected Result**:
    *   `reactant.con` is created in the EON directory.
    *   Atom types are correctly mapped (e.g., Type 1 -> Fe, Type 2 -> Pt).

## 2. Behavior Definitions

```gherkin
Feature: Advanced Simulation Capabilities

  Scenario: Simulating Deposition
    Given I want to simulate the growth of a nanoparticle
    When I configure the deposition rate and species
    Then the system should generate a LAMMPS script that deposits atoms periodically
    And the substrate atoms should be fixed at the bottom
    And the temperature should be controlled (NVT)

  Scenario: Simulating Long-Term Ordering (aKMC)
    Given I have a disordered cluster from MD
    And I want to find the ordered L10 structure
    When I run the Adaptive Kinetic Monte Carlo simulation
    Then the system should explore rare events (diffusion, rearrangement)
    And it should identify lower energy basins
    And the time scale should reach seconds or minutes (simulated time)
```
