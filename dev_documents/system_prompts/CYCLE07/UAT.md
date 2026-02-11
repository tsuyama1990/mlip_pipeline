# Cycle 07 User Acceptance Testing (UAT)

## 1. Test Scenarios

### Scenario 7.1: File Format Conversion
*   **Priority**: High
*   **Goal**: Verify data integrity for EON.
*   **Action**:
    1.  Create an `ase.Atoms` object (e.g., MgO).
    2.  Invoke `utils.file_formats.ase_to_con(atoms, "reactant.con")`.
    3.  Read the file back.
*   **Expectation**:
    *   File exists.
    *   Contains correct lattice vectors and coordinates.
    *   Roundtrip gives back `ase.Atoms` with error < 1e-6.

### Scenario 7.2: Bridge Script Execution (Mock)
*   **Priority**: Critical
*   **Goal**: Verify potential evaluation interface.
*   **Action**:
    1.  Prepare a mock potential.yace.
    2.  Run `python -m mlip_autopipec.components.dynamics.eon_bridge` with input piped from a `.con` file.
*   **Expectation**:
    *   Stdout contains Energy (scalar) and Forces (vector).
    *   Exit code is 0.

### Scenario 7.3: EON Uncertainty Interrupt (Mock)
*   **Priority**: High
*   **Goal**: Verify safety mechanism.
*   **Action**:
    1.  Force the potential (mock calculator) to return high gamma.
    2.  Run the bridge script.
*   **Expectation**:
    *   Exit code is non-zero (e.g., 100).
    *   Stderr contains "Uncertainty limit exceeded".
    *   File `bad.con` is created.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: Adaptive Kinetic Monte Carlo

  Scenario: EON requests energy from Python
    Given a running EON simulation
    When it queries the potential bridge
    Then the bridge must return accurate ACE energies and forces
    And maintain communication protocol

  Scenario: EON halts on uncertainty
    Given a transition state search
    When the structure enters an unknown region (high gamma)
    Then the bridge must signal a halt
    And the Orchestrator must capture the structure for retraining
```
