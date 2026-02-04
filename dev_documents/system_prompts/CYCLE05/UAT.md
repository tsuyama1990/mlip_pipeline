# CYCLE 05 UAT: Dynamics Engine - MD & OTF

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-05-01** | High | Hybrid Potential Generation | Verify that the MD engine enforces the use of Hybrid/Overlay potentials (ACE + ZBL). |
| **UAT-05-02** | High | Uncertainty Watchdog Trigger | Verify that the simulation stops when extrapolation grade exceeds the threshold. |
| **UAT-05-03** | Medium | High-Gamma Extraction | Verify that the system extracts the specific atomic cluster responsible for the high uncertainty. |

## 2. Behavior Definitions

### Scenario: Hybrid Potential Safety
**GIVEN** an MD request for a system with Fe atoms
**WHEN** the `LammpsMD` engine generates the input script
**THEN** it should NOT use `pair_style pace` alone
**BUT** `pair_style hybrid/overlay pace zbl`
**AND** define ZBL coefficients for Fe-Fe.

### Scenario: Watchdog Halt
**GIVEN** a mocked LAMMPS execution that returns a "Halt" exit code
**WHEN** `run_simulation()` is called
**THEN** it should return a result status `HALTED`
**AND** provide the path to the dump file containing the critical snapshot.

### Scenario: Structure Extraction
**GIVEN** a dump file where atom #42 has $\gamma = 10.0$ (high)
**WHEN** `OTFHandler.process_halt()` is called
**THEN** it should return a `StructureMetadata` object
**AND** the structure should be centered near atom #42.
