# CYCLE 06 UAT: Scale-up, Validation & Integration

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-06-01** | High | Elastic Constant Validation | Verify that the system correctly calculates elastic constants ($C_{11}, C_{12}, C_{44}$) for a cubic crystal. |
| **UAT-06-02** | High | EOS Validation | Verify that the system generates a valid Equation of State curve and Bulk Modulus. |
| **UAT-06-03** | Medium | EON Driver Execution | Verify that the external EON driver script works correctly with the generated potential. |
| **UAT-06-04** | Critical | Full Fe/Pt Scenario (Mock) | Verify that the entire pipeline runs end-to-end for the target use case. |

## 2. Behavior Definitions

### Scenario: Elastic Validation
**GIVEN** a potential that perfectly reproduces a material with $C_{11}=200$ GPa
**WHEN** `PhysicsValidator` runs the elasticity test
**THEN** the calculated $C_{11}$ should be within 5% of 200 GPa
**AND** the result should be marked as PASS.

### Scenario: Full System Integration (The Grand Finale)
**GIVEN** the "Fe/Pt on MgO" configuration
**AND** execution mode is "Mock"
**WHEN** the user runs `mlip-pipeline run config.yaml`
**THEN** the system should:
1.  Initialize.
2.  Run Mock MD (Cycle 5).
3.  Detect Uncertainty (Cycle 5).
4.  Embed Cluster (Cycle 3).
5.  Run Mock DFT (Cycle 3).
6.  Train Potential (Cycle 2).
7.  Validate Potential (Cycle 6).
8.  Exit with Success.
