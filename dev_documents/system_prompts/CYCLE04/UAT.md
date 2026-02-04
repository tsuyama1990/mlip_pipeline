# CYCLE 04 UAT: Structure Generator

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-04-01** | High | Defect Injection | Verify that the explorer can create vacancies and interstitials in a bulk crystal. |
| **UAT-04-02** | Medium | Random Displacement (Rattling) | Verify that the explorer can generate "rattled" structures to sample harmonic vibrations. |
| **UAT-04-03** | Low | Policy Switching | Verify that the explorer changes its sampling strategy based on the cycle number or configuration. |

## 2. Behavior Definitions

### Scenario: Vacancy Generation
**GIVEN** a perfect 2x2x2 Aluminum supercell (32 atoms)
**WHEN** the Explorer applies the "Vacancy Strategy"
**THEN** the returned structure should have 31 atoms
**AND** the cell dimensions should remain unchanged.

### Scenario: Rattling Strategy
**GIVEN** a structure at equilibrium positions
**WHEN** the Explorer applies "RandomDisplacement" with sigma=0.1
**THEN** the positions of atoms should differ from the original
**AND** the maximum displacement should be consistent with the sigma value.

### Scenario: Adaptive Policy
**GIVEN** an `AdaptiveExplorer` configured to use "Rattle" in Cycle 1 and "Defect" in Cycle 2
**WHEN** `generate_candidates()` is called with `cycle=2`
**THEN** it should return structures with defects, not just rattled ones.
