# Cycle 02 UAT: The Intelligent Explorer

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-02-01** | High | **Policy Driven Generation** | Verify that the generator produces different outputs based on the input policy (e.g., Strain vs. Rattle). |
| **UAT-02-02** | Medium | **Defect Injection** | Verify that the system can successfully create vacancy defects without breaking the simulation cell. |

## 2. Behavior Definitions

### UAT-02-01: Policy Driven Generation

**GIVEN** a perfect MgO crystal structure
**WHEN** the `StructureGenerator` is invoked with `StrainHeavyPolicy` (range +/- 10%)
**THEN** the output structures should have varying cell volumes
**AND** the atomic fractional coordinates should remain constant (affine deformation)
**BUT** **WHEN** the `StructureGenerator` is invoked with `RattlePolicy` (sigma=0.1A)
**THEN** the cell vectors should remain constant
**AND** the atomic positions should deviate from the perfect lattice sites

### UAT-02-02: Defect Injection

**GIVEN** a 2x2x2 Supercell of Al (32 atoms)
**WHEN** the `StructureGenerator` is asked to create 1 vacancy
**THEN** the returned structure should contain exactly 31 atoms
**AND** the cell dimensions should remain unchanged
**AND** the structure should be tagged with `defect_type="vacancy"` in its metadata
