# Cycle 03 UAT: Physics-Informed Generator

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-03-01** | High | **Alloy SQS Generation** | Verify that the system can generate a Special Quasirandom Structure (SQS) for a specified binary composition (e.g., Fe-Ni 70:30) that matches the target stoichiometry exactly. The structure should be periodic and space-filling. |
| **UAT-03-02** | Medium | **Distortion Pipeline** | Verify that the system can take a base structure and generate a "family" of distorted structures including volumetric strain, shear strain, and thermal rattling. The volume changes should match the inputs. |
| **UAT-03-03** | Low | **Molecule NMS** | Verify that for a molecular system (e.g., H2O), the system generates distorted geometries corresponding to bending and stretching modes at specified temperatures. |
| **UAT-03-04** | High | **Metadata Integrity** | Verify that every generated structure includes the correct metadata (config_type, parents, distortion parameters) which is crucial for the learning process. |

### Recommended Demo
Create `demo_03_generator.ipynb`.
1.  **Block 1**: Configure `GeneratorConfig` for "Fe70Ni30" with 32 atoms.
2.  **Block 2**: Run `AlloyGenerator`.
3.  **Block 3**: Visualize the resulting SQS using `ase.visualize` (or a static 2D plot of atomic positions).
4.  **Block 4**: Print the chemical formula (should be Fe22Ni10 or similar).
5.  **Block 5**: Apply 5% strain and 0.1A rattling. Plot the Radial Distribution Function (RDF) of the original vs. distorted structure to show the broadening of peaks.
6.  **Block 6**: Show the `atoms.info` dictionary for a generated structure.

## 2. Behavior Definitions

### Scenario: SQS Generation
**GIVEN** a target composition of Fe 50%, Ni 50%.
**AND** a supercell size of 32 atoms.
**WHEN** the generator is executed.
**THEN** it should produce an `Atoms` object.
**AND** `atoms.get_chemical_symbols().count('Fe')` should be 16.
**AND** `atoms.get_chemical_symbols().count('Ni')` should be 16.
**AND** the structure should possess periodic boundary conditions.
**AND** the `config_type` should be `sqs`.

### Scenario: Strain and Rattling
**GIVEN** a cubic unit cell with side length 10.0 (Volume 1000.0).
**WHEN** `apply_strain` is called with a strain tensor corresponding to +10% hydrostatic expansion.
**THEN** the returned atom's cell volume should be approximately 1100.0.
**WHEN** `apply_rattle` is called with `sigma=0.1`.
**THEN** the atomic positions should differ from the original positions.
**AND** the cell vectors should remain unchanged.
**AND** the RMS displacement should be approximately 0.17 (sqrt(3)*0.1).

### Scenario: Metadata Tagging
**GIVEN** a generation pipeline.
**WHEN** a rattled structure is produced.
**THEN** `atoms.info` dictionary should contain:
    -   `config_type`: "rattled"
    -   `parent_id`: (ID of the SQS)
    -   `sigma`: 0.1
**THIS** allows us to filter the database later (e.g., "Select only undistorted structures").

### Scenario: Vacancy Generation
**GIVEN** a perfect 2x2x2 supercell of Silicon (8 atoms).
**WHEN** `create_vacancy` is requested.
**THEN** the result should be a list containing at least one structure.
**AND** the structure should contain 7 atoms.
**AND** the cell dimensions should match the input.
