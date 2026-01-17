# Cycle 03: User Acceptance Testing (UAT)

## 1. Test Scenarios

Cycle 03 is about creativity. The system must demonstrate it can imagine new, physically relevant structures.

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-03-01** | High | Alloy Generation | Verify that the system can generate a set of chemically disordered alloy structures with varying lattice strains. This is essential for multicomponent systems. |
| **UAT-03-02** | Medium | Defect Creation | Verify that the system can automatically generate vacancy and interstitial defects for a given crystal. This ensures the potential learns about diffusion barriers. |
| **UAT-03-03** | Medium | Vibrational Sampling | Verify that the NMS generator produces structures that are distorted from equilibrium but not "broken" (i.e., bond lengths are within physical tolerance). |

### Recommended Notebooks
*   `notebooks/UAT_03_Generators.ipynb`:
    1.  Initialize `AlloyGenerator` for "FeNi".
    2.  Generate 5 strained structures.
    3.  Visualize them (using `ase.visualize` or simply printing cell vectors) to confirm distortion.
    4.  Initialize `DefectGenerator` for Silicon.
    5.  Generate a vacancy and print the number of atoms (should decrease by 1).

## 2. Behavior Definitions

### UAT-03-01: Alloy & Strain

**Narrative**:
The user needs to train a potential for Fe-Ni. They need structures that cover the full composition range and various strain states to learn the elastic constants. They ask the `AlloyGenerator` for 10 samples. The system uses SQS to create random-like distributions and applies random strains. The user checks the output and sees structures with 70% Fe, 30% Ni, and unit cells that are slightly triclinic (distorted from cubic).

```gherkin
Feature: Structure Generation

  Scenario: Generating Random Alloys with Strain
    GIVEN a target composition of Fe70Ni30
    WHEN the AlloyGenerator is requested to create 10 structures with 5% max strain
    THEN each structure should contain Fe and Ni atoms
    AND the ratio of Fe:Ni should be approximately 7:3
    AND the cell volumes should vary between 0.85 and 1.15 times the equilibrium volume
    AND the configuration type metadata should be recorded as "sqs_strain"
    AND the cell vectors should generally be non-orthogonal

  Scenario: Generating Vacancy Defects
    GIVEN a perfect 32-atom Silicon supercell
    WHEN the DefectGenerator creates a single vacancy
    THEN the resulting structure should have 31 atoms
    AND the configuration type should be labeled "vacancy"
    AND the remaining atoms should retain their crystal lattice positions (mostly)
    AND the energy (if calculated cheaply) should be higher than the perfect crystal
```

### UAT-03-03: Vibrational Sampling

**Narrative**:
To train the potential for finite temperature MD, we need snapshot structures that look like a vibrating lattice. Randomly rattling atoms is too uncorrelated. The user asks the `MoleculeGenerator` to use Normal Mode Sampling at 300K. The system calculates the phonons (approximate) and displaces atoms along the modes. The user verifies that the displacements follow a physical distribution (larger amplitudes for low-frequency modes).

```gherkin
Feature: Normal Mode Sampling

  Scenario: Sampling Thermal Vibrations
    GIVEN a water molecule equilibrium structure
    WHEN the MoleculeGenerator runs NMS at 300K for 50 samples
    THEN the O-H bond lengths in the generated samples should vary around the equilibrium value
    AND the variation should be consistent with a Boltzmann distribution at 300K
    AND no atoms should overlap (distance < 0.5 A)
```
