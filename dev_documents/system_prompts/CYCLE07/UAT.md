# Cycle 07: User Acceptance Testing (UAT)

## 1. Test Scenarios

Cycle 07 is about active intelligence. We verify the loop closure.

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-07-01** | High | Periodic Embedding | Verify that the system can excise a local environment into a periodic box and correctly mask the buffer forces. This is the hardest algorithm in the project. |
| **UAT-07-02** | High | OTF Uncertainty Detection | Verify that the system stops an MD run when extrapolation grade exceeds the threshold. |

### Recommended Notebooks
*   `notebooks/UAT_07_Inference.ipynb`:
    1.  Create a large random structure (simulating an MD snapshot).
    2.  Pick a random atom.
    3.  Run `extract_periodic_box`.
    4.  Visualize the result: Show the small box, highlight the "Core" atoms in Red and "Buffer" atoms in Blue.
    5.  Print the `force_mask` array to verify 1s and 0s.

## 2. Behavior Definitions

### UAT-07-01: Embedding & Masking

**Narrative**:
An MD simulation of a 10,000 atom supercell finds a crack tip with high uncertainty. We cannot run DFT on 10,000 atoms. We need to cut out the crack tip. We select the atom at the tip. We define a 5A radius. We extract a box of size 12A around it. The atoms between 5A and 6A are the "buffer". We verify that the new small box is periodic (atoms crossing the boundary are wrapped). We verify that the mask array has 1.0 for the crack tip atoms and 0.0 for the boundary atoms.

```gherkin
Feature: Periodic Embedding

  Scenario: Extracting a Local Environment
    GIVEN a large MD frame (1000 atoms) with periodic boundaries
    AND a target atom ID located near the center
    WHEN the Periodic Embedding tool is run with radius=5.0 and buffer=3.0
    THEN a new Atoms object (approx 16 Angstrom box) should be returned
    AND the number of atoms should be significantly smaller than 1000
    AND atoms within 5.0 Angstroms of the center should have mask=1.0
    AND atoms between 5.0 and 8.0 Angstroms should have mask=0.0
    AND the new box should satisfy periodic boundary conditions (no atoms overlapping)
```

### UAT-07-02: OTF Loop

**Narrative**:
We start an MD simulation of melting Aluminum. The temperature ramps up. At 900K, the lattice starts to break. The ML potential hasn't seen liquid Al before. The extrapolation grade spikes to 15. The `OTFManager` sees this spike. It sends a SIGTERM to LAMMPS. It extracts the liquid snapshot. It puts it in the DFT queue.

```gherkin
Feature: On-The-Fly Active Learning

  Scenario: Detecting High Uncertainty
    GIVEN an MD simulation running with a YACE potential
    AND the potential has not been trained on liquid structures
    WHEN the simulation temperature exceeds the melting point
    THEN the extrapolation grade (gamma) should eventually exceed the threshold (e.g., 5.0)
    AND the simulation should stop automatically
    AND the high-uncertainty configuration should be extracted
    AND the system should flag this configuration for DFT calculation
```
