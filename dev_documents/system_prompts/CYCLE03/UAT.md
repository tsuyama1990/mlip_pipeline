# User Acceptance Test (UAT): Cycle 03

## 1. Test Scenarios

### Scenario 03-01: The Explorer's Map (Priority: Medium)
**Objective**: Verify that the system can generate a set of strained structures for initial training.

**Description**:
To learn elastic constants, the potential needs to see unit cells that are squashed and stretched. The user wants to ensure that the system automatically generates these "EoS (Equation of State)" configurations.

**User Journey**:
1.  User creates a config with `exploration_strategy: strain`.
2.  User runs the system on a perfect Crystal.
3.  The system logs "Generating 20 strain samples (+/- 10%)."
4.  The system outputs `candidates.xyz` containing 20 frames.
5.  User checks the file: Frame 1 is compressed (high density), Frame 20 is expanded (low density).

**Success Criteria**:
*   20 valid structures are produced.
*   Volumes of the structures vary linearly or according to the specified range.

### Scenario 03-02: The Defect Hunter (Priority: High)
**Objective**: Verify that the "Periodic Embedding" logic works for local defects.

**Description**:
The user wants to train the potential to handle vacancies. They ask the system to generate a vacancy structure. The system must create a supercell (to isolate the vacancy) and then cut a computable box.

**User Journey**:
1.  User inputs a primitive cell of MgO.
2.  User requests `exploration_strategy: defect`.
3.  The system internally creates a $3\times3\times3$ supercell.
4.  The system removes one Oxygen atom.
5.  The system (optionally) applies "Periodic Embedding" to cut a smaller optimized box if the supercell is too huge (though for $3\times3\times3$ it might keep it).
6.  The system outputs a structure with formula Mg27 O26.

**Success Criteria**:
*   The output structure has exactly one fewer atom than the perfect supercell.
*   The lattice vectors are orthogonal (or appropriate for the embedding).

## 2. Behavior Definitions (Gherkin)

### Feature: Structure Generation

```gherkin
Feature: Adaptive Exploration

  Scenario: Generate Strain Samples
    GIVEN a primitive unit cell
    WHEN the StrainGenerator is invoked with range +/- 10%
    THEN it should produce a trajectory of structures
    AND the volume of the smallest structure should be ~90% of original
    AND the volume of the largest structure should be ~110% of original

  Scenario: Periodic Embedding of Defects
    GIVEN a large supercell with a single vacancy
    AND a cutoff radius of 5.0 Angstroms
    WHEN the EmbeddingHandler extracts a box around the vacancy
    THEN the new box dimensions should be at least 10.0 Angstroms (2 * cutoff)
    AND the atomic environment within the cutoff should be identical to the supercell
```
