# Cycle 07 UAT: Scalable Inference Engine (Part 2)

## 1. Test Scenarios

| ID | Priority | Name | Description |
| :--- | :--- | :--- | :--- |
| **UAT-07-01** | High | **Local Environment Extraction** | Verify that the system can correctly excise a local atomic environment around a specific atom from a large supercell, preserving the local geometry and neighbor relationships. |
| **UAT-07-02** | High | **Force Mask Generation** | Verify that the system generates a force mask array where atoms in the core region are weighted 1.0 and atoms in the buffer region are weighted 0.0. |
| **UAT-07-03** | Medium | **Periodic Box Wrapping** | Verify that the extraction logic correctly handles atoms near the periodic boundaries of the original simulation box (Minimum Image Convention). |
| **UAT-07-04** | Low | **Metadata Tracing** | Verify that the new small structure retains metadata pointing back to the original large simulation (UUID, Frame Number). |

### Recommended Demo
Create `demo_07_embedding.ipynb`.
1.  **Block 1**: Create a 5x5x5 FCC Aluminum supercell.
2.  **Block 2**: Randomly displace one atom (Atom ID 50).
3.  **Block 3**: Run `EmbeddingExtractor` on Atom 50 with Core=4.0, Buffer=2.0.
4.  **Block 4**: Print the number of atoms in the extracted cell (should be small, e.g., ~50).
5.  **Block 5**: Print the `force_mask` array. Verify the center atom has mask 1.
6.  **Block 6**: Visualize the result (2D plot), coloring atoms by their mask value.

## 2. Behavior Definitions

### Scenario: Core-Shell Extraction
**GIVEN** a large dense system (1000 atoms).
**WHEN** `extract(center_idx, core=3.0, buffer=2.0)` is called.
**THEN** the returned atom object should contain all atoms within distance 5.0 of `center_idx`.
**AND** the cell size should be approximately 10.0x10.0x10.0.
**AND** the central atom should be at the center of the new cell.
**AND** the bond lengths between the central atom and its neighbors should be identical to the original system.

### Scenario: Mask Logic
**GIVEN** an extracted cluster.
**WHEN** `ForceMasker.apply(atoms, radius=3.0)` is called.
**THEN** `atoms.arrays['force_mask']` should be created.
**AND** for any atom $i$, if $dist(i, center) < 3.0$, mask should be 1.
**AND** if $dist(i, center) > 3.0$, mask should be 0.
**AND** the array length should equal `len(atoms)`.

### Scenario: Provenance Preservation
**GIVEN** an extracted structure.
**WHEN** inspecting its `info` dictionary.
**THEN** it should contain `src_id` (UUID of the original MD frame).
**AND** it should contain `src_index` (original index of the focal atom).
**THIS** allows us to trace a training point back to the exact moment in the simulation where it failed.

### Scenario: Boundary Condition
**GIVEN** an atom at [0.1, 0.1, 0.1] in a 10x10x10 box.
**AND** a neighbor at [9.9, 0.1, 0.1] (distance 0.2 via PBC).
**WHEN** extraction happens around the first atom.
**THEN** the neighbor should be included in the extracted cluster.
**AND** their distance in the new cluster should still be 0.2.
