# Cycle 05: Selection (Active Learning Logic)

## 1. Summary
This cycle implements the "Brain" of the Active Learning loop: the **Selection** module. It bridges the gap between the Dynamics Engine (which finds dangerous structures) and the Oracle (which labels them). Its primary duties are to extract relevant snapshots from halted simulations, process them into periodic supercells suitable for DFT (**Periodic Embedding**), and filter out redundant structures to maximize data efficiency (**D-Optimality**).

## 2. System Architecture

We add the `phases/selection` module.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── orchestration/
│   └── phases/
│       └── **selection/**
│           ├── **__init__.py**
│           ├── **manager.py**         # SelectionPhase implementation
│           ├── **candidate.py**       # Candidate extraction from Dump
│           ├── **embedding.py**       # Periodic Embedding Logic
│           └── **active_set.py**      # D-Optimality Filter (MaxVol)
└── tests/
    └── **test_selection.py**
```

## 3. Design Architecture

### Candidate Extraction (`candidate.py`)
*   Reads the LAMMPS dump file.
*   Identifies the frame where `gamma > threshold`.
*   Identifies the specific atom indices responsible for the high uncertainty.

### Periodic Embedding (`embedding.py`)
*   **Problem**: We cannot just cut a cluster and put it in vacuum (surface effects) nor use the whole huge MD box (too expensive for DFT).
*   **Solution**:
    1.  Select the high-$\gamma$ atom and its neighbors within $R_{cut} + R_{buffer}$.
    2.  Construct a **minimal orthogonal supercell** that fits this cluster.
    3.  Populate this supercell with atoms, ensuring periodic boundaries are respected (i.e., it looks like bulk to the central atom).
*   **Output**: A `CandidateStructure` that is small enough for DFT but physically representative of the bulk environment.

### Active Set Selection (`active_set.py`)
*   Uses `pace_activeset` command or internal linear algebra.
*   Given a pool of candidates, selects those that maximize the determinant of the descriptor matrix (MaxVol algorithm).

## 4. Implementation Approach

1.  **Extraction**: Implement `read_halted_frame(dump_path) -> Atoms`.
2.  **Embedding**: Implement `create_embedded_supercell(atoms, center_atom_index, r_cut)`. This requires careful manipulation of lattice vectors and scaled positions.
3.  **Active Set**: Implement a wrapper around `pace_activeset`.
4.  **Integration**: `SelectionPhase.run()`:
    *   Input: List of Halted MD runs.
    *   Step 1: Extract frames.
    *   Step 2: Generate local perturbations (optional).
    *   Step 3: Run Active Set selection.
    *   Step 4: Apply Periodic Embedding to selected structures.
    *   Output: List of structures ready for Oracle.

## 5. Test Strategy

### Unit Testing
*   **`test_embedding.py`**: Create a known crystal structure. Select an atom. Verify that the embedded supercell correctly reproduces the neighbor list of the original structure (up to $R_{cut}$).
*   **`test_candidate.py`**: Parse a dummy dump file and extract the correct frame.

### Integration Testing
*   **Pipeline Flow**: Pass a dummy halted run -> Selection -> Verify output is a list of small, dense periodic structures.
