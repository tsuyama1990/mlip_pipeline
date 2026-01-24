# Cycle 02 Specification: Structure Generation

## 1. Summary
Cycle 02 implements the **Structure Generator** module, the "Explorer" of the system. This module is responsible for creating atomic structures that will be used for training and testing. It supports random generation, defect introduction, and heuristic-based sampling (Adaptive Policy groundwork).

## 2. System Architecture

```ascii
mlip_autopipec/
├── config/
│   └── schemas/
│       └── generator.py        # **Generator Configuration**
├── generator/
│   ├── __init__.py
│   ├── builder.py              # **StructureBuilder Class**
│   ├── transformations.py      # **Mutation Logic (Rattle, Strain)**
│   └── defects.py              # **Point Defect Generation**
```

**Key Modifications:**
- Add `mlip_autopipec/generator/` package.
- Implement `StructureBuilder` to coordinate generation strategies.

## 3. Design Architecture

### 3.1. Structure Builder (`generator/builder.py`)
The factory class that takes a configuration and produces a list of `ase.Atoms`.
- **Input**: `GeneratorConfig` (composition, supercell size, strategies).
- **Output**: `List[CandidateStructure]`.
- **Strategies**:
  - `RandomSubstitution`: Randomly populate a lattice.
  - `Rattle`: Apply random Gaussian noise to positions.
  - `Strain`: Apply deformation tensor to cell.

### 3.2. Transformations (`generator/transformations.py`)
Pure functions that take `Atoms` and return modified `Atoms`.
- `apply_strain(atoms, strain_tensor)`
- `apply_rattle(atoms, stdev)`
- **Invariant**: Must preserve the number of atoms (unless defect generation) and species mapping.

### 3.3. Defect Generator (`generator/defects.py`)
- **Function**: `create_vacancy(atoms, indices)`, `create_interstitial`.
- **Logic**: Use `pymatgen` or custom logic to remove/add atoms while maintaining periodic boundary conditions.

## 4. Implementation Approach

1.  **Config Schema**: Define `GeneratorConfig` in `config/schemas/generator.py` (e.g., `rattle_stdev: float`, `num_structures: int`).
2.  **Transformations**: Implement `apply_strain` and `apply_rattle` with unit tests.
3.  **Defects**: Implement basic vacancy generation.
4.  **Builder**: Create `StructureBuilder` class that:
    - Accepts a seed structure (e.g., from MP or CIF).
    - Iterates `num_structures` times.
    - Applies selected transformations pipeline.
    - Returns list of objects.

## 5. Test Strategy

### 5.1. Unit Testing
- **Transformations**:
    - Assert `rattle` changes positions but not cell.
    - Assert `strain` changes cell and positions (fractional coords stay same?).
- **Builder**:
    - Check if it generates exact number of requested structures.
    - Check reproducibility (using random seed).

### 5.2. Integration Testing
- **Config to Generation**:
    1.  Load a config specifying "Generate 10 rattled FCC Al structures".
    2.  Run `StructureBuilder`.
    3.  Verify the output is a list of 10 valid ASE objects.
    4.  Save them to the DB (using Cycle 01's DB Manager).
