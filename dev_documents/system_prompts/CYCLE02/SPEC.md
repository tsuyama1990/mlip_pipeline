# Cycle 02 Specification: Structure Generator

## 1. Summary
The "Structure Generator" is the creative engine of the MLIP pipeline. It is responsible for exploring the vast configuration space of a material system to identify structures that will maximally improve the potential's accuracy. This cycle moves beyond simple random structure generation; it implements an "Adaptive Exploration Policy" that intelligently decides *how* to sample based on the material's properties (e.g., band gap, bulk modulus). By the end of this cycle, the system will be able to generate diverse, physically meaningful atomic structures—including defects, strained cells, and high-temperature snapshots—ready for DFT calculation.

## 2. System Architecture

Files in **bold** are to be created or modified.

```ascii
src/mlip_autopipec/
├── domain_models/
│   ├── **structure.py**            # Enhance with tags/features
│   └── **enums.py**                # Add sampling strategy enums
├── components/
│   ├── generator/
│   │   ├── **__init__.py**
│   │   ├── **base.py**             # BaseGenerator
│   │   ├── **structure_generator.py** # Main implementation
│   │   ├── **adaptive_policy.py**  # The Policy Engine
│   │   └── **builder.py**          # Utility: Defects, Strains
│   └── ...
└── core/
    └── **orchestrator.py**         # Update to use generator
```

## 3. Design Architecture

### 3.1 Adaptive Exploration Policy (`adaptive_policy.py`)

This class encapsulates the "scientific intuition". It takes `MaterialFeatures` as input and returns a `SamplingConfig`.

**Input: `MaterialFeatures` (Pydantic)**
*   `band_gap`: Float (eV). Used to distinguish Metals vs Insulators.
*   `bulk_modulus`: Float (GPa). Used to set strain ranges.
*   `composition`: Dict. Used to set MC ratios.
*   `uncertainty_stats`: Dict (mean, max gamma).

**Output: `SamplingConfig` (Pydantic)**
*   `md_steps`: Int.
*   `mc_swap_ratio`: Float (0.0 to 1.0).
*   `temperature_schedule`: List[Tuple[Time, Temp]].
*   `strain_range`: Float (e.g., 0.15).
*   `defect_density`: Float (e.g., 0.01 vacancies/atom).

**Logic Example:**
```python
if features.band_gap < 0.1:  # Metal
    config.mc_swap_ratio = 0.5  # High swapping for alloys
else:
    config.mc_swap_ratio = 0.0  # No swapping for ionic/covalent
```

### 3.2 Structure Builder (`builder.py`)

A utility module for manipulating `ase.Atoms` objects deterministically.

**Functions:**
*   `apply_strain(atoms, strain_tensor) -> Atoms`: Deforms the cell.
*   `create_supercell(atoms, size) -> Atoms`: Repeats unit cell.
*   `inject_defects(atoms, vacancies=N, interstitials=M) -> Atoms`: Randomly removes/adds atoms while maintaining minimum distance constraints.
*   `rattle(atoms, stdev) -> Atoms`: Adds Gaussian noise to positions.

### 3.3 Structure Generator (`structure_generator.py`)

Inherits from `BaseGenerator`.

**Method: `generate(n_structures: int, strategy: StrategyEnum) -> List[Structure]`**
1.  **Cold Start:** If no previous potential exists, use `builder.py` to create random symmetry structures or perturbed standard crystals (FCC, BCC, HCP).
2.  **Adaptive Mode:** If a potential exists (from previous cycles), consult `AdaptiveExplorationPolicy` to determine parameters.
3.  **Execution:**
    *   Calls `builder.py` to generate initial candidates.
    *   (Future: Run short MD with cheap potential if available).
    *   Returns a list of `Structure` objects with metadata (tags).

## 4. Implementation Approach

1.  **Enhance Domain Models**: Add necessary fields to `Structure` (e.g., `tags` dict for provenance).
2.  **Implement Builder**: Create `builder.py` with robust ASE manipulations. Ensure `inject_defects` handles supercell creation automatically if the unit cell is too small.
3.  **Develop Policy Engine**: Implement `AdaptiveExplorationPolicy` with simple heuristic rules first (Metal vs Non-Metal).
4.  **Implement Generator**: Create `StructureGenerator`. Connect it to the Policy engine.
5.  **Integration**: Update `Orchestrator` to instantiate `StructureGenerator` via Factory.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Builder Logic**:
    *   Test `inject_defects`: Verify atom count changes correctly (N - 1 for vacancy). Ensure no atoms overlap (min distance check).
    *   Test `apply_strain`: Verify cell volume changes as expected.
*   **Policy Logic**:
    *   Test that `band_gap=0` triggers `mc_swap_ratio > 0`.
    *   Test that high `bulk_modulus` triggers smaller `strain_range`.

### 5.2 Integration Testing
*   **Generation Pipeline**:
    *   Configure `StructureGenerator` in `config.yaml`.
    *   Run `generator.generate(n=10)`.
    *   Verify 10 `Structure` objects are returned.
    *   Verify they are valid `ase.Atoms` (no NaN positions).
    *   Check `structure.provenance` tags are correct.
