# Cycle 02 Specification: Structure Generator & Initial Exploration

## 1. Summary
**Goal**: Implement the `StructureGenerator` component to create diverse atomic structures for initial training and ongoing exploration. This cycle introduces the **Adaptive Exploration Policy** engine, capable of switching between random structure search (RSS), M3GNet-based pre-screening, and physical manipulations (Strain/Defects).

**Key Features**:
*   **AdaptivePolicy**: A strategy pattern implementation that selects the best generator based on material type (Metal/Insulator) or current uncertainty.
*   **M3GNet Integration**: "Cold Start" capability using a pre-trained universal potential to find approximate minima.
*   **RandomGenerator**: Basic RSS for unbiased sampling.
*   **StrainGenerator**: Apply affine transformations to explore elastic properties.

## 2. System Architecture

Files to be implemented/modified are **bold**.

```ascii
src/mlip_autopipec/
├── domain_models/
│   ├── **generator.py**        # New Generator Config Models
│   └── ...
├── structure_generator/
│   ├── **__init__.py**
│   ├── **base.py**             # Abstract Base Class
│   ├── **policies.py**         # Adaptive Policy Engine
│   └── generators/
│       ├── **__init__.py**
│       ├── **random.py**       # RSS
│       ├── **strain.py**       # Elastic/EOS sampling
│       └── **m3gnet.py**       # Universal Potential (Optional)
└── tests/
    └── **test_generator/**
        ├── **test_policies.py**
        └── **test_generators.py**
```

## 3. Design Architecture

### 3.1. Domain Models (`src/mlip_autopipec/domain_models/generator.py`)

*   **`GeneratorType`** (Enum): `RANDOM`, `M3GNET`, `STRAIN`, `MD`, `MC`.
*   **`ExplorationContext`**:
    *   Holds current `cycle` (int), `best_structure` (Structure), `uncertainty_stats` (dict).
    *   Passed to `AdaptivePolicy.decide()` to choose the next action.

### 3.2. Generator Component (`src/mlip_autopipec/structure_generator/`)

#### `base.py`
*   **`BaseGenerator`** (ABC):
    *   `generate(n_structures: int, context: ExplorationContext) -> List[Structure]`
    *   Must return structures with proper `provenance` tags.

#### `policies.py`
*   **`AdaptivePolicy`**:
    *   The decision maker.
    *   Logic:
        *   If `cycle == 0`: Use `M3GNet` (if available) or `Random` for cold start.
        *   If `uncertainty` is high: Use `MD` with lower temperature (cautious).
        *   If `uncertainty` is low: Use `Strain` or `Defects` to push boundaries.

#### `generators/random.py`
*   **`RandomGenerator`**:
    *   Generates random structures with hard-sphere constraints (min_distance) to prevent nuclear fusion.
    *   Supports multicomponent systems based on composition config.

#### `generators/m3gnet.py`
*   **`M3GNetGenerator`**:
    *   Uses `m3gnet` (external lib) to relax random structures.
    *   Filters out unstable structures (high energy above hull).
    *   **Fallback**: If `m3gnet` is not installed, logs warning and falls back to `Random`.

#### `generators/strain.py`
*   **`StrainGenerator`**:
    *   Takes a seed structure and applies:
        *   Hydrostatic strain ($\pm 10\%$).
        *   Shear strain (preserving volume).
    *   Crucial for training EOS and Elastic Constants.

## 4. Implementation Approach

1.  **Define Generator Interfaces**: Create `base.py` and domain models.
2.  **Implement Random Generator**: Robust implementation with `ase.build` tools.
3.  **Implement Strain Generator**: Use `ase.deformation` logic.
4.  **Implement Adaptive Policy**: Simple rule-based logic initially.
5.  **Implement M3GNet Wrapper**: With try/except import block for optional dependency.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_generators.py`**:
    *   `RandomGenerator`: Ensure `min_distance` is respected (no overlapping atoms).
    *   `StrainGenerator`: Check volume changes match the applied strain.
*   **`test_policies.py`**:
    *   Mock `ExplorationContext` with different `cycle` numbers and assert the correct generator type is returned.

### 5.2. Integration Testing
*   **Generation Pipeline**: Verify that a generated list of structures can be serialized to JSON/XYZ and read back by `ase.io`.
