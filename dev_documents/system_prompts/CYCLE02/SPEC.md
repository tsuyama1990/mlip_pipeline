# Cycle 02 Specification: Structure Generator

## 1. Summary
In this cycle, we implement the "Explorer" component of the system: the `StructureGenerator`. The role of this module is to propose atomic configurations that will be labeled by the Oracle. Unlike simple random sampling, this module implements an **Adaptive Exploration Policy**. It decides *how* to sample (e.g., Molecular Dynamics vs Monte Carlo, High Temperature vs Low Temperature) based on the material's properties (e.g., predicted melting point) or the current uncertainty state. This cycle also introduces the `M3GNet` integration (or a mock equivalent) for "Cold Start" structure generation, allowing the system to begin learning even without an initial dataset.

## 2. System Architecture

### 2.1. File Structure
files to be created/modified in this cycle are bolded.

```text
src/mlip_autopipec/
├── components/
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── base.py                 # [CREATE] Abstract Base Class
│   │   ├── adaptive_policy.py      # [CREATE] Strategy Logic
│   │   ├── structure_gen.py        # [CREATE] Main Generator Implementation
│   │   └── utils.py                # [CREATE] Supercell/Distortion helpers
├── domain_models/
│   ├── config.py                   # [MODIFY] Add GeneratorConfig
│   └── enums.py                    # [MODIFY] Add GeneratorType
└── core/
    └── orchestrator.py             # [MODIFY] Integrate Generator into explore()
```

### 2.2. Component Interaction
1.  **`Orchestrator`** calls `generator.generate(n_candidates=10, context=state)`.
2.  **`StructureGenerator`**:
    *   Calls `AdaptiveExplorationPolicy.decide_strategy(context)`.
    *   Receives a `Strategy` object (e.g., `action='MD', temp=1000K`).
    *   Executes the strategy:
        *   If `Cold Start`: Calls M3GNet to relax a random crystal.
        *   If `Perturbation`: Takes an existing structure and applies random noise / strain.
        *   If `MD`: (Mock for now, or simple ASE MD) Runs a short trajectory.
3.  Returns a list of `ase.Atoms` objects to the Orchestrator.

## 3. Design Architecture

### 3.1. Domain Models

#### `enums.py`
*   `GeneratorType`: `RANDOM`, `M3GNET`, `ADAPTIVE`.
*   `ExplorationStrategy`: `RANDOM_DISTORTION`, `HIGH_TEMP_MD`, `DATA_MINING`.

#### `config.py`
*   `GeneratorConfig`:
    *   `strategy`: GeneratorType
    *   `distortion_magnitude`: float (default 0.1)
    *   `supercell_matrix`: List[int]

### 3.2. Core Logic

#### `adaptive_policy.py`
*   **Responsibility**: The "Brain" of exploration.
*   **Logic**:
    *   Input: `WorkflowState` (cycle number, previous validation error).
    *   Output: `ExplorationParameters` (temperature, pressure, method).
    *   **Rule Example**:
        *   If `cycle == 0`: Use `RANDOM_DISTORTION` or `M3GNET`.
        *   If `cycle > 0`: Use `HIGH_TEMP_MD` (simulated).

#### `structure_gen.py`
*   **Responsibility**: Manipulate atoms.
*   **Methods**:
    *   `apply_distortion(atoms, magnitude)`: Rattles positions and deforms cell.
    *   `make_supercell(atoms, matrix)`: Expands unit cell.
    *   `generate_initial_population()`: Reads CIF/POSCAR or creates from scratch.

## 4. Implementation Approach

### Step 1: Interface Definition
*   Define `BaseGenerator` abstract class in `components/generator/base.py`.
*   Define `generate(self, count: int) -> List[Atoms]`.

### Step 2: Policy Engine
*   Implement `AdaptiveExplorationPolicy`.
*   Start with a simple heuristic: "Cycle 0 -> Random, Cycle N -> Random from previous best".

### Step 3: Generator Implementation
*   Implement `StructureGenerator`.
*   Use `ase.build` for creating bulk structures (MgO, FePt) for testing.
*   Implement `rattle` and `strain` functions.

### Step 4: Orchestrator Integration
*   Update `Orchestrator.explore()` to instantiate and call the generator.
*   Save the generated structures to `work_dir/iter_XX/candidates/`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_policy.py`**:
    *   Context cycle=0 -> Expect Cold Start strategy.
    *   Context cycle=5 -> Expect Active Learning strategy.
*   **`test_generator_utils.py`**:
    *   Test `apply_distortion`: Ensure atom positions change but count remains same.
    *   Test `make_supercell`: Ensure atom count multiplies correctly.

### 5.2. Integration Testing
*   **`test_exploration_phase.py`**:
    *   Configure Orchestrator with `AdaptiveGenerator`.
    *   Run one cycle.
    *   Assert that `work_dir/iter_001/candidates/` contains `.xyz` files.
    *   Assert that the structures are valid readable ASE atoms.
