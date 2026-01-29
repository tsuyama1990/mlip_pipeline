# Cycle 02 Specification: Basic Exploration

## 1. Summary

Cycle 02 implements the **Structure Generator** module. This module is responsible for creating atomic structures that serve as candidates for DFT calculations. In this cycle, we focus on two main strategies: "Cold Start" (initial random/template generation) and basic "Random Perturbation". The goal is to produce a diverse set of valid atomic structures (avoiding overlapping atoms) that can be passed to the Oracle in the next cycle. We will leverage `ase` and `pymatgen` for symmetry analysis and structure manipulation.

## 2. System Architecture

### File Structure

Files to be created/modified are **bold**.

```ascii
src/mlip_autopipec/
├── domain_models/
│   ├── config.py                     # Update: Add ExplorationConfig
│   └── structure.py                  # Update: Enhance Structure validation
├── **modules/**
│   ├── **__init__.py**
│   └── **structure_gen/**
│       ├── **__init__.py**
│       ├── **generator.py**          # Main StructureGenerator class
│       └── **strategies.py**         # Cold Start & Random strategies
└── orchestration/
    └── phases/
        ├── **__init__.py**
        └── **exploration.py**        # ExplorationPhase implementation
```

## 3. Design Architecture

### Domain Models

#### `config.py`
*   **`ExplorationConfig`**:
    *   `strategy`: Enum (RANDOM, TEMPLATE, ADAPTIVE)
    *   `supercell_size`: List[int]
    *   `rattle_amplitude`: float
    *   `num_candidates`: int

### Components (`modules/structure_gen/`)

#### `generator.py`
*   **`StructureGenerator`**:
    *   `__init__(config: ExplorationConfig)`
    *   `generate_initial_set(composition) -> List[Candidate]`: For Cold Start.
    *   `apply_strategy(strategy, base_structures) -> List[Candidate]`: Apply specific strategy.

#### `strategies.py`
*   **`ColdStartStrategy`**:
    *   Takes a chemical formula (e.g., "Ti3Al").
    *   Uses `pymatgen` or `ase` to generate prototype structures (fcc, bcc, hcp) or loads from Materials Project (mocked/cached for now).
    *   Applies random supercells and rattling.
*   **`RandomPerturbationStrategy`**:
    *   Takes an input structure.
    *   Applies random strain (cell deformation).
    *   Applies random atomic displacement (rattling).

### Orchestration (`orchestration/phases/exploration.py`)

#### `ExplorationPhase`
*   Implements the `Phase` protocol (to be defined).
*   **`execute(state, config)`**:
    *   If state.dataset is empty -> Call `StructureGenerator.generate_initial_set`.
    *   Else -> (Future: Adaptive logic) Call `StructureGenerator.apply_strategy`.
    *   Update `state` with generated candidates.

## 4. Implementation Approach

1.  **Update Config**: Add `ExplorationConfig` to `domain_models/config.py`.
2.  **Implement Strategies**:
    *   Create `strategies.py`. Implement `RattledStructure` generator using `ase.Atoms.rattle`.
    *   Implement `StrainGenerator` using `ase.Atoms.set_cell`.
3.  **Implement Generator**:
    *   Create `generator.py` that delegates to `strategies.py`.
    *   Ensure generated structures check for minimal atomic distance (sanity check).
4.  **Implement Exploration Phase**:
    *   Create `orchestration/phases/exploration.py`.
    *   Wire it into the `WorkflowManager` (from Cycle 01).
5.  **CLI Integration**:
    *   Update `mlip-auto run-loop` to execute the Exploration phase.

## 5. Test Strategy

### Unit Testing
*   **`test_structure_gen.py`**:
    *   Generate structures for "Si". Verify count.
    *   Check that atoms are not too close (collision check).
    *   Verify `rattle` actually moves atoms.
*   **`test_config_exploration.py`**:
    *   Verify `ExplorationConfig` parses correctly from YAML.

### Integration Testing
*   **`test_exploration_phase.py`**:
    *   Initialize a `WorkflowState`.
    *   Run `ExplorationPhase.execute`.
    *   Verify that `WorkflowState` now contains a list of candidates.
