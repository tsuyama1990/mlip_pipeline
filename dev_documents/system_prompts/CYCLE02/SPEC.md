# Cycle 02: Structure Generator & Adaptive Policy

## 1. Summary

Cycle 02 focuses on implementing the "Structure Generator" module, which is responsible for creating diverse and physically relevant atomic structures for the training dataset. Unlike traditional random sampling, this module will implement an "Adaptive Exploration Policy" that intelligently decides *how* to sample the configuration space based on the current state of knowledge (uncertainty) and the target material's properties.

We will move away from the "Mock" generator and build a robust `StructureGenerator` that can create supercells, introduce defects (vacancies, interstitials), apply strains, and perform random displacements. The policy engine will determine the ratio of these operations (e.g., 70% MD snapshots, 20% high-temperature Monte Carlo, 10% strained lattices).

## 2. System Architecture

The following file structure will be modified/created. Files in **bold** are the primary deliverables for this cycle.

```
.
├── config.yaml
├── src/
│   └── mlip_autopipec/
│       ├── core/
│       │   ├── config.py         # Update GeneratorConfig
│       │   └── **policy.py**     # Adaptive Exploration Logic
│       ├── domain_models/
│       │   ├── inputs.py
│       │   └── enums.py          # Update GeneratorType
│       ├── components/
│       │   ├── base.py
│       │   └── **generator.py**  # Concrete Implementation
│       └── utils/
│           └── **geometry.py**   # Geometric transformations
└── tests/
    ├── **test_generator.py**
    └── **test_policy.py**
```

## 3. Design Architecture

### Adaptive Exploration Policy (`core/policy.py`)
The policy engine is a stateless functional component that takes a `PolicyInput` (material properties, uncertainty distribution) and returns a `ExplorationStrategy`.

*   `ExplorationStrategy`: A Pydantic model defining parameters like:
    *   `md_mc_ratio`: Ratio of MD steps to Monte Carlo swap steps.
    *   `temperature_schedule`: List of temperatures to sample.
    *   `strain_range`: Range for EOS calculations (e.g., 0.05).
    *   `defect_density`: Probability of introducing defects.

### Structure Generator (`components/generator.py`)
The `StructureGenerator` implements the `BaseGenerator` interface. It uses the strategy provided by the policy to generate structures.

*   `generate(n_structures)`:
    1.  Consult `AdaptiveExplorationPolicy` to get the strategy.
    2.  Based on the strategy, call internal methods:
        *   `_apply_strain(atoms, strain_tensor)`
        *   `_introduce_defect(atoms, defect_type)`
        *   `_rattle(atoms, displacement_std)`
        *   `_supercell(atoms, size)`
    3.  Return a list of `Structure` objects (which wrap `ase.Atoms`).

### Geometric Utilities (`utils/geometry.py`)
Helper functions for manipulating atomic structures using ASE.
*   `create_supercell`: Uses `ase.build.make_supercell`.
*   `apply_strain`: Modifies the cell vectors and scales positions.
*   `add_vacuum`: Adds vacuum layers for surface creation.

## 4. Implementation Approach

1.  **Policy Logic**: Implement `core/policy.py`. Start with simple heuristic rules (e.g., "If uncertainty is high, lower the temperature and increase random displacement").
2.  **Geometric Utils**: Implement `utils/geometry.py` to handle the actual atomic manipulations. Ensure these functions are robust and handle edge cases (e.g., minimum image convention).
3.  **Generator Implementation**: Implement `components/generator.py`. Wire it up to use the config and the policy.
4.  **Configuration Update**: Update `core/config.py` to include parameters for the policy (e.g., `default_temperature`, `max_strain`).
5.  **Integration**: Update the `Orchestrator` to instantiate the real `StructureGenerator` instead of the Mock one (controlled via config).

## 5. Test Strategy

### Unit Testing
*   **Policy Test**: Feed different `PolicyInput` scenarios (e.g., high uncertainty vs low uncertainty) and assert that the output `ExplorationStrategy` changes logically.
*   **Geometry Test**:
    *   Create a unit cell. Apply 5% strain. Verify the cell volume increases by approx 15% (cubic).
    *   Introduce a vacancy. Verify the atom count decreases by 1.
    *   Test `check_min_distance` to ensure atoms are not placed too close (prevent nuclear fusion).

### Integration Testing
*   **Generator Output**:
    *   Configure the generator to produce 100 structures.
    *   Run `generate()`.
    *   Verify that the output list contains a mix of structures (some strained, some with defects) according to the configured ratio.
    *   Visualize a few structures using `ase.visualize` (manual check) or assert statistical properties (e.g., distribution of volumes).
