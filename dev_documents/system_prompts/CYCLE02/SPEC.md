# Cycle 02 Specification: Structure Generator & Adaptive Policy

## 1. Summary

In this cycle, we replace the `MockGenerator` with a functional `StructureGenerator`. The goal is to intelligently explore the chemical and structural space to create high-value training candidates. Instead of simple random sampling, we implement an **Adaptive Exploration Policy** that adjusts sampling strategies (MD vs. MC, Temperature ramping, Defect density) based on the material's characteristics and the current uncertainty state.

## 2. System Architecture

The following file structure will be created. **Bold** files are the targets for this cycle.

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── **policy.py**               # Pydantic Models for Exploration Policies
├── components/
│   ├── generators/
│   │   ├── **random.py**           # Basic Random Structure Generator
│   │   └── **adaptive.py**         # Adaptive Generator with Policy Logic
│   └── **base.py**                 # (Existing)
└── utils/
    └── **symmetry.py**             # Helper for creating distorted structures
tests/
└── **test_generator.py**           # Tests for Random and Adaptive Generators
```

## 3. Design Architecture

### 3.1. Domain Models (`domain_models/policy.py`)
*   **`ExplorationPolicy`**: Defines the strategy for a generation cycle.
    *   `md_mc_ratio`: float (Ratio of Molecular Dynamics to Monte Carlo steps)
    *   `temperature_schedule`: List[Tuple[float, float]] (Start/End temperatures)
    *   `defect_density`: float (Probability of creating vacancies/interstitials)
    *   `strain_range`: float (Max strain for deformation)

### 3.2. Generator Components (`components/generators/`)
*   **`RandomGenerator`**:
    *   `generate(n)`: Creates structures by randomly placing atoms in a box (Rattle), ensuring minimum distance constraints (using `ase.geometry.get_distances` or `packmol` logic).
*   **`AdaptiveGenerator`**:
    *   `__init__(config)`: Loads the policy configuration.
    *   `generate(n, context)`: Accepts a `context` (e.g., current iteration, uncertainty stats) to select the active policy.
        *   **Action 1 (Cold Start)**: If `iteration=0`, use heuristics (e.g., M3GNet or known crystal structures + rattle).
        *   **Action 2 (Refinement)**: If `uncertainty` is high, generate local distortions around the problematic structures.
        *   **Action 3 (Exploration)**: Run short, high-temperature MD/MC bursts (using ASE calculators like `EMT` or `LJ` as proxies if MLIP is not ready, or just geometric perturbations).

## 4. Implementation Approach

1.  **Policy Model**: Define the Pydantic model for `ExplorationPolicy` in `domain_models/policy.py`.
2.  **Random Generator**: Implement `RandomGenerator` using `ase.build.bulk` and random displacements. Ensure strict checks for overlapping atoms (distance < 1.5 Å).
3.  **Adaptive Logic**: Implement `AdaptiveGenerator`.
    *   Create a method `_select_policy(context)` that returns a specific `ExplorationPolicy`.
    *   Implement strategies: `perturb_positions`, `create_vacancy`, `apply_strain`.
4.  **Integration**: Update `config.py` to include `StructureGeneratorConfig` and allow selecting `adaptive` in the main config.

## 5. Test Strategy

### 5.1. Unit Testing
*   **`test_generator.py`**:
    *   **Test Random**: Request 10 structures. Assert that 10 `ase.Atoms` objects are returned. Assert that no two atoms are closer than `min_distance`.
    *   **Test Adaptive**:
        *   Mock the context (e.g., `iteration=1`).
        *   Verify that `generate` returns structures with defects if the policy dictates `defect_density > 0`.
        *   Verify that `generate` returns strained structures if `strain_range > 0`.

### 5.2. Integration Testing
*   **Pipeline Test**: Update the mock loop (from Cycle 01) to use `AdaptiveGenerator` (but still Mock Oracle/Trainer).
*   **Verification**: Check logs to see "Generated 10 structures using Policy: High-Temperature". Inspect the generated structures (saved to disk) to confirm they are not empty.
