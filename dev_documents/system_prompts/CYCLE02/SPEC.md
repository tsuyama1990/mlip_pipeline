# Cycle 02 Specification: Structure Generator & Adaptive Policy

## 1. Summary

Cycle 02 implements the `StructureGenerator` component, which is responsible for proposing new atomic configurations to explore the chemical and structural space. In this cycle, we move beyond simple random generation and introduce an "Adaptive Exploration Policy". This policy dynamically adjusts the sampling strategy based on the current state of the Active Learning loop (e.g., initial exploration vs. refinement).

The `StructureGenerator` must be capable of:
1.  **Cold Start**: Generating initial structures when no potential exists (using Random, M3GNet, or Templates).
2.  **Perturbation**: Modifying existing structures (Rattling, Scaling).
3.  **Defect Injection**: Creating Vacancies, Interstitials, and Antisites to robustly sample defects.
4.  **Policy Execution**: Deciding *which* generator to use based on a configuration schedule.

By the end of this cycle, the `Orchestrator` will be able to call `generator.generate()` and receive a list of diverse, valid `Structure` objects ready for DFT calculation.

## 2. System Architecture

This cycle focuses on the `components/generator` package.

### File Structure
Files to be created/modified in this cycle are marked in **bold**.

```
src/mlip_autopipec/
├── components/
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── **base.py**             # Enhanced Abstract Base Class
│   │   ├── **random.py**           # Random/Rattle Generator
│   │   ├── **defect.py**           # Defect Generator (Vacancy/Interstitial)
│   │   ├── **m3gnet.py**           # Pre-trained MLIP Generator (Optional)
│   │   └── **policy.py**           # The Adaptive Logic
│   └── factory.py                  # Update to register Generators
├── core/
│   └── **candidate_generator.py**  # Helper for local perturbations
└── domain_models/
    └── **config.py**               # Add GeneratorConfig details
└── tests/
    └── **test_generator.py**
```

## 3. Design Architecture

### 3.1. Generator Configuration (`domain_models/config.py`)
Update `GeneratorConfig` to include:
*   `initial_count`: Number of structures for cycle 0.
*   `refine_count`: Number of structures for subsequent cycles.
*   `strategy`: List of strategies (e.g., `["random", "defect", "m3gnet"]`).
*   `rattle_amplitude`: Float (e.g., 0.1 Å).
*   `supercell_size`: List[int] (e.g., [2, 2, 2]).

### 3.2. Adaptive Policy (`components/generator/policy.py`)
The `AdaptivePolicy` class decides the mix of structures to generate.
*   **Input**: `cycle_index`, `previous_validation_metrics`.
*   **Output**: A dictionary of `{ "strategy_name": count }`.
*   **Logic**:
    *   Cycle 0: 100% Random/M3GNet (Exploration).
    *   Cycle > 0: Mix of Random (20%) and Defect/Perturbed (80%) based on `GeneratorConfig`.

### 3.3. Generators (`components/generator/*.py`)
All generators inherit from `BaseGenerator`.
*   `RandomGenerator`: Takes a prototype structure, creates a supercell, and applies random displacement (Rattle) and volume scaling.
*   `DefectGenerator`: Takes a prototype, creates a supercell, removes N atoms (Vacancy), adds N atoms (Interstitial), or swaps types (Antisite).
*   `M3GNetGenerator`: Uses `matgl` (if installed) to relax random structures to a local minimum, providing "reasonable" starting points.

## 4. Implementation Approach

1.  **Enhance `BaseGenerator`**: Ensure `generate(n_structures)` returns `List[Structure]`.
2.  **Implement `RandomGenerator`**: Use `ase.Atoms.rattle()` and `set_cell(scale_atoms=True)`.
3.  **Implement `DefectGenerator`**:
    *   Use `pymatgen.analysis.defects` or implement simple index-based removal/addition logic using ASE tags.
    *   Ensure strict checking (don't remove the last atom).
4.  **Implement `Policy` Logic**: Create a class that reads the config and returns the generation plan.
5.  **Factory Registration**: Update `ComponentFactory` to instantiate the correct generator based on string names.
6.  **Orchestrator Integration**: Update `Orchestrator.run()` to call `self.generator.generate()` at the start of a cycle.

## 5. Test Strategy

### 5.1. Unit Testing (`tests/test_generator.py`)
*   **Random Generator**:
    *   Input: A single unit cell (e.g., Al FCC).
    *   Action: Generate 10 structures.
    *   Assert: Structures have noise (positions != original). Cell volumes vary if scaling enabled.
*   **Defect Generator**:
    *   Input: A 2x2x2 Supercell.
    *   Action: Generate 1 Vacancy.
    *   Assert: Number of atoms is `original - 1`.
*   **Policy**:
    *   Input: Cycle 0. Config strategy="adaptive".
    *   Assert: Returns mostly "random".

### 5.2. Integration Testing
*   **Orchestrator Call**:
    *   Mock the `Oracle` (return dummy energies).
    *   Run Orchestrator for 1 cycle.
    *   Assert that `generator.generate()` was called and returned non-empty list.
    *   Assert that generated structures are saved to `active_learning/cycle_001/candidates`.
