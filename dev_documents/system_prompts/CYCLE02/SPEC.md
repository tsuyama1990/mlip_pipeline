# Cycle 02: Structure Generator Specification

## 1. Summary

The goal of Cycle 02 is to implement the **Structure Generator** module. This component is responsible for creating the initial atomic configurations that seed the active learning loop, as well as generating candidate structures during the refinement phase. A key innovation in this cycle is the **Adaptive Exploration Policy**, which dynamically adjusts sampling parameters (e.g., MD temperature, MC swap ratio) based on the material's properties (e.g., predicted melting point, band gap). This allows the system to explore chemical space more efficiently than simple random sampling.

## 2. System Architecture

This cycle focuses on the `components/generator` package and integration with the `ase` library.

### File Structure

The following file structure will be created. **Bold** files are to be implemented in this cycle.

*   **`src/`**
    *   **`mlip_autopipec/`**
        *   **`components/`**
            *   **`generator/`**
                *   **`__init__.py`**
                *   **`structure_generator.py`** (Main class)
                *   **`policies.py`** (Adaptive Exploration Policy)
                *   **`random_generator.py`** (Baseline/Fallback)
                *   **`defect_generator.py`** (Point defects, strains)
        *   **`core/`**
            *   **`structure_utils.py`** (ASE helpers)

## 3. Design Architecture

### 3.1 Components

#### `BaseGenerator` (Refinement)
The abstract base class defined in Cycle 01 will be extended to support adaptive parameters.
*   **`generate(n_structures: int, context: dict) -> list[Structure]`**: Context includes cycle number, previous metrics, etc.

#### `StructureGenerator`
The main implementation. It orchestrates the generation process.
*   **`__init__(config: GeneratorConfig)`**: Initializes with a configuration object.
*   **`generate_initial_structures()`**: Creates random bulk, surface, and cluster structures using `ase.build`.
*   **`generate_candidates(parent: Structure, policy: Policy) -> list[Structure]`**: Generates perturbations (displacements, strains, defects) based on the policy.

#### `AdaptiveExplorationPolicy`
A stateless logic engine that determines *how* to sample.
*   **`decide_strategy(features: MaterialFeatures) -> ExplorationStrategy`**:
    *   Input: `MaterialFeatures` (e.g., composition, estimated $T_m$).
    *   Output: `ExplorationStrategy` (e.g., `temperature_max`, `mc_ratio`, `strain_range`).
    *   Logic: Implements the decision tree defined in the spec (e.g., "If metal, increase MC ratio").

#### `defect_generator.py`
Helper functions to introduce defects.
*   **`create_vacancy(structure: Structure, concentration: float)`**
*   **`create_interstitial(structure: Structure, element: str)`**
*   **`apply_strain(structure: Structure, strain_tensor: list[float])`**

### 3.2 Domain Models

*   **`Structure`**: Extended to include `provenance` metadata (e.g., `provenance="random_strain"`, `parent_id="..."`).
*   **`MaterialFeatures`**: Pydantic model for input features.
*   **`ExplorationStrategy`**: Pydantic model for output strategy.

## 4. Implementation Approach

1.  **ASE Integration**: Implement `structure_utils.py` to wrap common ASE functions (building bulk, surfaces, clusters) with type safety.
2.  **Policies**: Implement the logic in `policies.py`. Start with simple rules (e.g., linear temperature ramp).
3.  **Defect Generation**: Implement `defect_generator.py` using `pymatgen` or pure `ase` logic. Ensure correct handling of periodic boundary conditions.
4.  **Structure Generator**: Implement `structure_generator.py`. Connect the policy engine to the generation logic.
5.  **Configuration**: Update `config.py` with `GeneratorConfig` (e.g., `initial_count`, `composition`).
6.  **Factory**: Register `StructureGenerator` in `ComponentFactory`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **`test_policies.py`**:
    *   Mock `MaterialFeatures` input.
    *   Verify that `decide_strategy` returns the expected parameters (e.g., high $T$ for high $T_m$).
*   **`test_defect_generator.py`**:
    *   Create a perfect crystal.
    *   Apply vacancy generation.
    *   Assert that the number of atoms decreases by the expected amount.
    *   Assert that cell parameters remain unchanged (unless relaxation is simulated).

### 5.2 Integration Testing
*   **Structure Validity**:
    *   Generate a batch of 100 structures.
    *   Verify that all are valid `ase.Atoms` objects.
    *   Check for minimum interatomic distances (no fusion).
    *   Verify that `provenance` metadata is correctly attached.
