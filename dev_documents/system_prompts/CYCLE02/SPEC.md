# Cycle 02 Specification: Intelligent Structure Generation

## 1. Summary

In Cycle 02, we replace the `MockExplorer` with a fully functional `StructureGenerator`. The goal is to move from "random atoms" to "physically relevant candidate structures." We introduce the **Adaptive Exploration Policy**, a decision-making engine that determines *what* kind of data is needed next.

Instead of a hardcoded sampling strategy, the system extracts "Material DNA" (features like band gap, bulk modulus, current uncertainty distribution) from the current state and dynamically adjusts exploration parameters (Temperature, Pressure, Strain, Defect Density). This ensures that we sample the configuration space efficiently, focusing on regions where the potential is weak or where interesting physics (e.g., phase transitions) occurs.

## 2. System Architecture

New files to be created are **bold**.

```ascii
src/mlip_autopipec/
├── ...
├── structure_generation/
│   ├── __init__.py
│   ├── **generator.py**     # Main StructureGenerator class
│   ├── **policies.py**      # Exploration Policies
│   ├── **mutators.py**      # Atomic manipulation logic (Strain, Rattle)
│   └── **features.py**      # Material DNA extraction
└── ...
```

## 3. Design Architecture

### 3.1. Exploration Policy (`policies.py`)
The `AdaptiveExplorationPolicy` class is the brain of the Explorer.
*   **Input**: `ExplorationState` (current cycle, max uncertainty observed, estimated material properties).
*   **Logic**: A set of rules (or a simple decision tree).
    *   *Example*: If `max_uncertainty` is low (< 2.0) AND `cycle` < 5 -> Trigger `StrainHeavyPolicy` to stiffen the potential.
    *   *Example*: If `material_type` is "Metal" -> Prioritize `MeltingPolicy` (high temperature MD).
*   **Output**: `ExplorationStrategy` object containing parameters:
    *   `md_temperature_range`: (300K, 2000K)
    *   `strain_range`: (-0.1, 0.1)
    *   `defect_density`: 0.01

### 3.2. Structure Generator (`generator.py`)
The executor class that takes the `ExplorationStrategy` and produces `ase.Atoms`.
*   **Method**: `generate(strategy: ExplorationStrategy, seed_structure: Atoms) -> List[Atoms]`
*   **Mutators**:
    *   `apply_strain(atoms, tensor)`: Deforms the simulation box.
    *   `rattle(atoms, stdev)`: Randomly displaces atoms (mimicking thermal noise).
    *   `introduce_vacancy(atoms, count)`: Removes random atoms.

### 3.3. Feature Extraction (`features.py`)
A utility to analyze the input structure and "guess" its nature (Material DNA).
*   Uses `pymatgen` or `ase` analysis tools.
*   Extracts: `formula`, `volume_per_atom`, `coordination_number`.
*   (Future: Could interface with M3GNet for zero-shot property prediction).

## 4. Implementation Approach

1.  **Develop Mutators**: Implement robust functions in `mutators.py` to manipulate `ase.Atoms`. Ensure they handle Periodic Boundary Conditions (PBC) correctly.
2.  **Implement Features**: Create the `FeatureExtractor` to gather metadata from the seed structure.
3.  **Design Policy Engine**: Implement `AdaptiveExplorationPolicy` in `policies.py`. Start with a simple rule-based system (Rule Engine).
4.  **Connect Generator**: Implement `StructureGenerator.generate_candidates()` to call the appropriate mutators based on the policy output.
5.  **Orchestrator Update**: Update `orchestrator.py` to inject the `StructureGenerator` instead of `MockExplorer`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Mutator Test**: Apply 10% strain to a unit cube. Verify the new cell vectors are exactly 1.1x length.
*   **Policy Test**: Feed a "High Uncertainty" state to the policy. Verify it returns a "Cautious" strategy (lower temperature, smaller steps).

### 5.2. Integration Testing
*   **Generation Loop**: Run the generator with a seed crystal (e.g., Al FCC). Verify it produces 50 varied structures (some strained, some rattled).
*   **Physics Check**: Calculate the radial distribution function (RDF) of the generated "liquid" structures to ensure they are not just exploded gas (atoms should still be close).
