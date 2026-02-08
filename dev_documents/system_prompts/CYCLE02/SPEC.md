# Cycle 02: Structure Generator

## 1. Summary

In this cycle, we replace the `MockGenerator` with a fully functional **Structure Generator** capable of creating physically meaningful atomic configurations. This module is the "Explorer" of the active learning loop, responsible for proposing diverse structures that cover the relevant phase space (temperature, pressure, composition, defects).

A key innovation is the **Adaptive Exploration Policy**. Instead of a static strategy (e.g., "always generate random rattled structures"), the generator will inspect the current state of the potential (e.g., training error, uncertainty distribution) to decide *what* to generate next. For example, if the potential is accurate for bulk but fails for surfaces, the policy will prioritize surface generation.

We will leverage `ase.build` for standard crystallographic operations (bulk, surface, nanotube) and `pymatgen` for advanced symmetry analysis and defect generation if needed.

## 2. System Architecture

Files in **bold** are new or modified.

```ascii
src/mlip_autopipec/
├── components/
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── **policy.py**         # Adaptive Policy Logic
│   │   ├── **builder.py**        # Structure Builders (Bulk, Surface, Cluster)
│   │   └── **rattle.py**         # Displacement/Strain logic
│   └── ...
```

## 3. Design Architecture

### 3.1. Adaptive Exploration Policy (`policy.py`)
-   **Class `ExplorationPolicy`**:
    -   `decide_next_batch(current_metrics: Dict) -> List[GenerationTask]`
    -   Logic:
        -   If `cycle == 0` (Cold Start): Return tasks for random bulk + EOS (Equation of State) scan.
        -   If `validation_error.surface > threshold`: Prioritize `SurfaceBuilder`.
        -   If `uncertainty.high`: Prioritize `RattleBuilder` around high-uncertainty configurations.

### 3.2. Structure Builders (`builder.py`)
-   **`StructureBuilder` (Abstract Base)**:
    -   `build(config: Dict) -> List[Structure]`
-   **`BulkBuilder`**: Generates supercells with random strains.
-   **`SurfaceBuilder`**: Cleaves surfaces ((100), (110), (111)) and adds vacuum.
-   **`DefectBuilder`**: Introduces vacancies or interstitials.
-   **`ClusterBuilder`**: Cuts clusters from bulk or generates nanoparticles.

### 3.3. Rattle & Strain (`rattle.py`)
-   **`RattleTransform`**: Applies random Gaussian noise to atomic positions ($\sigma = 0.01 - 0.1 \AA$).
-   **`StrainTransform`**: Applies a deformation gradient tensor $F$ to the cell and positions.

## 4. Implementation Approach

1.  **Refactor**: Ensure `Structure` domain model supports all necessary attributes (tags for "bulk", "surface").
2.  **Builders**: Implement `BulkBuilder` using `ase.build.bulk`. Add method to apply random strain.
3.  **Surfaces**: Implement `SurfaceBuilder` using `ase.build.surface`.
4.  **Policy**: Implement a simple rule-based policy first.
    -   `Cycle 0`: 50% Bulk (strained), 30% Surface, 20% Clusters.
    -   `Cycle > 0`: 40% based on high-uncertainty seeds (if available), 60% exploration.
5.  **Integration**: Update `Orchestrator` to use the new `StructureGenerator` (which wraps the Policy and Builders).

## 5. Test Strategy

### 5.1. Unit Tests
-   **Builder Tests**: Verify `BulkBuilder` returns correct crystal structure (e.g., FCC for Al). Verify `SurfaceBuilder` adds vacuum (check cell dimensions).
-   **Transformation Tests**: Verify `StrainTransform` changes the cell volume correctly. Verify `RattleTransform` moves atoms (assert positions != original).
-   **Policy Tests**: Mock the `current_metrics` input and assert the `ExplorationPolicy` returns the expected distribution of tasks (e.g., "return Surface task if error is high").

### 5.2. Integration Tests
-   **Generator Integration**: Run `StructureGenerator.generate(n=100)`. Check that the output list contains a mix of structures with correct tags.
-   **Visual Inspection (UAT)**: Use `ase.visualize` to manually inspect a few generated files in the UAT notebook.

```python
# Example Test for Builder
def test_bulk_builder():
    builder = BulkBuilder(element='Fe', crystal_structure='bcc')
    structures = builder.build(n=5)
    assert len(structures) == 5
    assert structures[0].info['type'] == 'bulk'
    assert len(structures[0]) > 0
```
