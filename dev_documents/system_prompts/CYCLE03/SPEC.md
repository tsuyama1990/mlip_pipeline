# Cycle 03 Specification: The Structure Generator (Exploration Policies)

## 1. Summary

Cycle 03 focuses on the **Structure Generator**, the component responsible for proposing new atomic configurations to be labeled. A smart generator is crucial for Active Learning; simply generating random structures leads to inefficient sampling of high-energy states that are irrelevant for actual simulations.

We will implement an **Adaptive Exploration Policy**. Instead of hardcoding "run MD at 300K," the generator will select strategies based on the current state of the project. For this cycle, we implement the foundational strategies: **Initial Random Exploration** (for cold starts) and **Distortion Engineering** (Rattling/Strain) to probe local energy landscapes.

## 2. System Architecture

Files to create/modify are **bolded**.

```ascii
src/mlip_autopipec/
├── config/
│   └── config_model.py         # Update ExplorerConfig
├── infrastructure/
│   └── explorer/
│       ├── **__init__.py**
│       ├── **structure_generator.py** # Main class
│       └── **policies.py**            # Exploration strategies
```

## 3. Design Architecture

### 3.1. Explorer Configuration
We extend `ExplorerConfig` in `config_model.py`:
*   `initial_exploration_type`: Literal["random", "m3gnet"] (default "random")
*   `policy_map`: Dict[str, Any] (Configuration for different policies)

### 3.2. StructureGenerator Logic
*   **Inheritance**: Inherits from `BaseExplorer` (conceptually, though `BaseExplorer` in Cycle 01 was for Dynamics. We might need to split `Explorer` into `GlobalExplorer` (Generator) and `LocalExplorer` (Dynamics). For now, we treat `StructureGenerator` as a distinct entity used by the Orchestrator).
*   **Method**: `generate_candidates(seed_structure: Atoms, count: int, strategy: str) -> List[Atoms]`

### 3.3. Policies
*   **RandomSubstitution**: Takes a prototype (e.g., FCC) and randomly assigns species. Used for "Cold Start".
*   **Rattling**: Applies random Gaussian noise to atomic positions. $\vec{r}_i' = \vec{r}_i + \mathcal{N}(0, \sigma)$. Good for sampling local vibrational modes.
*   **VolumeScaling**: Applies isotropic strain. $L' = (1 + \epsilon)L$. Good for EOS (Equation of State) learning.

## 4. Implementation Approach

1.  **Refactor Interfaces**: Ensure `StructureGenerator` is clearly defined in `interfaces/explorer.py`.
2.  **Initial Exploration**: Implement `policies.RandomSubstitutionPolicy`.
    *   Input: Composition (e.g., "Fe2Pt2"), Spacegroup/Prototype.
    *   Output: `Atoms` object.
3.  **Distortion**: Implement `policies.RattlingPolicy` and `policies.StrainPolicy`.
    *   Use `ase.constraints` or simple numpy operations to modify positions/cell.
4.  **Integration**: Wire `StructureGenerator` into the `Orchestrator`'s "Exploration" phase (specifically for the initial step or when Dynamics is not yet ready).

## 5. Test Strategy

### 5.1. Unit Testing
*   **Rattling**: Take a perfect crystal. Apply rattling. Assert positions have changed but composition is same. Assert minimum distance is not violated (atoms didn't fuse).
*   **Strain**: Apply +10% volume strain. Assert cell vectors are larger.

### 5.2. Integration Testing
*   **Cold Start**: Run the Orchestrator with an empty dataset.
    *   Assert `StructureGenerator` is called to create the initial batch of N structures.
    *   Assert these structures are passed to the Oracle.
