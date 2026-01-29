# Cycle 07: Adaptive Strategy

## 1. Summary

Up to Cycle 06, our exploration strategy was likely simple: "Run MD at 300K". While this works for simple systems, it fails to explore the phase space efficiently for complex materials. For example, in an alloy, diffusion might be too slow at MD timescales to see segregation. In a hard ceramic, 300K MD will never sample high-energy distorted configurations.

Cycle 07 introduces the **Adaptive Exploration Policy**. This module acts as the "Strategist". It analyses the material's properties (or the uncertainty distribution of the previous generation) and dynamically decides *how* to explore.
-   **Metals**: Might require high-temperature MD or Monte Carlo swaps to explore chemical ordering.
-   **Insulators**: Might require explicit defect generation (vacancies) or large strain application to sample bond breaking.
-   **High Uncertainty**: If the model is clueless, the policy might prescribe a "Cautious" approach (lower temperature) to build a foundation.

We also implement advanced **Structure Generators** (Defects, Strain) that go beyond simple MD snapshots.

## 2. System Architecture

We expand the `physics/structure_gen` package.

### File Structure
Files to be created/modified are in **bold**.

```ascii
mlip_autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   └── config.py               # Update with PolicyConfig
│       ├── physics/
│       │   ├── structure_gen/
│       │   │   ├── **policy.py**       # The Decision Engine
│       │   │   ├── **defects.py**      # Point defect generator
│       │   │   └── **strain.py**       # Strain/Deformation generator
│       └── orchestration/
│           └── phases/
│               └── **exploration.py**  # Update to use Policy
└── tests/
    └── physics/
        └── structure_gen/
            └── **test_policy.py**
```

### Component Interaction

1.  **Orchestrator** (Exploration Phase) queries **`AdaptivePolicy`**.
2.  **`AdaptivePolicy`** analyzes the `Config` (Material type) and `WorkflowState` (Cycle count).
3.  **Decision**:
    -   If Cycle 0 (Cold Start): Return `RandomStrain` + `M3GNet` (if available).
    -   If Metal & Cycle > 0: Return `HybridMDMC` (MD + Monte Carlo).
    -   If Insulator: Return `DefectSampling`.
4.  **Orchestrator** executes the chosen strategy.
    -   If `DefectSampling`: Calls `DefectGenerator`.
    -   If `HybridMDMC`: Configures `LammpsRunner` with `fix atom/swap`.

## 3. Design Architecture

### 3.1. Policy Domain Model
-   **Class `ExplorationTask`**:
    -   `method`: `Literal["MD", "MC", "Minimization", "Static"]`.
    -   `temperature`: `float`.
    -   `pressure`: `float`.
    -   `steps`: `int`.
    -   `modifiers`: `List[str]` (e.g., "swap", "strain").

### 3.2. Defect Generator (`physics/structure_gen/defects.py`)
-   **Class `DefectStrategy`**:
    -   `apply(structure)`: Returns a list of structures.
    -   **Vacancy**: Remove 1 atom.
    -   **Interstitial**: Add 1 atom in a void (Voronoi analysis or random).
    -   **Antisite**: Swap species A and B.

### 3.3. Strain Generator (`physics/structure_gen/strain.py`)
-   **Class `StrainStrategy`**:
    -   `apply(structure)`: Returns deformed structures.
    -   **Uniaxial**: Stretch along x, y, z.
    -   **Shear**: Tilt the cell vectors.
    -   **Rattle**: Random noise (already in Cycle 02, but refined here).

## 4. Implementation Approach

### Step 1: Generators
-   Implement `defects.py`. Use `ase` manipulation.
-   Implement `strain.py`. Use `structure.cell *= (1 + epsilon)`.

### Step 2: Policy Engine
-   Implement `policy.py`.
-   Define a simple rule-based system first.
    ```python
    def get_strategy(cycle, config):
        if cycle == 0:
            return "RANDOM_STRAIN"
        if config.is_metal:
            return "MD_MC_HYBRID"
        return "MD_RAMP"
    ```

### Step 3: Orchestrator Integration
-   Modify `exploration.py` phase in the Orchestrator.
-   Instead of hardcoding "Run MD", ask the Policy what to do.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Defects**:
    -   Create a 2-atom cell. Apply Vacancy. Assert 1 atom remains.
    -   Apply Antisite on elemental crystal. Assert no change (or error).
-   **Strain**:
    -   Apply 10% strain. Check cell vectors.

### 5.2. Policy Logic Test
-   Mock the config as "Metal".
-   Call `policy.decide()`.
-   Assert it recommends Monte Carlo swapping.
