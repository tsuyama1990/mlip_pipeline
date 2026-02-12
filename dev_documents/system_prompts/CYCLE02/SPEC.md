# Cycle 02 Specification: Structure Generator (Adaptive Exploration)

## 1. Summary

Cycle 02 focuses on implementing the **Structure Generator** module, the component responsible for exploring the vast chemical and structural space of the material system. The primary goal is to move beyond simple random sampling and implement an **Adaptive Exploration Policy**. This policy intelligently decides *how* to sample the potential energy surface (PES) based on the material's characteristics (e.g., metal vs. insulator) and the current state of knowledge (uncertainty distribution).

We will implement three core exploration strategies:
1.  **RandomGenerator**: A baseline strategy that generates random structures by perturbing atomic positions and lattice vectors. Useful for initial "Cold Start".
2.  **M3GNetGenerator** (Optional/Mock-able): Uses a pre-trained universal potential (M3GNet) to quickly screen stable structures and provide initial guesses.
3.  **AdaptiveGenerator**: The main engine. It dynamically selects between Molecular Dynamics (MD) at various temperatures and Monte Carlo (MC) atom swaps based on a policy configuration.

This cycle also introduces the interface to **LAMMPS** for generating the necessary input scripts to run these explorations, although the full-scale MD execution will be refined in Cycle 05.

## 2. System Architecture

Files in **bold** are to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── **config.py**         # Update GeneratorConfig
├── generator/
│   ├── **__init__.py**
│   ├── **interface.py**      # Enhanced BaseGenerator
│   ├── **random_gen.py**     # Random Structure Generation
│   ├── **m3gnet_gen.py**     # Pre-trained MLIP Wrapper
│   └── **adaptive.py**       # Adaptive Policy Logic
└── tests/
    └── unit/
        └── **test_generator.py**
```

## 3. Design Architecture

### 3.1 Adaptive Exploration Policy (`generator/adaptive.py`)

The `AdaptiveGenerator` class implements the logic to choose the best sampling strategy.
*   **Input**: `ExplorationContext` (current cycle number, previous generation's metrics).
*   **Policy Logic**:
    *   **Temperature Scheduling**: Starts with high temperatures to overcome barriers, then cools down (Simulated Annealing) as the potential matures.
    *   **MD/MC Ratio**: For multi-component systems (e.g., alloys), it mixes MC swap steps with MD steps to accelerate chemical ordering.
    *   **Defect Injection**: Introduces vacancies or interstitials if the uncertainty is low in bulk regions but high in defect regions.

### 3.2 Generator Configuration (`domain_models/config.py`)

We expand `GeneratorConfig` to support:
```python
class ExplorationPolicyConfig(BaseModel):
    strategy: Literal["random", "m3gnet", "adaptive"]
    temperature_schedule: List[float]  # [300, 600, 1200, ...]
    md_steps: int = 1000
    mc_swap_prob: float = 0.1
    defect_density: float = 0.0
```

### 3.3 Integration with Orchestrator

The Orchestrator will now instantiate the specific generator class based on the config.
`generator.explore(context)` will return a list of `Structure` objects (wrappers around `ase.Atoms`) with metadata indicating their provenance (e.g., `provenance="md_300K"`).

## 4. Implementation Approach

1.  **Enhance Domain Models**: Update `GeneratorConfig` in `domain_models/config.py`.
2.  **Implement RandomGenerator**: Create `generator/random_gen.py`. It should take a seed structure and apply random strain and atomic displacements (Rattling).
3.  **Implement M3GNetGenerator**: Create `generator/m3gnet_gen.py`. If `m3gnet` is not installed, it should fallback to a warning or mock behavior.
4.  **Implement AdaptiveGenerator**: Create `generator/adaptive.py`. Implement the logic to read the `cycle` number and select the temperature/pressure from the schedule.
5.  **LAMMPS Script Generation**: The generator needs to create simple LAMMPS input strings for MD exploration. We will implement a helper method `_generate_lammps_input(temp, steps)` within `AdaptiveGenerator`.
6.  **Unit Tests**: Verify that each generator produces valid `ase.Atoms` objects with correct tags.

## 5. Test Strategy

### 5.1 Unit Testing
*   **RandomGenerator**: Verify that output structures have perturbed positions compared to input.
*   **AdaptiveGenerator**:
    *   Test that it picks the correct temperature for a given cycle (e.g., Cycle 0 -> 300K, Cycle 1 -> 600K).
    *   Test that it correctly constructs the LAMMPS command string (or ASE calculator setup).

### 5.2 Integration Testing
*   **Orchestrator Integration**: Run a cycle where the Orchestrator calls `AdaptiveGenerator`. Verify that the returned structures are passed to the (Mock) Oracle.
