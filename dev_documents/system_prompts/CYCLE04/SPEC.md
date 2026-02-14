# Cycle 04 Specification: Structure Generator & Adaptive Policy

## 1. Summary
Cycle 04 focuses on the "Structure Generator", the component responsible for exploring the configuration space and proposing new candidates for the active learning loop. Instead of relying solely on random sampling, this module implements an "Adaptive Exploration Policy" that intelligently selects sampling strategies based on the current state of knowledge.

Key features:
1.  **Adaptive Policy Engine**: A decision-making module that analyzes input features (e.g., composition, current uncertainty distribution) and selects the optimal exploration strategy (e.g., "High-Temperature MD", "Defect Sampling", "Cold Start").
2.  **Diverse Sampling Strategies**: Implementation of multiple strategies:
    -   **Cold Start (M3GNet)**: Uses a universal potential to find approximate ground states for new materials.
    -   **Random Perturbation**: Applies random thermal displacements (rattling) and strain to existing structures.
    -   **Defect Engineering**: Systematically introduces vacancies, interstitials, and antisite defects.
3.  **Atomic Mutations**: A library of robust geometric operations (strain, rattle, swap) applied to `ase.Atoms` objects.

## 2. System Architecture

The file structure expands `src/pyacemaker/generator`. **Bold files** are new.

```text
src/
└── pyacemaker/
    ├── core/
    │   └── **config.py**       # Updated with GeneratorConfig
    └── **generator/**
        ├── **__init__.py**
        ├── **policy.py**       # Decision Logic
        ├── **strategies.py**   # Sampling Strategies (Strategy Pattern)
        └── **mutations.py**    # Low-level atomic operations
```

### File Details
-   `src/pyacemaker/generator/policy.py`: Contains `AdaptivePolicy` class. It decides *which* strategy to use.
-   `src/pyacemaker/generator/strategies.py`: Contains `ExplorationStrategy` ABC and concrete implementations (`M3GNetStrategy`, `RandomStrategy`, `DefectStrategy`).
-   `src/pyacemaker/generator/mutations.py`: Helper functions to modify `ase.Atoms` (e.g., `apply_strain`, `rattle_atoms`, `create_vacancy`).
-   `src/pyacemaker/core/config.py`: Expanded to include `GeneratorConfig` (e.g., defect concentration, strain range).

## 3. Design Architecture

### 3.1. Generator Configuration
```python
class GeneratorConfig(BaseModel):
    initial_exploration: str = "m3gnet" # or "random"
    strain_range: float = 0.15
    rattle_amplitude: float = 0.1
    defect_density: float = 0.01
```

### 3.2. Strategy Pattern
We use the Strategy Pattern to decouple the *decision* of what to sample from the *implementation* of sampling.

```python
class ExplorationStrategy(ABC):
    @abstractmethod
    def generate(self, seed: Atoms, n_candidates: int) -> List[Atoms]:
        pass

class RandomStrategy(ExplorationStrategy):
    def generate(self, seed: Atoms, n_candidates: int) -> List[Atoms]:
        # Apply strain and rattle
        pass

class DefectStrategy(ExplorationStrategy):
    def generate(self, seed: Atoms, n_candidates: int) -> List[Atoms]:
        # Create supercell and introduce defects
        pass
```

### 3.3. Adaptive Policy
```python
class AdaptivePolicy:
    def decide_strategy(self, context: ExplorationContext) -> ExplorationStrategy:
        if context.is_cold_start:
            return M3GNetStrategy()
        if context.uncertainty > high_threshold:
            return CautiousStrategy() # Sample near known data
        return AggressiveStrategy() # Explore new regions (High T, Defects)
```

## 4. Implementation Approach

### Step 1: Update Configuration
-   Modify `src/pyacemaker/core/config.py` to add `GeneratorConfig`.

### Step 2: Atomic Mutations
-   Implement `src/pyacemaker/generator/mutations.py`.
-   `apply_strain(atoms, strain_tensor)`: Deforms the simulation box and atomic positions.
-   `rattle_atoms(atoms, stdev)`: Adds Gaussian noise to positions.
-   `create_vacancy(atoms, index)`: Removes an atom.

### Step 3: Strategies
-   Implement `src/pyacemaker/generator/strategies.py`.
-   **RandomStrategy**: Simple loop calling mutations.
-   **DefectStrategy**: Create supercell ($2 \times 2 \times 2$ or similar) -> Remove/Add atoms.
-   **M3GNetStrategy**: (Optional/Mockable) If `m3gnet` is installed, run relaxation. If not, fallback to Random or return input.

### Step 4: Adaptive Policy
-   Implement `src/pyacemaker/generator/policy.py`.
-   Implement logic to switch between "Cold Start" (Cycle 0) and "Refinement" phases.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Mutations Testing (`tests/generator/test_mutations.py`)**:
    -   Create a unit cell. Apply 10% strain. Assert volume changed by ~33% (approx).
    -   Apply rattle. Assert positions changed but cell didn't.
-   **Strategy Testing (`tests/generator/test_strategies.py`)**:
    -   **RandomStrategy**: Verify it returns `n_candidates` distinct structures.
    -   **DefectStrategy**: Verify it returns structures with $N-1$ atoms (vacancy).

### 5.2. Integration Testing
-   **Policy Integration (`tests/generator/test_policy.py`)**:
    -   Simulate a "Cold Start" context. Assert `M3GNetStrategy` is chosen.
    -   Simulate a "High Uncertainty" context. Assert appropriate strategy is chosen.
