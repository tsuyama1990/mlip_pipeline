# Cycle 02 Specification: Structure Generator & Adaptive Policy

## 1. Summary

Cycle 02 implements the **Structure Generator**, the component responsible for creating atomic configurations that will be fed into the active learning loop. This cycle moves beyond simple random sampling by introducing an **Adaptive Exploration Policy** engine. This engine intelligently decides *how* to sample the chemical and structural space based on the material's properties (e.g., band gap, bulk modulus) and the current uncertainty state.

We also implement the "Cold Start" capability using **M3GNet**, allowing the system to rapidly guess stable crystal structures without expensive DFT calculations, providing a high-quality initial population for the first generation.

## 2. System Architecture

### File Structure

Files in **bold** are new or modified in this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── generator/
│   │   ├── **__init__.py**
│   │   ├── **base.py**          # BaseGenerator class
│   │   ├── **random.py**        # Random substitution/rattling
│   │   ├── **m3gnet.py**        # M3GNet wrapper for cold start
│   │   ├── **defects.py**       # Point defect generator
│   │   └── **policy.py**        # Adaptive Policy Logic
│   └── mock.py
├── domain_models/
│   ├── **structures.py**        # Structure Pydantic models
│   └── config.py                # Updated with GeneratorConfig
└── core/
    └── orchestrator.py          # Updated to use Generator
```

### Component Interaction
1.  **Orchestrator** calls `generator.generate(n_structures, context)`.
2.  **Generator** consults **AdaptivePolicy (`policy.py`)** to determine the strategy (e.g., "70% Random, 30% Defect" or "Use M3GNet").
3.  **AdaptivePolicy** returns a list of strategies.
4.  **Sub-Generators (`random.py`, `defects.py`)** execute the strategies using `ASE` to manipulate atoms.
5.  **Generator** returns a list of `Structure` objects (Pydantic models) to the Orchestrator.

## 3. Design Architecture

### 3.1. Structure Models (`domain_models/structures.py`)

We need a robust data transfer object (DTO) for atomic structures that wraps ASE's `Atoms` object but adds metadata.

```python
class Structure(BaseModel):
    atoms: Any  # ASE Atoms object (custom validator needed)
    provenance: str  # e.g., "m3gnet_relax", "random_rattle"
    features: Dict[str, float] = {}  # e.g., {"energy": -10.5, "uncertainty": 0.1}

    class Config:
        arbitrary_types_allowed = True
```

### 3.2. Generator Configuration (`domain_models/config.py`)

Update `GeneratorConfig` to support strategies.

```python
class GeneratorConfig(BaseComponentConfig):
    strategy: Literal["random", "adaptive", "m3gnet"]
    supercell_matrix: List[int] = [2, 2, 2]
    rattle_amplitude: float = 0.1
```

### 3.3. Adaptive Policy Logic (`components/generator/policy.py`)

The policy engine is a decision tree.

```python
class ExplorationPolicy:
    def decide(self, current_generation: int, metrics: dict) -> List[dict]:
        if current_generation == 0:
            return [{"method": "m3gnet", "count": 10}]

        # If uncertainty is high in bulk, focus on bulk distortions
        if metrics.get("max_uncertainty") > 1.0:
            return [{"method": "random_rattle", "amplitude": 0.2, "count": 20}]

        # Default: Mix of defects and strains
        return [
            {"method": "defects", "type": "vacancy", "count": 5},
            {"method": "strain", "range": 0.05, "count": 5}
        ]
```

## 4. Implementation Approach

1.  **Define Structure Model**: Implement `domain_models/structures.py`. Ensure serialization of ASE atoms (e.g., to/from dict or JSON).
2.  **Implement Base Generator**: Create the interface in `components/generator/base.py`.
3.  **Implement Random Generator**: Use `ase.build.bulk` and `atoms.rattle()` in `components/generator/random.py`.
4.  **Implement Defect Generator**: Write simple logic to remove random atoms (Vacancy) or swap species (Antisite) in `components/generator/defects.py`.
5.  **Implement M3GNet Generator**:
    -   Optional dependency: Check if `matgl` is installed.
    -   If yes, load model and relax structure.
    -   If no, fallback to random or warn user.
6.  **Implement Policy Engine**: Write the decision logic in `components/generator/policy.py`.
7.  **Orchestrator Integration**: Update `Orchestrator` to initialize the generator and call `generate()` in the exploration phase.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Random Generator**: Verify it produces `n` structures and they are *different* from the input (positions changed).
-   **Defect Generator**: Verify it returns a structure with `N-1` atoms for vacancy mode.
-   **Policy**: Feed mock metrics and verify it returns the correct strategy distribution.

### 5.2. Integration Testing
-   **M3GNet Fallback**: Run in an environment without `matgl` and verify it degrades gracefully (doesn't crash).
-   **Orchestrator Loop**: Verify Orchestrator can call generator and receive a list of `Structure` objects.
