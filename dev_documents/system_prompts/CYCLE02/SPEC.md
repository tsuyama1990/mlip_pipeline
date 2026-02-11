# Cycle 02 Specification: Structure Generator (Exploration)

## 1. Summary

Cycle 02 implements the **Structure Generator** module, which is responsible for exploring the chemical and structural space to propose candidate structures for training. The core innovation here is the **Adaptive Exploration Policy**, which moves away from static, rule-based sampling to a dynamic strategy that adapts to the material's properties (e.g., using different sampling methods for metals vs. insulators).

This cycle focuses on implementing the `BaseGenerator` interface and several concrete implementations: `RandomGenerator` (for baseline noise), `M3GNetGenerator` (for "Cold Start" using pre-trained universal potentials), and the `AdaptiveGenerator` which acts as a meta-generator orchestrating the others based on a policy.

## 2. System Architecture

We expand the `components/generator` module and introduce new configuration models.

### File Structure
**bold** indicates files to be created or modified in this cycle.

```
.
├── src/
│   └── mlip_autopipec/
│       ├── domain_models/
│       │   ├── **config.py**      # Add GeneratorConfig, AdaptivePolicyConfig
│       │   └── **enums.py**       # Add GeneratorType (RANDOM, M3GNet, ADAPTIVE)
│       └── components/
│           ├── **generator/**
│           │   ├── **__init__.py**
│           │   ├── **base.py**        # BaseGenerator (Abstract)
│           │   ├── **random.py**      # Random perturbation/strain
│           │   ├── **m3gnet.py**      # M3GNet wrapper (Universal Potential)
│           │   ├── **adaptive.py**    # The Policy Engine
│           │   └── **utils.py**       # Symmetry/Cell manipulation helpers
│           └── **factory.py**         # Component Factory (updates)
```

### Key Components
1.  **BaseGenerator (`src/mlip_autopipec/components/generator/base.py`)**: Defines the contract `generate(n_structures: int) -> List[Structure]`.
2.  **RandomGenerator**: Applies random strains to the unit cell and random displacements to atomic positions. Useful for sampling the local potential energy surface (PES).
3.  **M3GNetGenerator**: Utilizes the `matgl` library (M3GNet) to run short MD trajectories or relaxations. This provides reasonable initial structures ("Cold Start") when no MLIP exists yet.
4.  **AdaptiveGenerator**: The intelligent agent. It holds a reference to other generators and selects which one to use based on the `AdaptivePolicyConfig` and the current iteration state.

## 3. Design Architecture

### 3.1. Domain Models
*   **AdaptivePolicyConfig**:
    *   `initial_exploration_ratio`: Percentage of structures from M3GNet in early cycles.
    *   `random_perturbation_scale`: Magnitude of displacements (e.g., 0.1 Angstrom).
    *   `strain_range`: Max strain percentage (e.g., 0.05 for 5%).
*   **GeneratorConfig**: A discriminated union supporting `type="random"`, `type="m3gnet"`, `type="adaptive"`.

### 3.2. Adaptive Policy Logic
The `AdaptiveGenerator` uses a **Strategy Pattern**.
*   **Input**: Current cycle number, previous cycle metrics (e.g., max uncertainty - though this might be mocked in Cycle 02).
*   **Logic**:
    *   **Cold Start (Cycle 0)**: Use `M3GNetGenerator` to get physically sensible structures (approximating the convex hull).
    *   **Exploration (Cycle 1-N)**: Mix `RandomGenerator` (to probe local curvature) and `MDGenerator` (placeholder for future cycles).
    *   **Ratio Control**: The policy determines the fraction of structures from each source.

### 3.3. Dependencies
*   **ASE (Atomic Simulation Environment)**: Used for all structure manipulations (strains, displacements).
*   **MatGL (M3GNet)**: Optional dependency. If not installed, `M3GNetGenerator` should raise a helpful error or fallback to Random.

## 4. Implementation Approach

1.  **Update Dependencies**: Add `matgl` and `ase` to `pyproject.toml`.
2.  **Domain Models**: Update `config.py` with the new generator configurations.
3.  **Base Class**: Implement `BaseGenerator` in `components/generator/base.py`.
4.  **Random Generator**: Implement `RandomGenerator`. Use `ase.calculators.calculator.Calculator` interface concepts but for generating structures (applying `rattle` and `strain`).
5.  **M3GNet Integration**: Implement `M3GNetGenerator`. It should load a pre-trained model and run a short MD/Relaxation. *Note: Ensure this is wrapped in a try-except block to handle missing dependencies.*
6.  **Adaptive Logic**: Implement `AdaptiveGenerator`. It should instantiate sub-generators.
7.  **Factory**: Update the component factory to initialize these new generators based on the config.

## 5. Test Strategy

### 5.1. Unit Testing
*   **RandomGenerator**:
    *   Test that `generate(n=10)` returns 10 structures.
    *   Test that applying strain changes the cell vectors.
    *   Test that applying rattle changes atomic positions.
*   **AdaptivePolicy**:
    *   Test that it correctly switches strategies based on the configuration (e.g., if `ratio_m3gnet=1.0`, it only calls M3GNet).

### 5.2. Integration Testing
*   **M3GNet (Mocked)**: Since `matgl` is heavy, we will mock the `matgl` library calls in tests to verify the wrapper logic without downloading models.
*   **Orchestrator Integration**: Update the `mock` loop test to use the `RandomGenerator` instead of the `MockGenerator`. Verify that "real" ASE atoms are being passed around (even if they are just random noise).
