# Cycle 07 Specification: Adaptive Strategy

## 1. Summary

Cycle 07 introduces intelligence into the structure generation process. Instead of static random sampling, the **Adaptive Exploration Policy** dynamically determines the best sampling strategy based on the material's properties (e.g., band gap, bulk modulus) and the current state of the workflow (e.g., uncertainty distribution). This cycle implements the logic to switch between MD, MC (Monte Carlo), and varying temperature/pressure schedules to maximize chemical and structural diversity.

## 2. System Architecture

### File Structure

Files to be created/modified are **bold**.

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── config.py                     # Update: Add PolicyConfig
├── **modules/**
│   └── **structure_gen/**
│       ├── **policy.py**             # Adaptive Policy Engine
│       └── **strategies.py**         # Update: Add MD/MC hybrid strategies
└── orchestration/
    └── phases/
        └── **exploration.py**        # Update: Use Policy
```

## 3. Design Architecture

### Domain Models

#### `config.py`
*   **`PolicyConfig`**:
    *   `enable_adaptive`: bool
    *   `weights`: Dict (weights for different features)

### Components (`modules/structure_gen/policy.py`)

#### `AdaptivePolicy`
*   **`decide_next_step(workflow_state) -> ExplorationTask`**:
    *   Analyzes `workflow_state.meta` (e.g., "is_metal", "bulk_modulus").
    *   Returns an `ExplorationTask` object containing parameters:
        *   `method`: MD or MC
        *   `temperature`: max_T
        *   `pressure`: pressure_range
        *   `defects`: defect_density

### Components (`modules/structure_gen/strategies.py`)

#### `StrategyFactory`
*   Updates to support new strategies:
    *   **`HybridMDMC`**: Swaps atoms (MC) periodically during MD.
    *   **`DefectBuilder`**: Introduces vacancies/interstitials based on `defect_density`.

## 4. Implementation Approach

1.  **Implement Policy Engine**:
    *   Create a rule-based system (as defined in ALL_SPEC):
        *   If `Eg ~ 0` (Metal) -> High MC ratio.
        *   If `Eg > 0` (Insulator) -> Focus on Defects/Distortions.
        *   If `High Uncertainty` -> Lower Temperature.
2.  **Enhance Structure Generator**:
    *   Add methods to inject defects (vacancies, antisites).
    *   Add support for MC swap commands in LAMMPS (if running MD-based generation).
3.  **Update Exploration Phase**:
    *   Call `AdaptivePolicy.decide_next_step` before generating candidates.

## 5. Test Strategy

### Unit Testing
*   **`test_policy.py`**:
    *   Mock a workflow state with `Eg=0`. Verify Policy returns `method=MC`.
    *   Mock a workflow state with `Eg=5.0`. Verify Policy returns `defects=True`.

### Integration Testing
*   **`test_adaptive_exploration.py`**:
    *   Configure `enable_adaptive=True`.
    *   Run the phase.
    *   Verify that the generated candidates reflect the policy (e.g., contain vacancies if expected).
