# Cycle 08 Specification: Expansion (kMC & Scale-up)

## 1. Summary

Cycle 08 is the final expansion phase. It extends the system's capabilities beyond MD to include **Adaptive Kinetic Monte Carlo (aKMC)** via the EON software. This allows the system to explore long-timescale phenomena (diffusion, rare events) that MD cannot reach. Additionally, this cycle focuses on full system integration, polishing the codebase, and ensuring the transition from local active learning to large-scale production runs.

## 2. System Architecture

### File Structure

Files to be created/modified are **bold**.

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── config.py                     # Update: Add KMCConfig
├── **inference/**
│   └── **eon.py**                    # EON Wrapper
└── orchestration/
    └── phases/
        └── **dynamics.py**           # Update: Add kMC support
```

## 3. Design Architecture

### Domain Models

#### `config.py`
*   **`KMCConfig`**:
    *   `temperature`: float
    *   `prefactor`: float
    *   `search_method`: Enum (dimer, neb)

### Components (`inference/eon.py`)

#### `EONWrapper`
*   **`run_kmc(structure, potential, config)`**:
    *   Sets up EON directory structure (`config.ini`, `reactant.con`).
    *   Generates a driver script (`pace_driver.py`) that EON calls to get Energy/Forces from the ACE potential.
    *   **Crucial**: The driver script must perform the "Gamma Check" and exit with a specific code if uncertainty is high.
    *   Executes `eonclient`.

### Orchestration (`phases/dynamics.py`)

#### `DynamicsPhase` (Updated)
*   Adds logic to choose between `LammpsRunner` (MD) and `EONWrapper` (kMC) based on the `ExplorationTask` or Config.
*   Handles "Halt" signals from EON (via the specific exit code).

## 4. Implementation Approach

1.  **Implement EON Wrapper**:
    *   Create `inference/eon.py`.
    *   Implement `pace_driver.py` generation (this script sits inside the EON run directory).
2.  **Update Dynamics Phase**:
    *   Refactor to support pluggable engines.
3.  **Full Integration**:
    *   Ensure the loop works: EON -> Halt -> Extract -> Oracle -> Train -> EON.

## 5. Test Strategy

### Unit Testing
*   **`test_eon_wrapper.py`**:
    *   Verify `config.ini` generation.
    *   Verify `pace_driver.py` is correctly written and executable.

### Integration Testing
*   **`test_kmc_loop.py`**:
    *   Mock `eonclient` execution.
    *   Simulate a halt event from EON.
    *   Verify the orchestrator catches it and extracts the structure.
