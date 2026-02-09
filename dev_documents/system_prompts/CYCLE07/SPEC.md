# Cycle 07 Specification: Advanced Dynamics (kMC & Scale-up)

## 1. Summary

Cycle 07 addresses the "Time-Scale Problem". Standard MD (Cycle 05) is limited to nanoseconds, but many material phenomena (diffusion, phase transitions, ordering) happen over seconds or hours. To bridge this gap, we integrate **EON**, a software package for Adaptive Kinetic Monte Carlo (aKMC).

The key challenge here is interoperability. EON is a C++ code that expects a specific interface to calculate energies and forces. We must implement a "Bridge Script" (`pace_driver.py`) that allows EON to call our Python-based Pacemaker calculator. Furthermore, we extend the "On-the-Fly" (OTF) logic to kMC: if EON explores a high-energy transition state (Saddle Point) that is uncertain, we must halt it just like we halted MD.

By the end of this cycle, the Orchestrator will be able to perform long-timescale simulations, discovering reaction pathways and ordering events that MD would never find.

## 2. System Architecture

This cycle focuses on the `components/dynamics` package and external interfaces.

### File Structure
Files to be created/modified in this cycle are marked in **bold**.

```
src/mlip_autopipec/
├── components/
│   ├── dynamics/
│   │   ├── **eon_driver.py**       # Interface to EON
│   │   └── **bridge/**             # Directory for external scripts
│   │       └── **pace_driver.py**  # The script called by EON
│   └── factory.py                  # Register EONDynamics
├── domain_models/
│   └── **config.py**               # Add EONConfig
└── tests/
    └── **test_eon.py**
```

## 3. Design Architecture

### 3.1. EON Configuration (`domain_models/config.py`)
Update `DynamicsConfig` (or `EONConfig`):
*   `type`: "eon".
*   `temperature`: float.
*   `process_search_method`: "dimer" or "neb".
*   `event_table_size`: int.

### 3.2. EON Driver (`components/dynamics/eon_driver.py`)
*   `run_exploration()`:
    1.  Create `config.ini` for EON.
    2.  Place `reactant.con` (initial structure).
    3.  Copy `pace_driver.py` and `potential.yace` to the working directory.
    4.  Run `eonclient`.
    5.  Parse results (`results.dat`, `processes/`).

### 3.3. The Bridge Script (`components/dynamics/bridge/pace_driver.py`)
This is a standalone script that runs *inside* the EON process environment.
*   **Input**: Cartesian coordinates from EON (via stdin or file).
*   **Output**: Energy and Forces to EON (via stdout).
*   **OTF Logic**:
    *   Calculate $\gamma$ (extrapolation grade).
    *   If $\gamma > \text{threshold}$:
        *   Write `bad_structure.con`.
        *   Exit with a special error code (e.g., 100) to signal the Orchestrator.

## 4. Implementation Approach

1.  **Implement `pace_driver.py`**:
    *   Use `pyace` or `pacemaker` python bindings to load the potential.
    *   Implement the simple I/O protocol required by EON (often just `energy` followed by `force_x force_y force_z` per line).
2.  **Implement `EONDynamics`**:
    *   Generate the `config.ini` using a template.
    *   Handle the execution of `eonclient`.
    *   Catch the special exit code (100) from the bridge script to trigger the Halt logic (reusing Cycle 06 handler).
3.  **Mocking**:
    *   EON is complex to install. For CI, we will mock `eonclient`.
    *   The mock will read `reactant.con`, "wait" a bit, and write a `product.con` (random perturbation).

## 5. Test Strategy

### 5.1. Unit Testing
*   **Bridge Script**:
    *   Input: A dummy structure file.
    *   Action: Run `python pace_driver.py < input`.
    *   Assert: Output format matches EON requirements.

### 5.2. Integration Testing
*   **EON Execution (Mocked)**:
    *   Input: Structure, Potential.
    *   Action: `dynamics.explore()`.
    *   Assert: `config.ini` is created.
    *   Assert: `eonclient` was called.
    *   Assert: Returns the evolved structure (or halted state).
