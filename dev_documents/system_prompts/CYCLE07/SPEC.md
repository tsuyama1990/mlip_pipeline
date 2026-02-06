# Cycle 07 Specification: Advanced Exploration & Dynamics (kMC)

## 1. Summary

Cycle 07 expands the temporal reach of our system by integrating **Adaptive Kinetic Monte Carlo (aKMC)** via the **EON** software. While MD is excellent for short-term dynamics (picoseconds), it cannot simulate rare events like diffusion or chemical ordering that happen over seconds or hours. kMC bridges this gap.

We will treat EON as another type of "Dynamics Engine." The key challenge is that EON is an external C++ code that needs to call our Python-based potential function. We solve this by implementing a **Python Driver** (`pace_driver.py`) that acts as a bridge, and we inject the "Uncertainty Watchdog" logic directly into this driver.

## 2. System Architecture

Files to create/modify are **bolded**.

```ascii
src/mlip_autopipec/
├── infrastructure/
│   └── dynamics/
│       └── **eon_client.py**   # Wrapper for EON
└── **scripts**/
    └── **pace_driver.py**      # Standalone script for EON to call
```

## 3. Design Architecture

### 3.1. EON Configuration
*   `type`: Literal["eon"]
*   `temperature`: float
*   `events`: int (Number of kMC steps)

### 3.2. EonClient Logic
*   **Setup**: Creates the EON directory structure.
*   **Config**: Writes `config.ini`.
*   **Driver**: Copies `pace_driver.py` and the current `potential.yace` to the run directory.
*   **Execution**: Calls `eonclient` (assuming server/client model or standalone mode).
*   **Monitoring**: Checks the return code. If the driver exits with a specific error code (e.g., 100), it treats it as a **Halt**.

### 3.3. pace_driver.py Logic
*   This is a standalone script.
*   It reads atomic coordinates from Standard Input (or EON's format).
*   It loads `potential.yace` using `pyace` or `lammps` python interface.
*   It computes Energy, Forces, and **Gamma**.
*   **Watchdog**: `if max_gamma > threshold: exit(100)`.

## 4. Implementation Approach

1.  **Driver Script**: Implement `scripts/pace_driver.py`. It must be robust and self-contained.
2.  **Configuration**: Implement `config.ini` generation for EON.
3.  **Adapter**: Implement `EonClient`.
4.  **Orchestrator**: Ensure the Orchestrator can handle the `EonClient` interchangeably with `LammpsAdapter`.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Driver Test**: Run `pace_driver.py` manually. Pipe in a structure. Assert it outputs energies.
*   **Config Test**: Assert generated `config.ini` points to the driver script.

### 5.2. Integration Testing (Mocked EON)
*   **Mock**: Create a fake `eonclient` executable that calls the driver script once and then exits.
*   **Run**: `EonClient.explore()`.
*   **Verify**: Orchestrator receives the result.
