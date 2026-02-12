# Cycle 07 Specification: Scalability (aKMC & EON Integration)

## 1. Summary

Cycle 07 addresses the "Time-Scale Problem". While MD is excellent for nanosecond-scale dynamics, many critical materials phenomena (e.g., vacancy diffusion, ordering in alloys, surface reconstruction) happen over seconds, hours, or years. To capture these, we integrate **EON**, a software package for **Adaptive Kinetic Monte Carlo (aKMC)**.

This cycle implements a seamless bridge between the Python-based PYACEMAKER ecosystem and the C++-based EON client. We will create a mechanism for EON to use our trained ACE potentials as its energy/force engine, and conversely, for PYACEMAKER to treat EON as just another "Dynamics Engine" that can trigger Active Learning.

## 2. System Architecture

Files in **bold** are to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── dynamics/
│   ├── **eon_driver.py**     # Main EON Controller
│   └── **potential_server.py** # Script for EON to call (Potential Driver)
├── domain_models/
│   └── **config.py**         # Update DynamicsConfig (EON settings)
└── tests/
    └── unit/
        └── **test_eon.py**
```

## 3. Design Architecture

### 3.1 EON Driver (`dynamics/eon_driver.py`)
This class manages the EON client process.
*   **Setup**: Generates `config.ini`, `pos.con`, and `potential.yace` in a working directory.
*   **Execution**: Launches the `eonclient` binary as a subprocess.
*   **Monitoring**: Watches the output for "Process Search" completion or errors.
*   **Halt Detection**: If the potential driver exits with a specific code (due to high uncertainty in a saddle point search), the EON driver captures this.

### 3.2 Potential Server (`dynamics/potential_server.py`)
This is a standalone script that EON calls.
*   **Protocol**: It reads atomic coordinates from `stdin` (EON format) and writes Energy and Forces to `stdout`.
*   **Uncertainty Check**: Crucially, it calculates $\gamma$ for each configuration. If $\gamma > \text{threshold}$, it writes a "Bad Structure" file and exits with a special code to trigger a Halt in the main driver.

## 4. Implementation Approach

1.  **Enhance Domain Models**: Add `EONConfig` (temperature, prefactor, search_method).
2.  **Implement PotentialServer**: Create the script that loads `potential.yace` and acts as a calculator.
3.  **Implement EONDriver**:
    *   `prepare_run()`: Writes the necessary EON input files.
    *   `run()`: Executes `eonclient`.
    *   `parse_results()`: Reads `results.dat` or similar to get the time evolution.
4.  **Integration**: Update `Orchestrator` to support `mode: akmc` using `EONDriver`.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Server Protocol**: Feed a dummy coordinate string to `PotentialServer` (via stdin) and verify it prints the correct energy/force string (via stdout) matching the mock potential.
*   **Driver Config**: Verify `config.ini` generation.

### 5.2 Integration Testing
*   **Mock EON**: Create a dummy `eonclient` script that just calls the potential driver once. Verify the chain of execution (Driver -> EON -> Server -> Driver).
*   **Real EON (Optional)**: If EON is installed, run a simple adatom diffusion search.
