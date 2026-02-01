# Cycle 08: Expansion & Production

## 1. Summary

The final cycle extends the temporal reach of our system. While Molecular Dynamics (MD) is excellent for nanosecond-scale phenomena, it cannot simulate events that happen on the scale of seconds, hours, or years—such as diffusion in solids or corrosion. To address this, we integrate **Adaptive Kinetic Monte Carlo (aKMC)** via the **EON** software.

This integration is challenging because EON acts as the "Master" and calls our potential as a "Slave" client. We must provide a Python driver (`pace_driver.py`) that EON can execute to get energies and forces. Furthermore, we must monitor this external process for uncertainty ($\gamma$). If the aKMC search encounters an unknown transition state, we must halt it, learn the state, and resume. This is "On-the-Fly" learning for rare events.

Finally, Cycle 08 includes the **Production Release** steps. Once a potential has survived all cycles and passed validation, it is "Graduated". We implement a `ProductionDeployer` that packages the potential with metadata (citation, license, limits of validity) and prepares it for distribution.

## 2. System Architecture

We complete the `physics/dynamics` package and add `infrastructure/production`.

### File Structure
Files to be created/modified are in **bold**.

```ascii
mlip_autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── inference/
│       │   ├── **__init__.py**
│       │   └── **pace_driver.py**      # The script EON calls
│       ├── physics/
│       │   ├── dynamics/
│       │   │   └── **eon.py**          # Wrapper to launch EON
│       └── infrastructure/
│           └── **production.py**       # Packaging logic
└── tests/
    └── physics/
        └── dynamics/
            └── **test_eon.py**
```

### Component Interaction (EON Loop)

1.  **Orchestrator** launches `EonWrapper.run()`.
2.  **`EonWrapper`** creates `client/` directory and writes `config.ini`.
    -   Configures EON to use `Potentials = script`.
    -   Sets script path to `pace_driver.py`.
3.  **EON Binary** starts. It explores the Potential Energy Surface (PES) to find saddle points.
4.  **EON** calls `pace_driver.py` with atomic coordinates.
5.  **`pace_driver.py`**:
    -   Loads the ACE potential.
    -   Calculates Energy, Forces, and **Gamma**.
    -   **CRITICAL**: If Gamma > Threshold, it exits with a special status code (e.g., 100) or writes a "STOP" signal.
6.  **`EonWrapper`** detects the stop.
    -   Extracts the high-gamma saddle point candidate.
    -   Returns it to the Orchestrator for DFT.

## 3. Design Architecture

### 3.1. EON Driver (`inference/pace_driver.py`)
This is a standalone script, not a class used by the Orchestrator.
-   **Input**: Standard Input (EON format) or file.
-   **Output**: Standard Output (Energy, Forces).
-   **Logic**:
    ```python
    atoms = read_stdin()
    calc = PaceCalculator("potential.yace")
    gamma = calc.predict_gamma(atoms)
    if gamma > 5.0:
        dump_structure(atoms)
        sys.exit(100) # Signal Orchestrator
    print(calc.get_potential_energy(atoms))
    ```

### 3.2. Production Metadata
-   **Class `ProductionManifest`**:
    -   `version`: SemVer.
    -   `author`: User name.
    -   `training_set_size`: int.
    -   `validation_metrics`: `ValidationResult`.

## 4. Implementation Approach

### Step 1: Pace Driver
-   Implement `src/mlip_autopipec/inference/pace_driver.py`.
-   This script must be executable (`chmod +x`).
-   It must be lightweight (load potential once if possible, or use a daemon mode, but for now simple script invocation is easier).

### Step 2: EON Wrapper
-   Implement `physics/dynamics/eon.py`.
-   Similar to `LammpsRunner` but manages the EON directory structure (`runs/0001/`).
-   Handle the special exit code.

### Step 3: Production Packager
-   Implement `infrastructure/production.py`.
-   Zip the `.yace` file, the `report.html`, and a `metadata.json`.

## 5. Test Strategy

### 5.1. Integration Testing (Mocked EON)
-   **Mocking EON**:
    -   We don't need the EON binary. We need to test the *driver*.
    -   Feed a structure to `pace_driver.py` via stdin.
    -   Assert it prints numbers in the correct format.
    -   Feed a "weird" structure (high gamma). Assert it exits with code 100.

### 5.2. System Test (Final)
-   Run the full `mlip-auto run-loop`.
-   Check that at the end, a `dist/` folder exists with the packaged potential.
