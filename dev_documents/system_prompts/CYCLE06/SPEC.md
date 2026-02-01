# Specification: Cycle 06 - Advanced Integration (kMC & Production)

## 1. Summary

Cycle 06 represents the final frontier of the project. It extends the system's capabilities beyond the nanosecond scale of Molecular Dynamics into the second-to-hour scale of **Kinetic Monte Carlo (kMC)**. This is achieved by integrating with **EON**, a software package for long-timescale simulations.

The challenge here is "Interoperability". EON is designed to call external potentials. We must build a bridge that allows EON to use our ACE potential (and its Hybrid Safety Net) efficiently. Furthermore, this cycle focuses on **Production Readiness**. This includes polishing the command-line interface, creating distribution packages (making the potential portable), and ensuring that the system is robust enough for unattended multi-day runs on HPC clusters.

## 2. System Architecture

### 2.1 File Structure

Files to be created or modified (bold):

```
mlip-autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── physics/
│       │   └── dynamics/
│       │       ├── **eon_wrapper.py**      # EON management
│       │       └── **eon_config.ini**      # Template for EON
│       ├── inference/
│       │   └── **pace_driver.py**          # The bridge script for EON
│       └── infrastructure/
│           └── **production.py**           # Packaging/Deployment logic
├── tests/
│   └── integration/
│       └── **test_eon_interface.py**
└── config.yaml
```

### 2.2 Component Blueprints

#### `src/mlip_autopipec/physics/dynamics/eon_wrapper.py`

```python
class EonWrapper:
    def run_akmc(self, potential: Potential, structure: Atoms) -> EonResult:
        """
        Sets up the EON directory (config.ini, pos.con).
        Installs the pace_driver.py into the directory.
        Runs 'eonclient'.
        Monitors for OTF halts (via exit codes).
        """
        # ... logic ...
```

#### `src/mlip_autopipec/inference/pace_driver.py`

```python
# This is a standalone script copied to the EON work dir
import sys
from ase.io import read
from pyacemaker.calculator import PaceCalculator

def main():
    # Read geometry from stdin (EON format)
    atoms = read_geometry(sys.stdin)

    # Calculate
    calc = PaceCalculator("potential.yace")
    results = calc.calculate(atoms)

    # Check Uncertainty
    if results.gamma > THRESHOLD:
        sys.exit(100) # Signal Halt

    # Print Energy/Forces to stdout
    print_results(results)

if __name__ == "__main__":
    main()
```

## 3. Design Architecture

### 3.1 The "Driver" Pattern
EON communicates with potentials via standard input/output streams.
*   **Challenge**: Loading the ACE potential (which can be 50MB+) takes time. If we reload it for every force call, EON will be painfully slow.
*   **Solution**: We implement `pace_driver.py` efficiently. Ideally, we would use a persistent server mode (socket), but for Cycle 06 we will stick to the script-based interface, optimizing the startup time (lazy imports).

### 3.2 Production Manifest
When a potential is "released", it shouldn't just be a `.yace` file.
*   **Manifest**: We generate a `manifest.json` containing:
    *   `version`: SemVer string.
    *   `author`: User name.
    *   `training_set_size`: int.
    *   `validation_metrics`: dict.
    *   `creation_date`: timestamp.
*   **Package**: The system creates a `release_v1.0.zip` containing the potential, the manifest, and the HTML report.

## 4. Implementation Approach

### Step 1: EON Driver
1.  Implement `pace_driver.py`. This script must be robust and depend only on `ase` and `pyace` (or `lammps`).
2.  Test it manually by piping a coordinate string into it and checking the output format.

### Step 2: EON Wrapper
1.  Implement `EonWrapper`.
2.  Logic to write `config.ini` based on `config.yaml`.
3.  Logic to convert `ase.Atoms` to EON's `.con` format.

### Step 3: kMC Loop Integration
1.  Update `Orchestrator` to support `ExplorationMethod.AKMC`.
2.  Handle the special exit code (100) from the driver to trigger the OTF Selection phase.

### Step 4: Production Deployment
1.  Implement `ProductionDeployer` class.
2.  Add a `finalize` step to the Orchestrator to zip up the best potential.

## 5. Test Strategy

### 5.1 Unit Testing Approach (Min 300 words)
*   **Driver Protocol**: We will unit test `pace_driver.py`'s I/O logic. We will construct a string mimicking EON's input format, pass it to the driver's parser function, and assert it returns the correct `Atoms` object. We will also test the output formatter.
*   **Manifest Generation**: We will create a test that populates a `ProductionManifest` object and serializes it to JSON, verifying that all required fields are present and correctly formatted.

### 5.2 Integration Testing Approach (Min 300 words)
*   **EON Loop**: This is a complex integration. We will rely on a "Mock EON" if the binary is unavailable. The mock will simply call the `potential` script with a dummy geometry and assert that the script runs and returns *something*.
*   **Real Mode**: If EON is available, we run a very short "Saddle Search". We verify that EON correctly moves the atoms and that the potential energy decreases as it relaxes to the saddle point.
*   **Deployment**: We run the `ProductionDeployer` and verify that the resulting ZIP file contains all expected files and that the JSON is valid.
