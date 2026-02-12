# Cycle 05 Specification: Dynamics Engine (MD & Uncertainty)

## 1. Summary

Cycle 05 focuses on the **Dynamics Engine**, the component responsible for running molecular dynamics (MD) simulations using the trained potential. We integrate **LAMMPS**, the industry-standard code for MD.

The critical innovation here is the **Active Learning Loop implementation**:
1.  **Hybrid Potential (Overlay)**: To prevent simulation crashes due to unphysical behavior in extrapolation regions, we implement a "Hybrid Overlay" strategy. The MD engine *always* uses a sum of the Machine Learning Potential (ACE) and a physics-based baseline (ZBL/LJ). This ensures a hard repulsive wall at short distances, preventing nuclear fusion.
2.  **Uncertainty Watchdog**: We implement a real-time monitor using LAMMPS's `fix halt` command. The system calculates the extrapolation grade ($\gamma$) for every atom at every step. If the maximum $\gamma$ exceeds a safety threshold, the simulation is instantly halted. This "Halt" event triggers the Active Learning cycle (Cycle 06).

## 2. System Architecture

Files in **bold** are to be created or modified in this cycle.

```ascii
src/mlip_autopipec/
├── domain_models/
│   └── **config.py**         # Update DynamicsConfig (MD settings, thresholds)
├── dynamics/
│   ├── **__init__.py**
│   ├── **interface.py**      # Enhanced BaseDynamics
│   ├── **lammps_driver.py**  # Main LAMMPS Controller
│   ├── **hybrid_overlay.py** # Hybrid Potential Logic
│   └── **watchdog.py**       # Uncertainty Monitoring Logic
└── tests/
    └── unit/
        └── **test_dynamics.py**
```

## 3. Design Architecture

### 3.1 LAMMPS Driver (`dynamics/lammps_driver.py`)
This class constructs the LAMMPS input script and executes the simulation.
*   **Input**: `Structure`, `Potential`, `MDParameters` (steps, temperature).
*   **Output**: `Trajectory` (if successful) or `HaltEvent` (if triggered).

### 3.2 Hybrid Potential Logic (`dynamics/hybrid_overlay.py`)
The `HybridOverlay` helper constructs the `pair_style` command for LAMMPS.
*   **Logic**: It ensures that `pair_style hybrid/overlay` is used.
    *   `pair_coeff * * zbl ...` (Baseline)
    *   `pair_coeff * * pace ...` (ACE Potential)
*   **Constraint**: The baseline used here *must* match the one subtracted during training (Cycle 04).

### 3.3 Uncertainty Watchdog (`dynamics/watchdog.py`)
This logic configures the `fix halt` command.
*   **Command**: `compute gamma all pace ...` -> `fix halt ... v_max_gamma > threshold`.
*   **Return Code**: It configures LAMMPS to exit with a specific non-zero code (e.g., `100`) upon a halt event, allowing the Python driver to distinguish between "Crash" and "Active Learning Trigger".

## 4. Implementation Approach

1.  **Enhance Domain Models**: Update `DynamicsConfig` to include MD settings (timestep, thermostat) and `uncertainty_threshold`.
2.  **Implement HybridOverlay**: Write the logic to generate the `pair_style hybrid/overlay` command string.
3.  **Implement Watchdog**: Write the logic to generate the `compute pace` and `fix halt` command strings.
4.  **Implement LAMMPSDriver**: Orchestrate the simulation:
    1.  Generate Input Script (Structure + Hybrid Potential + Watchdog + MD Run).
    2.  Execute LAMMPS (using `lammps` python module or `subprocess`).
    3.  Monitor Execution.
    4.  Parse Output (Log file for thermodynamics, Dump file for trajectory).
    5.  Detect Halt: If return code is special, return `HaltEvent` with the snapshot.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Hybrid Config Test**: Verify that the generated input script contains `pair_style hybrid/overlay` and correct coefficients for ZBL.
*   **Watchdog Config Test**: Verify that the generated input script contains `fix halt` with the correct threshold.

### 5.2 Integration Testing
*   **LAMMPS Execution**: (Requires LAMMPS with PACE package) Run a short MD on a known structure. Verify it runs without error.
*   **Halt Trigger Test**: Run a simulation on a highly distorted structure (artificially created to have high $\gamma$). Verify that LAMMPS exits early with the specific return code.
