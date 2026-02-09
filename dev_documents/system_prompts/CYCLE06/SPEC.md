# Cycle 06 Specification: On-the-Fly (OTF) Learning Loop

## 1. Summary

Cycle 06 closes the Active Learning loop by enabling the system to learn from its mistakes in real-time. Instead of passively running MD simulations, the system becomes "Self-Aware". It monitors the extrapolation grade ($\gamma$) of the potential at every step. If the simulation enters an unknown region of the Potential Energy Surface (PES), it halts immediately to prevent unphysical behavior and requests clarification from the Oracle (DFT).

Key features:
1.  **Watchdog (Fix Halt)**: Configure LAMMPS to compute `pace gamma` and trigger a halt if $\gamma > \gamma_{thresh}$.
2.  **Halt Diagnostics**: When a halt occurs, the Orchestrator must identify *which* atoms are responsible and extract the problematic configuration.
3.  **Local Candidate Generation**: To fix the "hole" in the potential, we don't just add one point. We generate a cloud of local perturbations around the halted structure to learn the local curvature (Hessian) and robustly fill the gap.
4.  **Resume Capability**: After re-training, the simulation should ideally resume from where it left off (or restart with the new potential).

By the end of this cycle, the Orchestrator will be able to autonomously detect failure, trigger a retraining loop, and improve the potential without user intervention.

## 2. System Architecture

This cycle enhances the `components/dynamics` and `core` packages.

### File Structure
Files to be created/modified in this cycle are marked in **bold**.

```
src/mlip_autopipec/
├── components/
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── **otf_handler.py**      # Logic to parse halt dumps
│   │   └── **lammps_driver.py**    # Update to include fix halt
│   └── generator/
│       └── **local.py**            # Candidate generator around a halt
├── core/
│   └── **orchestrator.py**         # Update loop logic to handle Halts
└── domain_models/
│   └── **config.py**               # Add OTFConfig
└── tests/
    └── **test_otf.py**
```

## 3. Design Architecture

### 3.1. OTF Configuration (`domain_models/config.py`)
Update `DynamicsConfig` (or add `OTFConfig`):
*   `uncertainty_threshold`: float (e.g., 5.0).
*   `check_interval`: int (e.g., every 10 steps).
*   `patience`: int (how many halts before giving up).

### 3.2. Watchdog Implementation (`components/dynamics/lammps_driver.py`)
Add commands to `in.lammps`:
```
compute pace_gamma all pace ... gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt 10 v_max_gamma > 5.0 error hard
```
*   **Trigger**: If max_gamma exceeds 5.0, LAMMPS exits with a specific error code or message.

### 3.3. Halt Handler (`components/dynamics/otf_handler.py`)
*   **Input**: Path to LAMMPS log/dump.
*   **Logic**:
    1.  Detect "Halt" signal.
    2.  Read the final dump frame.
    3.  Identify atoms with high $\gamma$.
    4.  Extract a cluster around these atoms (or return the whole box).

### 3.4. Local Candidate Generator (`components/generator/local.py`)
*   **Input**: Halted Structure $S_0$.
*   **Output**: List of Structures $\{S_1, \dots, S_N\}$.
*   **Strategy**:
    *   **Rattle**: Small random displacements ($0.05 \AA$).
    *   **Scale**: Slight compression/expansion ($1\%$).
    *   This ensures the Oracle (DFT) sees not just the single bad point, but the local gradient, stabilizing the new potential.

## 4. Implementation Approach

1.  **Modify LAMMPS Driver**: Inject the `fix halt` commands if `uncertainty_threshold` is set.
2.  **Implement `HaltHandler`**: Write a parser for LAMMPS log files to detect the specific error message associated with `fix halt`.
3.  **Update Orchestrator Loop**:
    *   Run Dynamics.
    *   If `Halted`:
        *   Log "Uncertainty detected!".
        *   Call `HaltHandler` to get structure.
        *   Call `LocalCandidateGenerator`.
        *   Send candidates to Oracle (Cycle 03).
        *   Send labeled data to Trainer (Cycle 04).
        *   Update Potential.
        *   Restart Dynamics.
4.  **Mocking**:
    *   Simulate a LAMMPS run that runs for 50 steps, then exits with "Error: Fix halt triggered".
    *   Simulate a dump file with one atom having high gamma.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Local Generator**:
    *   Input: One structure.
    *   Action: Generate 10 local perturbations.
    *   Assert: All structures are very similar to input (RMSD < 0.1).

### 5.2. Integration Testing
*   **OTF Loop (Mocked)**:
    *   Mock LAMMPS to fail at step 100.
    *   Run Orchestrator.
    *   Assert:
        1.  Orchestrator catches the Halt.
        2.  New candidates are generated.
        3.  Oracle is called.
        4.  Trainer is called.
        5.  Potential version increments.
