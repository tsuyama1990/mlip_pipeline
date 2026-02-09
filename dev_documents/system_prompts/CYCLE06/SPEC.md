# Cycle 06 Specification: OTF Loop (On-the-Fly Learning)

## 1. Summary
The "OTF Loop" is the core intelligence of the system. It connects all previous components into a closed active learning cycle. The primary mechanism is the "Watchdog" (`fix halt`) in LAMMPS, which monitors the extrapolation grade ($\gamma$) in real-time. When uncertainty exceeds a threshold, the simulation halts, and the Orchestrator takes over to perform "Local Refinement"—extracting the problematic structure, generating candidates, labeling with DFT, retraining, and resuming the simulation. By the end of this cycle, the system will autonomously "learn as it goes".

## 2. System Architecture

Files in **bold** are to be created or modified.

```ascii
src/mlip_autopipec/
├── core/
│   ├── **orchestrator.py**         # Implement the full Loop
│   └── **candidate_generator.py**  # Local candidate logic
├── components/
│   ├── dynamics/
│   │   ├── **lammps_driver.py**    # Add `fix halt` logic
│   │   └── **parser.py**           # Parse halt info
│   └── ...
```

## 3. Design Architecture

### 3.1 LAMMPS Watchdog (`lammps_driver.py`)

We modify the LAMMPS input script to include `compute pace` and `fix halt`.

```lammps
compute pace_gamma all pace potential.yace ... gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt 10 v_max_gamma > 5.0 error hard
```

**Output**: When halted, LAMMPS exits with a specific error code (or just an error message). The driver must catch this and return a `HaltSignal` object containing the timestep and max gamma value.

### 3.2 Local Candidate Generator (`candidate_generator.py`)

Instead of just labeling the single "halted" snapshot (which is often just one point on a smooth curve), we generate a small cluster of structures around it to learn the local curvature of the Potential Energy Surface (PES).

**Strategies:**
1.  **Normal Mode Sampling**: Calculate Hessian (using ACE or generic potential) and displace along soft modes.
2.  **Random Displacement**: Perturb atoms with high gamma by small amounts ($\pm 0.05 \AA$).
3.  **Short MD**: Run a very short, high-temp MD burst (10 steps) from the halted structure.

### 3.3 The Loop Logic (`orchestrator.py`)

The `run_cycle` method implements the following state machine:
1.  **EXPLORE**: Run Dynamics.
    *   If smooth completion -> Done (Converged).
    *   If Halted -> Go to DETECT.
2.  **DETECT**: Parse `log.lammps` to find the exact frame where $\gamma > threshold$. Extract `halted_structure`.
3.  **SELECT**:
    *   Generate 20 candidates around `halted_structure`.
    *   Use `Trainer.select_active_set` (D-Optimality) to pick the best 5.
    *   Embed them (Cycle 03 logic) into periodic cells.
4.  **LABEL**: Send to Oracle (DFT).
5.  **TRAIN**: Update dataset, retrain potential (Fine-tuning).
6.  **DEPLOY**: Update `potential.yace` in the run directory.
7.  **RESUME**: Restart LAMMPS from the checkpoint.

## 4. Implementation Approach

1.  **Enhance LAMMPS Driver**: Add the `fix halt` command generation. Implement parsing logic to detect the halt condition.
2.  **Implement Candidate Generator**: Create `candidate_generator.py` with simple random displacement first.
3.  **Update Orchestrator**: Implement the state machine. Crucially, handle file management (moving potentials, restarting checkpoints).
4.  **Integration**: Test the full loop with a mock Oracle.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Halt Parsing**: Feed a log file with "ERROR: Fix halt condition met" to the parser. Verify it extracts the correct timestep.
*   **Candidate Generation**: Verify `generate_candidates(structure, n=10)` returns 10 distinct structures close to the original.

### 5.2 Integration Testing
*   **Mock Loop**:
    *   Configure `Dynamics` to halt at step 50 (force a halt in mock).
    *   Configure `Oracle` to return dummy energies.
    *   Run `Orchestrator.run()`.
    *   Verify:
        1.  Dynamics starts.
        2.  Halts at step 50.
        3.  Orchestrator logs "Halt detected".
        4.  Candidates generated and labeled.
        5.  Potential retraining triggered.
        6.  Dynamics resumes (or starts new segment).
