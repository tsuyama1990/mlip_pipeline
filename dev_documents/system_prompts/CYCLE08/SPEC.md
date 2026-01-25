# Cycle 08 Specification: Integration & EON (Full System)

## 1. Summary

Cycle 08 is the final implementation phase. It achieves two major goals:
1.  **kMC Integration**: Integrating **EON** (Eon Client/Server) to enable Adaptive Kinetic Monte Carlo (aKMC) simulations. This allows the system to explore rare events and long time scales (seconds to hours) that are inaccessible to MD.
2.  **Full System Integration**: Connecting all previous modules (Oracle, Trainer, Dynamics, Generator, Validation) into a seamless, self-driving loop managed by the Orchestrator.

By the end of this cycle, the "Zero-Config" vision will be realized. A user will provide a `config.yaml` and a `POSCAR`, and the system will iterate until convergence.

## 2. System Architecture

We complete the `dynamics` module and finalize the orchestration.

### 2.1 File Structure

```ascii
src/mlip_autopipec/
├── dynamics/
│   ├── **eon.py**              # EON Wrapper
│   └── **pace_driver.py**      # EON-compatible potential script
├── orchestration/
    └── **manager.py**          # Finalized Logic
```

## 3. Design Architecture

### 3.1 EON Wrapper (`dynamics/eon.py`)

*   **Responsibility**: Configure and run EON.
*   **Config Generation**: Must generate `config.ini` for EON, setting `job = process_search` (Dimer method) or `job = saddle_search`.
*   **Driver Interface**: EON communicates with potentials via stdin/stdout. We must provide a python script (`pace_driver.py`) that loads the ACE potential and computes forces for EON.

### 3.2 Pace Driver (`dynamics/pace_driver.py`)

*   This is a standalone script deployed to the EON working directory.
*   **Logic**:
    1.  Read coordinates from stdin.
    2.  Calculate E, F using `pyace` or `lammps` (via python interface).
    3.  **Critical**: Check $\gamma$ (extrapolation grade).
    4.  If $\gamma > threshold$, exit with a specific code (e.g., 100) to signal the Orchestrator that a new active learning configuration has been found.
    5.  Else, print E, F to stdout.

### 3.3 Final Orchestration Logic

The `WorkflowManager` loop is finalized:
```python
while cycle < max_cycles:
    1. Generator: Propose structures (Defects/Strain).
    2. Dynamics: Run MD/kMC (Exploration).
       - If Halt: Extract -> Embed -> Queue for Oracle.
    3. Oracle: Compute DFT for queued structures.
    4. Trainer: Update Potential.
    5. Validation: Check Physics.
       - If Pass: Deploy and Continue.
       - If Fail: Adjust strategy (more data).
    cycle += 1
```

## 4. Implementation Approach

1.  **Step 1: EON Driver.**
    *   Develop `pace_driver.py` which loads the `.yace` potential.
    *   Implement the I/O protocol strictly matching EON's expectation.

2.  **Step 2: EON Wrapper.**
    *   Implement `run_kmc()`.
    *   Handle the exit code 100 (Halt) from the driver.

3.  **Step 3: End-to-End Polish.**
    *   Refine the CLI (`main.py`) to handle resume operations cleanly.
    *   Ensure all logs are aggregated.

## 5. Test Strategy

### 5.1 Unit Testing
*   **Driver I/O:**
    *   Feed a sample coordinate block to `pace_driver.py` (via stdin) and assert the output format is correct.

### 5.2 Integration Testing
*   **EON Integration:**
    *   Mock `eonclient` execution.
    *   Test that the wrapper generates the correct `config.ini`.
*   **Full System Dry Run:**
    *   Run the entire pipeline with mocks for all external binaries.
    *   Verify that the `WorkflowState` transitions correctly through all phases (Exploration -> Selection -> Refinement -> Validation).
