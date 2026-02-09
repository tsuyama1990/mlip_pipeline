# Cycle 06: On-the-Fly (OTF) Loop Specification

## 1. Summary

Cycle 06 is the culmination of the core development phase, integrating all previously built components into the **Active Learning Loop**. The primary goal is to implement the **On-the-Fly (OTF)** monitoring mechanism, where the Dynamics Engine (LAMMPS) continuously checks the uncertainty ($\gamma$) of the ML potential. When this uncertainty exceeds a threshold, the simulation is halted, and the Orchestrator takes over to perform a "Halt & Diagnose" procedure. This involves extracting the problematic structure, generating local candidates, calculating their true energies with DFT (Oracle), and retraining the potential (Trainer). This cycle transforms the system from a set of disjoint tools into an autonomous, self-improving agent.

## 2. System Architecture

This cycle focuses on the `core/orchestrator.py` logic and enhancements to `components/dynamics/lammps_driver.py`.

### File Structure

The following file structure will be created/modified. **Bold** files are to be implemented/updated in this cycle.

*   **`src/`**
    *   **`mlip_autopipec/`**
        *   **`core/`**
            *   **`orchestrator.py`** (Main Active Learning Loop)
        *   **`components/`**
            *   **`dynamics/`**
                *   **`lammps_driver.py`** (Add `fix halt` logic)
                *   **`halt_handler.py`** (Extract & Diagnose logic)

## 3. Design Architecture

### 3.1 Components

#### `Orchestrator` (Enhanced)
The main loop logic:
*   **`run_cycle()`**:
    1.  **Exploration**: Call `dynamics.explore()`.
    2.  **Detection**: If `result.halted` is True, proceed to refinement.
    3.  **Selection**: Call `halt_handler.extract_high_gamma_structures()`.
    4.  **Generation**: Call `generator.generate_candidates()` around the halted structure.
    5.  **Calculation**: Call `oracle.compute()` on candidates (with embedding).
    6.  **Refinement**: Call `trainer.update_dataset()` and `trainer.train()`.
    7.  **Resume**: Restart MD with the new potential.

#### `LAMMPSDriver` (Enhanced)
Additions for OTF monitoring.
*   **`add_watchdog(threshold: float)`**:
    *   Injects `compute pace` command to calculate $\gamma$.
    *   Injects `fix halt` command to stop simulation if `max_gamma > threshold`.

#### `HaltHandler`
Logic for processing halted simulations.
*   **`extract_high_gamma_structures(dump_file: Path, threshold: float) -> list[Structure]`**:
    *   Parses the final frame of the dump file.
    *   Identifies atoms with high $\gamma$.
    *   Returns the full structure (or a cluster) centered on these atoms.

### 3.2 Domain Models

*   **`OrchestratorConfig`**:
    *   `otf_loop`:
        *   `uncertainty_threshold: float` (e.g., 5.0)
        *   `check_interval: int` (e.g., 10 steps)
        *   `max_cycles: int`

## 4. Implementation Approach

1.  **Watchdog Logic**: Update `LAMMPSDriver` to include the `compute pace` and `fix halt` commands. Ensure the `USER-PACE` package syntax is correct.
2.  **Halt Parsing**: Implement `LAMMPSDynamics._check_halt` to detect if the simulation stopped due to the watchdog or completed normally.
3.  **Structure Extraction**: Implement `HaltHandler` to read the dump file and find the exact frame and atoms responsible for the halt.
4.  **Orchestrator Logic**: Implement the state machine in `Orchestrator.run_cycle`. Use the `state_manager.py` (Cycle 01) to persist progress (e.g., "CYCLE_5_HALTED").
5.  **Refinement Loop**: Connect the Generator (Cycle 02), Oracle (Cycle 03), and Trainer (Cycle 04) to close the loop.

## 5. Test Strategy

### 5.1 Unit Testing
*   **`test_halt_handler.py`**:
    *   Create a dummy dump file with a "gamma" column.
    *   Set one atom's gamma to 10.0 (above threshold).
    *   Call `extract_high_gamma_structures`.
    *   Assert that the returned structure corresponds to that frame.
*   **`test_driver_watchdog.py`**:
    *   Call `add_watchdog(threshold=5.0)`.
    *   Verify the output string contains `fix halt ... v_max_gamma > 5.0`.

### 5.2 Integration Testing (Mocked Loop)
*   **Simulated Halt**:
    *   Configure `MockDynamics` to run for 10 steps and then return `halted=True`.
    *   Run the Orchestrator.
    *   Verify that:
        1.  `dynamics.explore` is called.
        2.  `oracle.compute` is triggered.
        3.  `trainer.train` is triggered.
        4.  The loop increments the cycle count.

### 5.3 System Testing
*   **Full Cycle**:
    *   Start with a random potential.
    *   Run the loop.
    *   The first MD run should halt almost immediately.
    *   The system should generate new data and retrain.
    *   The second MD run should last longer (demonstrating improvement).
