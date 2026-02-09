# Cycle 06 Specification: Active Learning Loop (The Brain - OTF)

## 1. Summary

This cycle implements the "On-the-Fly" (OTF) Active Learning loop, the core intelligence of **PyAceMaker**. The system will no longer just run simulations; it will monitor them in real-time. When the potential's uncertainty (extrapolation grade $\gamma$) exceeds a predefined threshold, the simulation will automatically halt.

The Orchestrator then takes over:
1.  **Diagnose**: Identify the "Halt Structure" (the configuration where uncertainty spiked).
2.  **Generate Candidates**: Create local perturbations around this structure (using the Structure Generator logic from Cycle 02).
3.  **Select**: Use D-Optimality to pick the most informative candidates.
4.  **Label**: Run DFT (Cycle 03) on these candidates.
5.  **Refine**: Retrain the potential (Cycle 04) with the new data.
6.  **Resume**: Restart the simulation from the halt point with the improved potential.

This closed-loop system allows the MLIP to "learn as it goes," exploring new chemical spaces safely and efficiently.

## 2. System Architecture

The following file structure will be created/modified. Files in **bold** are the specific deliverables for this cycle.

```ascii
src/mlip_pipeline/
├── components/
│   ├── dynamics/
│   │   └── **lammps_otf.py**       # Enhanced LAMMPS Driver with OTF logic
│   └── trainers/
│       └── **active_learning.py**  # Local Active Set Selection Logic
└── core/
    └── **orchestrator.py**         # (Modified) Implement the main OTF loop
```

## 3. Design Architecture

### 3.1. OTF Interface
The `BaseDynamics` class will be extended.

*   `otf_explore(self, structure: Structure, potential: Potential, settings: Dict) -> OTFResult`
    *   Input: Similar to `explore`, but with `uncertainty_threshold` (float) and `check_interval` (int).
    *   Output: `OTFResult` object containing:
        *   `halted`: Boolean.
        *   `halt_step`: Integer.
        *   `halt_structure`: Optional[Structure].
        *   `trajectory`: Path.

### 3.2. LAMMPS OTF Driver
Located in `src/mlip_pipeline/components/dynamics/lammps_otf.py`.

*   **Logic**:
    *   Inject `compute pace` commands into `in.lammps`.
    *   Define a variable `v_max_gamma = max(c_pace_gamma)`.
    *   Use `fix halt` to stop if `v_max_gamma > threshold`.
    *   On halt, the driver detects the exit code or log message, parses the dump file to extract the last frame, and returns it as `halt_structure`.

### 3.3. Orchestrator Logic
Located in `src/mlip_pipeline/core/orchestrator.py`.

*   **Main Loop**:
    ```python
    while not converged:
        result = dynamics.otf_explore(current_structure, current_potential)
        if result.halted:
            candidates = generator.generate_local(result.halt_structure)
            selected = selector.select(candidates)
            new_data = oracle.compute_batch(selected)
            current_potential = trainer.train(new_data, initial_potential=current_potential)
            current_structure = result.halt_structure
        else:
            converged = True
    ```

### 3.4. Local Candidate Generation
The Generator (Cycle 02) needs a method `generate_local(structure, n_candidates)`.
*   Strategies:
    *   **Normal Mode Sampling**: Calculate Hessian (approx) and displace along soft modes.
    *   **Random Displacement**: Simple rattle.
    *   **MD Burst**: Short high-T MD run (risky if potential is bad, but viable).

## 4. Implementation Approach

1.  **LAMMPS Command Injection**: Modify `in.lammps` generation to include `compute pace` and `fix halt`.
2.  **Halt Detection**: Ensure the Python driver can distinguish between a normal finish (max steps reached) and a halt (threshold exceeded).
3.  **Structure Extraction**: Implement logic to read the *last* frame of a LAMMPS dump file efficiently (using `ase.io.read(..., index=-1)`).
4.  **Loop Integration**: Update the `Orchestrator.run()` method to execute the `while` loop described above.

## 5. Test Strategy

### 5.1. Unit Testing
*   **Fix Halt Command**: Verify the generated LAMMPS input string contains correct `fix ... halt ... v_max_gamma > ...` syntax.
*   **Halt Parsing**: Simulate a LAMMPS log file ending with "Fix halt condition met". Assert `OTFResult.halted` is True.

### 5.2. Integration Testing
*   **Simulated Halt**: Use a very low threshold (e.g., $\gamma=0.0$) to force a halt at step 1. Verify the system catches it and extracts the structure.
*   **Full Loop (Mock)**: Run the Orchestrator with Mock Oracle and Mock Trainer.
    *   Step 1: Explore (Halt).
    *   Step 2: Oracle (Generate Data).
    *   Step 3: Trainer (New Potential).
    *   Step 4: Explore again (Should proceed further if Mock Trainer "improves" the potential).

### 5.3. Robustness
*   **Infinite Loop Prevention**: Add a `max_cycles` limit to the Orchestrator loop to prevent it from running forever if the potential doesn't improve.
