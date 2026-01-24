# Cycle 07 Specification: Advanced Expansion (kMC)

## 1. Summary
Cycle 07 extends the system's capabilities beyond the nanosecond scale by integrating **Kinetic Monte Carlo (kMC)** via the **EON** software. This allows the exploration of rare events (diffusion, reaction barriers). This cycle also implements the On-the-Fly (OTF) detection logic specifically for kMC, where high uncertainty during saddle point searches triggers a retraining loop.

## 2. System Architecture

```ascii
mlip_autopipec/
├── inference/
│   ├── eon.py                  # **EON Wrapper Class**
│   └── drivers/
│       └── pace_driver.py      # **EON-Pacemaker Interface Script**
```

## 3. Design Architecture

### 3.1. EON Wrapper (`inference/eon.py`)
- **Role**: Manages the `eonclient` process.
- **Config**: Writes `config.ini` for EON (Process Search, Temperature).
- **Execution**: Runs `eonclient` in a subprocess.

### 3.2. Pacemaker Driver for EON (`inference/drivers/pace_driver.py`)
EON communicates with potentials via external scripts/sockets. We need a script that:
1.  Reads coordinates from EON.
2.  Calculates Energy/Forces using Pacemaker.
3.  **Checks $\gamma$ (Uncertainty)**.
4.  If $\gamma >$ Threshold, exits with a specific error code (e.g., 100) to signal the Orchestrator.

### 3.3. Integration
The `WorkflowManager` (Cycle 06) must be updated to handle "kMC" as an exploration mode alongside "MD".

## 4. Implementation Approach

1.  **Driver Script**: Implement `pace_driver.py`. This is a standalone script that imports `pypacemaker` (if available) or calls `pace_calc`.
2.  **Wrapper**: Implement `EONWrapper.run()`.
3.  **Halt Handling**: Catch return code 100 from `eonclient`. Parse the `bad_structure.con` that generated the error.

## 5. Test Strategy

### 5.1. Unit Testing
- **Driver**: Feed a known structure to `pace_driver.py` and verify it outputs energy/forces in the format EON expects.
- **Wrapper**: Test config generation.

### 5.2. Integration Testing
- **Mock EON**: Simulate `eonclient` calling the driver.
- **Halt Trigger**: Force the driver to emit exit code 100. Verify the Wrapper catches this and returns the structure for labeling.
