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
├── config/
│   └── schemas/
│       └── inference.py        # Updated with EONConfig
```

## 3. Design Architecture

### 3.1. EON Configuration (`EONConfig`)
Defined in `src/mlip_autopipec/config/schemas/inference.py`.

```python
class EONConfig(BaseModel):
    eon_executable: Path | None = Field(None, description="Path to EON executable")
    job: Literal["process_search", "saddle_search", "minimization"] = Field("process_search", description="EON Job Type")
    temperature: float = Field(300.0, ge=0.0, description="Temperature (K)")
    pot_name: str = Field("pace_driver", description="Potential name (corresponds to script name)")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Additional EON parameters")

class InferenceConfig(BaseModel):
    # ... existing fields ...
    eon: EONConfig | None = Field(None, description="EON Configuration")
    active_engine: Literal["lammps", "eon"] = Field("lammps", description="Active Dynamics Engine")
```

### 3.2. EON Wrapper (`inference/eon.py`)
- **Role**: Manages the `eonclient` process.
- **Config**: Writes `config.ini` for EON based on `EONConfig`.
- **Execution**: Runs `eonclient` in a subprocess.
- **Halt Handling**: Checks for exit code 100 (which signifies high uncertainty from `pace_driver`).
- **Input/Output**: Converts ASE Atoms to EON format (`pos.con`) and back.

### 3.3. Pacemaker Driver for EON (`inference/drivers/pace_driver.py`)
A standalone Python script that acts as the potential function for EON.
- **Input**:
  - `coordinates` (via arguments or stdin).
  - `potential_path` (via environment variable `PACE_POTENTIAL_PATH` or argument).
  - `uncertainty_threshold` (via environment variable `PACE_GAMMA_THRESHOLD`).
- **Logic**:
  1.  Constructs an Atoms object from input coordinates.
  2.  Calculates Energy, Forces, and **Max Gamma** using Pacemaker (via `pypacemaker` or `pace_calc`).
  3.  **Check**: If `max_gamma > threshold`, exit with code **100**.
  4.  **Output**: If safe, print Energy and Forces in EON format (typically `E Fx Fy Fz ...`).

### 3.4. Integration (Orchestration)
The `InferencePhase` in `mlip_autopipec.orchestration.phases.inference` must be updated:
- Check `config.inference_config.active_engine`.
- If `"eon"`, instantiate `EONWrapper` and run.
- If `EONWrapper` returns halted status, locate the `bad_structure.con` (or equivalent), convert to ASE, and save as candidate with status `screening`.

## 4. Implementation Approach

1.  **Driver Script**: Implement `pace_driver.py`. This is a standalone script. It must be executable.
2.  **Wrapper**: Implement `EONWrapper.run()`. It must set environment variables for the driver (`PACE_POTENTIAL_PATH`, `PACE_GAMMA_THRESHOLD`).
3.  **Halt Handling**: Catch return code 100 from `eonclient`. Parse the `bad_structure.con` that generated the error.

## 5. Test Strategy

### 5.1. Unit Testing
- **Driver**: Feed a known structure to `pace_driver.py` (via subprocess or direct import if structured well) and verify it outputs energy/forces. Test threshold triggering.
- **Wrapper**: Test config generation and execution logic.

### 5.2. Integration Testing
- **Mock EON**: Simulate `eonclient` calling the driver.
- **Halt Trigger**: Force the driver to emit exit code 100. Verify the Wrapper catches this.
