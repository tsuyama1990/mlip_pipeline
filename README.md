# MLIP-AutoPipe

## Cycle 01: Core Framework & DFT Factory

### Summary
Implementation of the Automated DFT Factory using Quantum Espresso. This cycle establishes the core infrastructure for configuration, database management, and robust DFT execution.

### Key Features
- **Type-Safe Configuration**: Pydantic models for strict validation of global and DFT parameters.
- **Database Management**: Wrapper around `ase.db` for storing atomic structures and calculation results.
- **Automated DFT Execution**:
  - `QERunner`: Robust execution of `pw.x` with timeout handling.
  - `InputGenerator`: Automatic generation of inputs with required flags (`tprnfor`, `tstress`).
  - `Parsers`: Safe parsing of output files with validation for convergence and physical sanity (NaN/Inf checks).
- **Auto-Magnetism**: Detection of magnetic elements (Fe, Co, Ni) and automatic initialization of spin-polarized calculations.
- **CLI**: `mlip-auto check-config` command for validating configuration files.

### Usage

#### Check Configuration
```bash
mlip-auto check-config config.yaml
```

#### Configuration Example
```yaml
global:
  project_name: "MyMaterial"
  database_path: "data.db"
  logging_level: "INFO"

dft:
  command: "mpirun -np 4 pw.x"
  pseudopotential_dir: "/path/to/pseudos"
  ecutwfc: 50.0
  scf_convergence_threshold: 1e-6
  kpoints_density: 0.15
```

### Testing
Run the full test suite:
```bash
uv run pytest
```
