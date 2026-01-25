# Cycle 08 Specification: Validation Suite & Production Release

## 1. Summary
Cycle 08 focuses on **Quality Assurance** and **User Experience**. It implements the comprehensive `Validator` suite to ensure the physical correctness of the generated potentials (Phonons, Elasticity, EOS). It also finalizes the CLI (`mlip-auto`) for end-users and ensures the entire system is robust for production deployment.

## 2. System Architecture

```ascii
mlip_autopipec/
├── validation/
│   ├── __init__.py
│   ├── phonon.py               # **Phonopy Interface**
│   ├── elasticity.py           # **Elastic Constants**
│   └── eos.py                  # **Equation of State**
└── app.py                      # **Final CLI Implementation**
```

## 3. Design Architecture

### 3.1. Validator (`validation/`)
- **Phonon**: Calculate phonon band structure via `phonopy`. Check for imaginary frequencies (dynamic instability).
- **Elasticity**: Calculate elastic tensor $C_{ij}$. Check Born stability criteria.
- **EOS**: Fit Birch-Murnaghan EOS. Check bulk modulus $B_0$.

### 3.2. CLI (`app.py`)
Finalize the Typer app.
- Commands: `init`, `run`, `validate`, `analyze`.
- **Rich Integration**: Progress bars, status tables.

#### Validate Command Overload
The `validate` command will support two modes:
1. **Config Validation**: `mlip-auto validate config.yaml` (default behavior if no physics flags are passed).
2. **Physics Validation**: `mlip-auto validate config.yaml --phonon --elastic --eos` (runs physics checks).

## 4. Implementation Approach

1.  **Validator**: Implement physics checks.
    - Use `ase.phonons` or `phonopy` API.
    - Use `ase.eos` for EOS.
    - Implement `ValidationConfig` in schemas.
2.  **CLI**: Connect `WorkflowManager` to the `mlip-auto run` command.
3.  **Documentation**: Generate final API docs.

## 5. Test Strategy

### 5.1. Unit Testing
- **Validator**: Run checks on a known good potential (e.g., LJ argon) and a known bad one. Verify Pass/Fail verdicts.

### 5.2. Integration Testing
- **E2E Run**: "Hello World" run. Start from empty folder, run `mlip-auto run config.yaml`, and let it complete 1 cycle.
