# Cycle 03 Specification: Oracle Interface (DFT)

## 1. Summary
Cycle 03 builds the **Oracle**, the source of truth for the system. It enables the execution of First-Principles calculations (DFT) using Quantum Espresso (and potentially VASP). This cycle also implements the critical **Periodic Embedding** logic to cut out clusters from large MD snapshots and **Self-Healing** capabilities to recover from SCF convergence failures automatically.

## 2. System Architecture

```ascii
mlip_autopipec/
├── config/
│   └── schemas/
│       └── dft.py              # **DFT Config Schema**
├── dft/
│   ├── __init__.py
│   ├── runner.py               # **Base Runner Class**
│   ├── qe.py                   # **Quantum Espresso Implementation**
│   ├── inputs.py               # **Input File Generation**
│   ├── parsers.py              # **Output Parsing**
│   └── recovery.py             # **Error Handler Strategy**
└── utils/
    └── embedding.py            # **Periodic Embedding Logic**
```

## 3. Design Architecture

### 3.1. DFT Runner (`dft/runner.py`, `dft/qe.py`)
Abstract base class `DFTRunner` defining the contract:
- `calculate(atoms, run_dir) -> DFTResult`
- `DFTResult` contains energy, forces, stress, and success status.
- `QERunner` implements this for Quantum Espresso, handling `pw.x` execution via `subprocess`.

### 3.2. Input Generation (`dft/inputs.py`)
- **Auto K-Spacing**: Logic to convert `kspacing` (1/Å) into a k-point grid mesh (e.g., `4 4 4`) based on cell dimensions.
- **Pseudopotentials**: Integration with SSSP library paths.
- **Flags**: Ensure `tprnfor=.true.` and `tstress=.true.` are always set.

### 3.3. Recovery Handler (`dft/recovery.py`)
A strategy pattern to handle failures.
- **Errors**: `ConvergenceError`, `WalltimeError`.
- **Strategies**:
    1.  Reduce `mixing_beta`.
    2.  Change diagonalization (e.g., `david` -> `cg`).
    3.  Increase temperature (smearing).

### 3.4. Periodic Embedding (`utils/embedding.py`)
Logic to extract a local environment.
- **Input**: `Atoms` (large), `center_index`, `cutoff`.
- **Output**: `Atoms` (small supercell).
- **Algorithm**:
    1. Select atoms within `cutoff + buffer`.
    2. Construct a new minimal orthorhombic cell that fits these atoms.
    3. Map atoms into the new cell.

## 4. Implementation Approach

1.  **Config**: Define `DFTConfig` (command, pseudopotential_dir, kspacing).
2.  **Input Gen**: Implement `write_pw_input`. Test with various cell sizes.
3.  **Parsers**: Implement `parse_pw_output`. Must extract `TOTAL_ENERGY`, forces array, stress tensor.
4.  **Runner**: Implement `QERunner`. Use `shutil.which` to find `pw.x`.
5.  **Recovery**: Implement a retry loop in `QERunner.calculate` that calls `RecoveryHandler` upon failure.

## 5. Test Strategy

### 5.1. Unit Testing
- **Input Generator**: Check generated strings against expected QE format. Verify k-point logic.
- **Parser**: Feed mock `pw.x` output files (success and failure) and assert parsed values.
- **Recovery**: Simulate a sequence of failures and verify the strategy updates parameters correctly.

### 5.2. Integration Testing
- **Mock Binary**: Create a dummy `pw.x` script that reads input and writes a predefined output (or error).
- **Execution**: Run `QERunner` against the mock binary.
- **Embedding**: Test extracting a cluster from a known crystal structure and verify the geometry is preserved in the new cell.
