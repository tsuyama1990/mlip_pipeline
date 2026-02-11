# Cycle 03 Specification: Oracle (DFT) & Periodic Embedding

## 1. Summary

Cycle 03 introduces the **Oracle**, the component responsible for calculating the "ground truth" labels (energy, forces, stress) for atomic structures. Since these calculations are computationally expensive and prone to numerical instability (e.g., SCF non-convergence), the Oracle includes a "Self-Healing" mechanism to automatically retry failed calculations with safer parameters.

Critically, this cycle implements **Periodic Embedding**, a technique to cut out local clusters from large MD snapshots and embed them into small periodic cells. This allows us to perform accurate DFT calculations on "interesting" local environments without simulating the entire massive system, drastically reducing computational cost.

## 2. System Architecture

### File Structure

Files in **bold** are new or modified in this cycle.

```ascii
src/mlip_autopipec/
├── components/
│   ├── oracle/
│   │   ├── **__init__.py**
│   │   ├── **base.py**          # BaseOracle class
│   │   ├── **qe.py**            # Quantum Espresso wrapper (ASE)
│   │   ├── **vasp.py**          # VASP wrapper (Placeholder)
│   │   ├── **embedding.py**     # Periodic Embedding logic
│   │   └── **healer.py**        # Self-healing logic (SCF retry)
│   └── mock.py                  # MockOracle update
├── domain_models/
│   ├── **results.py**           # DFTResult model
│   └── config.py                # Updated with OracleConfig
└── core/
    └── orchestrator.py          # Updated to use Oracle
```

### Component Interaction
1.  **Orchestrator** identifies structures needing labels (from Generator or Dynamics Halt).
2.  **Orchestrator** calls `oracle.compute(structures)`.
3.  **Oracle** checks if **Periodic Embedding** is needed (e.g., if structure > 200 atoms).
    -   If yes, `embedding.embed_cluster(structure)` creates a smaller cell.
4.  **Oracle** delegates to `qe.py` (or `vasp.py`).
5.  **QE Wrapper** generates input files and runs the external binary.
6.  **Healer (`healer.py`)** monitors the execution.
    -   If successful, parse output.
    -   If failed (e.g., `convergence not achieved`), modify parameters (e.g., `mixing_beta`) and retry.
7.  **Oracle** returns `DFTResult` to Orchestrator.

## 3. Design Architecture

### 3.1. DFT Result Model (`domain_models/results.py`)

```python
class DFTResult(BaseModel):
    structure: Structure  # The structure that was calculated
    energy: float
    forces: List[List[float]]
    stress: List[List[float]]
    converged: bool
    wall_time: float
    meta: Dict[str, Any]  # e.g., k-points used, mixing_beta
```

### 3.2. Oracle Configuration (`domain_models/config.py`)

```python
class OracleConfig(BaseComponentConfig):
    type: Literal["qe", "vasp", "mock"]
    binary_path: str = "pw.x"
    pseudopotential_dir: Path
    scf_params: Dict[str, Any] = {"ecutwfc": 50, "kspacing": 0.04}
    embedding_radius: float = 6.0  # Cutoff + Buffer
```

### 3.3. Periodic Embedding Logic (`components/oracle/embedding.py`)

The algorithm:
1.  Select central atom(s) (usually the ones with high uncertainty).
2.  Select all neighbors within `R_cut + R_buffer`.
3.  Place these atoms in a new orthorhombic box that fits them with minimal vacuum.
4.  Apply periodic boundary conditions.
5.  (Important) Store a mapping so we know which atoms in the embedded cell correspond to the original "valid" atoms (inside `R_cut`) versus "buffer" atoms.

### 3.4. Self-Healing Logic (`components/oracle/healer.py`)

```python
def run_with_healing(calculator, atoms):
    try:
        atoms.get_potential_energy()
    except CalculationFailed:
        # Strategy 1: Reduce mixing beta
        calculator.parameters["mixing_beta"] = 0.3
        try:
             atoms.get_potential_energy()
        except CalculationFailed:
             # Strategy 2: Increase temperature (smearing)
             calculator.parameters["smearing"] = "mv"
             calculator.parameters["degauss"] = 0.02
             atoms.get_potential_energy()
```

## 4. Implementation Approach

1.  **Implement Results Model**: Define `DFTResult` in `domain_models/results.py`.
2.  **Implement Base Oracle**: Create the abstract base class.
3.  **Implement Mock Oracle**: Update `components/mock.py` to return consistent fake forces (e.g., using a Lennard-Jones potential as a "fake DFT").
4.  **Implement QE Wrapper**: Use `ase.calculators.espresso.Espresso`. Ensure proper handling of `tprnfor` and `tstress`.
5.  **Implement Periodic Embedding**: Use `ase.neighborlist` to find neighbors and build the new `Atoms` object.
6.  **Implement Healer**: Write a decorator or wrapper function that catches `ase.calculator.CalculationFailed`.
7.  **Orchestrator Integration**: Add the labeling step to the workflow.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Embedding**: Create a large random structure, select one atom, and verify the embedded cluster contains correct neighbors and fits in the box.
-   **Healer**: Mock a calculator that raises an exception on the first call and succeeds on the second. Verify the healer retries and updates parameters.

### 5.2. Integration Testing
-   **Mock Oracle**: Feed structures to the Mock Oracle and verify `DFTResult` objects are returned with correct shapes (N, 3).
-   **Real QE (Optional)**: If `pw.x` is available in the CI environment (rare), run a tiny H2 molecule calculation. Otherwise, rely on Mock Oracle for CI.
