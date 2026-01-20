# Cycle 02: Physics-Informed Generator

## 1. Summary

Cycle 02 tackles the "Cold Start" problem in machine learning potential development. Before an active learning loop can even begin, it requires a seed dataset. The quality, diversity, and physical relevance of this seed dataset are the single most important predictors of the initial model's stability. If the dataset is too narrow (e.g., only containing relaxed ground-state structures), the model will have no knowledge of the repulsive potential walls or thermal fluctuations, leading to immediate failure when Molecular Dynamics (MD) is attempted. Conversely, if the dataset is randomly generated (e.g., placing atoms at random coordinates), it will be dominated by high-energy "garbage" configurations that waste expensive DFT resources without teaching the model relevant physics.

In this cycle, we implement **Module A: Physics-Informed Generator**. This module is designed to procedurally generate a diverse, physically meaningful set of atomic structures by leveraging domain knowledge from solid-state physics. We move beyond simple randomization and implement structured algorithms that target specific regions of the Potential Energy Surface (PES).

We will implement three distinct generation strategies:
1.  **Special Quasirandom Structures (SQS)**: For alloy systems, simply randomly substituting atoms in a supercell creates "clumping" (short-range order) that is unrepresentative of a true random solid solution. We use SQS (via `icet` or `mcsqs`) to mathematically optimize the atomic arrangement, matching the correlation functions of a perfectly random alloy within a finite supercell.
2.  **Lattice Strain Engineering**: To teach the model about the Equation of State (EOS) and elastic constants ($C_{11}, C_{12}, C_{44}$), we systematically apply isotropic expansion/compression and shear deformations to the equilibrium lattice. This ensures the model learns the "curvature" of the energy landscape near the minimum.
3.  **Thermal Rattling & Normal Mode Sampling (NMS)**: To simulate finite temperatures without running MD (which we can't do yet!), we apply Gaussian noise ("Rattling") to atomic positions. For molecular systems, we implement Normal Mode Sampling, which displaces atoms along the eigenvectors of the Hessian matrix, efficiently sampling the vibrational phase space.

By the end of this cycle, the system will be able to accept a chemical composition (e.g., "Fe70Ni30") and autonomously produce a database of thousands of `ase.Atoms` objects covering chemical disorder, elastic deformation, and thermal vibration, ready to train a robust initial model.

## 2. System Architecture

### 2.1. Code Blueprint and File Structure

The generator module is a self-contained package within the `mlip_autopipec` library. It interacts with the rest of the system primarily by producing lists of `ase.Atoms` objects that are then consumed by the database or surrogate modules.

The following file structure will be implemented. Files in **bold** are the primary deliverables.

```
mlip_autopipec/
├── generator/
│   ├── **__init__.py**
│   ├── **config.py**               # Pydantic schemas specific to generation parameters
│   ├── **structure_builder.py**    # The Facade class orchestrating the generation pipeline
│   ├── **sqs.py**                  # Logic for generating chemically disordered alloys (SQS)
│   ├── **strain.py**               # Logic for applying affine deformations (Strain)
│   ├── **rattle.py**               # Logic for thermal displacements (Rattling/NMS)
│   └── **utils.py**                # Geometry helpers (checking minimal distances)
└── tests/
    └── generator/
        ├── **test_sqs.py**
        ├── **test_strain.py**
        ├── **test_rattle.py**
        └── **test_structure_builder.py**
```

### 2.2. Component Interaction and Data Flow

The data flow is designed as a pipeline of transformations. A "Base Structure" enters the pipeline and is multiplied into many variations.

1.  **Initialization**:
    The process starts with `generator.structure_builder.StructureBuilder`. This class is initialized with a `GenerationConfig` object (loaded from the global config). It resolves the target system (e.g., "FCC Aluminum") into a primitive unit cell using `ase.build`.

2.  **Chemical Disorder (The SQS Stage)**:
    If the target is an alloy (e.g., Fe-Ni), the primitive cell is passed to `generator.sqs.generate_sqs()`.
    -   This module calculates the required supercell size (e.g., 32 atoms) to represent the target composition (e.g., 70:30) as integer atom counts.
    -   It invokes the `icet` library (or an internal Monte Carlo solver) to find the atomic arrangement that minimizes the error in pair correlation functions compared to a random alloy.
    -   *Output*: A single, chemically optimized `Atoms` object (the "Parent").

3.  **Elastic Deformation (The Strain Stage)**:
    The Parent structure is passed to `generator.strain.apply_strain_set()`. This function generates a *list* of new structures.
    -   **Hydrostatic**: It creates copies scaled by volume factors (e.g., 0.90, 0.95, 1.00, 1.05, 1.10).
    -   **Shear**: It creates copies with monoclinic or triclinic distortion matrices applied to the cell vectors.
    -   *Output*: A list of ~10-20 "Deformed Parents".

4.  **Thermal Disorder (The Rattling Stage)**:
    Each of the Deformed Parents is passed to `generator.rattle.apply_rattling()`. This acts as a multiplier.
    -   For each parent, it generates $N$ samples (e.g., 10).
    -   It adds random Gaussian noise $\mathcal{N}(0, \sigma)$ to atomic positions. $\sigma$ is chosen to match a target temperature (e.g., 0.1 Angstrom $\approx$ 1000K).
    -   **Sanity Check**: It calls `utils.check_distances()`. If any two atoms are closer than a hard cutoff (e.g., 1.5 Angstrom), the structure is rejected or re-rattled. This prevents sending "nuclear fusion" geometries to the DFT code.
    -   *Output*: A final list of hundreds/thousands of `Atoms` objects.

5.  **Metadata Tagging**:
    Before returning, the `StructureBuilder` iterates through the list and attaches a dictionary `info['provenance']` to each atom.
    -   Example: `{"type": "sqs_strain_rattle", "vol_strain": 0.05, "rattle_temp": 1000}`.
    -   This metadata is crucial for Cycle 4 (Training), allowing us to balance the dataset (e.g., "train on more strained structures").

## 3. Design Architecture

### 3.1. Generator Configuration (`generator/config.py`)

We define strict schemas to control the combinatorial explosion of structure generation.

-   **`GenerationConfig`**:
    -   `supercell_matrix`: `List[List[int]]` (e.g., `[[2,0,0], [0,2,0], [0,0,2]]`). Defines the size of the system.
    -   `vol_strain_range`: `Tuple[float, float]` (default `(-0.1, 0.1)`). The min/max volume scaling.
    -   `n_vol_steps`: `int` (default 5). Number of points along the EOS curve.
    -   `shear_strain_magnitude`: `float` (default 0.05).
    -   `rattle_sigmas`: `List[float]` (default `[0.01, 0.1, 0.3]`). Corresponds to low, medium, and high temperature.
    -   `n_rattle_samples`: `int` (default 5). How many snapshots per sigma.
    -   `min_distance`: `float` (default 1.5). The safety cutoff in Angstroms.

### 3.2. SQS Generator (`generator/sqs.py`)

This module wraps the complexity of the Cluster Expansion formalism.

-   **Function**: `generate_sqs(primitive: Atoms, composition: Dict[str, float], size: int) -> Atoms`
    -   **Input**: A primitive cell (e.g., FCC), a composition map (e.g., `{'Fe': 0.7, 'Ni': 0.3}`), and a target size (e.g., 32 atoms).
    -   **Logic**:
        1.  **Quantization**: Convert floats to integers. $N_{Fe} = \text{round}(32 \times 0.7) = 22$. $N_{Ni} = 10$. Check if errors are acceptable.
        2.  **Supercell**: Use `ase.build.make_supercell` to create the 32-atom box.
        3.  **Optimization**:
            -   *Path A (`icet` installed)*: Initialize `ClusterSpace`. Run Monte Carlo to minimize the objective function $J = \sum (\Pi_{SQS} - \Pi_{Random})^2$.
            -   *Path B (Fallback)*: Randomly shuffle the atomic symbols. Calculate the Short-Range Order (SRO) parameter. Repeat 1000 times. Pick the one with SRO closest to 0. This ensures we don't need heavy dependencies for simple cases.

### 3.3. Strain Engineering (`generator/strain.py`)

This module applies affine transformations.

-   **Function**: `apply_hydrostatic_strain(atoms: Atoms, strain: float) -> Atoms`
    -   **Math**: Scaling factor $s = (1 + \text{strain})^{1/3}$.
    -   **Operation**: `atoms.set_cell(atoms.cell * s, scale_atoms=True)`.
-   **Function**: `apply_shear_strain(atoms: Atoms, gamma: float, mode: str) -> Atoms`
    -   **Math**: Define deformation tensor $F$. For monoclinic shear: $F = [[1, \gamma, 0], [0, 1, 0], [0, 0, 1]]$.
    -   **Operation**: `new_cell = old_cell @ F`. `new_positions = old_positions @ F`.

### 3.4. Rattler (`generator/rattle.py`)

-   **Function**: `apply_rattling(atoms: Atoms, sigma: float, n_samples: int) -> List[Atoms]`
    -   **Operation**: `atoms.rattle(stdev=sigma)`.
    -   **Validation**: Uses `scipy.spatial.distance.pdist` to find all pairwise distances. If $\min(d_{ij}) < \text{cutoff}$, discard and retry.
    -   **Retry Logic**: Attempt 10 times. If it still fails (e.g., density is too high), raise `GenerationError` or relax the cutoff slightly (with warning).

## 4. Implementation Approach

We will build the components in order of complexity, starting with the simplest geometric transformations.

1.  **Phase 1: Strain & Rattle (The Foundation)**
    -   Implement `strain.py`. This is pure linear algebra. Easy to verify.
    -   Implement `rattle.py`. Crucial to get the neighbor list checking correct. We will use `ase.neighborlist` for efficiency with Periodic Boundary Conditions (PBC).
    -   Test: Create a dummy cell. Apply strain. Verify volume. Apply rattle. Verify positions changed.

2.  **Phase 2: SQS (The Logic)**
    -   Implement `sqs.py` with the "Fallback" strategy (Random Shuffle + SRO check) first. This removes the blocking dependency on `icet`.
    -   (Optional) If `icet` is available in the environment, add the advanced path.
    -   Test: Generate an Fe-Ni alloy. Check that the composition is exactly correct.

3.  **Phase 3: The Builder (The Glue)**
    -   Implement `StructureBuilder`.
    -   Create the pipeline: `Base -> SQS -> Strain -> Rattle`.
    -   Implement the metadata tagging system. This is vital for the database.

4.  **Phase 4: CLI Integration**
    -   Add a command `mlip-auto generate input.yaml` that runs the builder and saves an `initial_dataset.xyz` file.
    -   Visualize this file in Ovito to visually inspect the diversity.

## 5. Test Strategy

### 5.1. Unit Testing
-   **Strain Correctness**:
    -   Input: Cubic box $L=10$, Volume=1000.
    -   Action: Apply `vol_strain=0.1` (+10%).
    -   Assert: New Volume $\approx 1100$.
    -   Assert: Fractional coordinates remain identical.
-   **Rattle Safety**:
    -   Input: A dimer with distance 2.0A. Cutoff 1.5A.
    -   Action: Rattle with large sigma.
    -   Check: If atoms move to 1.4A, the function should return `None` or raise Error.
-   **Composition Integrity**:
    -   Input: Target `{'A': 0.5, 'B': 0.5}`, Size 10.
    -   Output: 5 atoms of A, 5 atoms of B.
    -   Assert: `assert atoms.get_chemical_symbols().count('A') == 5`.

### 5.2. Integration Testing
-   **Full Pipeline Run**:
    -   Config: `Al`, fcc, 5 volume steps, 5 rattle samples.
    -   Action: Run `StructureBuilder.build()`.
    -   Assert: Returns $1 \times 5 \times 5 = 25$ structures.
    -   Assert: All structures have valid `info` tags.
-   **Symmetry Breaking**:
    -   Input: Perfect FCC crystal.
    -   Action: Run SQS + Rattle.
    -   Assert: `spglib.get_spacegroup(atoms)` is NOT `Fm-3m` (should be lower symmetry, likely `P1`). This proves we successfully generated disorder.

### 5.3. Performance Testing
-   **Memory**: Generate 10,000 structures. Ensure the list of objects fits in memory (standard Atoms object is small, so this should be fine, but good to check).
-   **Speed**: SQS generation is the bottleneck. Verify that for a 64-atom cell, the random shuffle fallback takes < 1 second.
