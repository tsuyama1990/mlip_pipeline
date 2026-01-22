# Cycle 02 Specification: Physics-Informed Generator

## 1. Summary

Cycle 02 focuses on **Module A: Generator**. In the context of machine learning potentials, the quality of the initial training data largely determines the final performance of the model. If we start with only ground-state structures (perfect crystals), the model will fail catastrophically when simulations heat up or deform (the "extrapolation" problem). Conversely, if we use purely random structures, the high energies will cause convergence issues in DFT and teach the model irrelevant high-energy physics.

The goal of this cycle is to implement a "Physics-Informed" generation strategy. Instead of brute-force randomness, we use scientifically grounded algorithms:
1.  **Special Quasirandom Structures (SQS)**: For alloys (e.g., Fe-Ni), we cannot simulate infinite randomness in a small box. SQS allows us to construct a small supercell (e.g., 32 atoms) that mathematically mimics the correlation functions of a perfectly random infinite alloy. This provides the "canonical" disordered state.
2.  **Lattice Strain**: To teach the model about elasticity and pressure (Equation of State), we apply affine transformations to the simulation box. This includes isotropic compression/expansion and shear deformations.
3.  **Atomic Rattling**: To simulate thermal vibrations (phonons), we displace atoms slightly from their equilibrium positions using Gaussian noise.
4.  **Defect Engineering**: Real materials have defects. We programmatically introduce vacancies (removing atoms) and interstitials (inserting atoms) to ensure the potential learns about defect formation energies.

By the end of this cycle, the system will be able to programmatically populate the database with thousands of these diverse, physically meaningful structures, ready for the Surrogate module (Cycle 03).

## 2. System Architecture

The focus is on the `generator` package and its integration with `data_models` and `config`.

### File Structure
**bold** files are to be created or modified.

```
mlip_autopipec/
├── generator/
│   ├── **__init__.py**
│   ├── **builder.py**          # Main StructureBuilder Facade
│   ├── **sqs.py**              # SQS Generation Logic (wraps icet/ase)
│   ├── **transformations.py**  # Strain and Rattle logic
│   └── **defects.py**          # Point defect generation
├── data_models/
│   └── **structure_enums.py**  # Enum for structure types (SQS, Defect, etc.)
└── config/
    └── schemas/
        └── **generator.py**    # Config for generator parameters
```

### Data Dictionary

| Model Name | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| **SQSConfig** | enabled | bool | If True, use SQS. If False, use random substitution. |
| | supercell_size | List[int] | Dimensions of supercell (e.g., `[2, 2, 2]`). |
| **DistortionConfig** | enabled | bool | If True, apply strain and rattle. |
| | strain_range | Tuple[float, float] | Range of linear strain (e.g., `(-0.05, 0.05)`). |
| | rattle_stdev | float | Standard deviation of Gaussian noise for thermal displacements (Angstrom). |
| **DefectConfig** | enabled | bool | If True, apply defects. |
| | vacancies | bool | Enable vacancy generation. |
| | interstitials | bool | Enable interstitial generation. |
| | interstitial_elements | List[str] | Elements to insert as interstitials (optional). |
| **GeneratorConfig** | sqs | SQSConfig | SQS configuration. |
| | distortion | DistortionConfig | Distortion configuration. |
| | defects | DefectConfig | Defect configuration. |
| | number_of_structures | int | Number of unique structures to generate per batch. |
| **CandidateData** | atoms | ase.Atoms | The atomic structure. |
| | config_type | str | Tag indicating origin (e.g., "sqs_strained"). |
| | provenance | dict | Metadata (e.g., `{"strain": 0.02, "rattle": 0.1}`). |

## 3. Design Architecture

### Configuration (`GeneratorConfig`)
We add a new section to the Pydantic config in `config/schemas/generator.py`.
-   **SQSConfig**:
    -   `enabled`: `bool`.
    -   `supercell_size`: `List[int]` (e.g., `[2, 2, 2]`). Defines the size of the box relative to the primitive cell.
-   **DistortionConfig**:
    -   `enabled`: `bool`.
    -   `strain_range`: `Tuple[float, float]` (e.g., `(-0.05, 0.05)`). Min and max linear strain.
    -   `rattle_stdev`: `float` (Angstroms). The standard deviation of the Gaussian noise.
-   **DefectConfig**:
    -   `enabled`: `bool`.
    -   `vacancies`: `bool`.
    -   `interstitials`: `bool`.
    -   `interstitial_elements`: `List[str]`.
-   **GeneratorConfig**:
    -   Aggregates the above.
    -   `number_of_structures`: `int`. How many structures to generate in a batch.

### Generation Strategy Pattern
The `StructureBuilder` class acts as a Facade for various strategies. This allows us to plug in new generation methods (e.g., Interface generation) later without changing the core loop.

1.  **`SQSStrategy` (Chemical Order)**:
    -   **Algorithm**: If `icet` is installed, use `icet.tools.structure_generation.generate_sqs`. This minimizes the error between the cluster vector of the supercell and the target random vector.
    -   **Fallback**: If `icet` is missing, create a supercell with the correct number of atoms (e.g., 16 Fe, 16 Ni), and use `numpy.random.shuffle` to swap positions.
    -   **Input**: `Atoms` (primitive), `target_composition`.
    -   **Output**: `Atoms` (supercell, chemically disordered).

2.  **`StrainStrategy` (Elasticity)**:
    -   **Algorithm**: $v' = (I + \epsilon) v$. Generate a strain tensor $\epsilon$.
    -   **Hydrostatic**: $\epsilon_{xx} = \epsilon_{yy} = \epsilon_{zz} = \delta$. Off-diagonals are 0.
    -   **Shear**: Off-diagonals are non-zero.
    -   **Volume Preservation**: For pure shear, ensure $\det(I+\epsilon) = 1$.

3.  **`DefectStrategy` (Topology)**:
    -   **Vacancy**: Randomly select $N$ indices using `random.sample(range(len(atoms)), k=N)`. Create a new Atoms object excluding these indices.
    -   **Interstitial**: Use Voronoi tessellation (via `scipy.spatial.Voronoi`) to find the vertices of the Wigner-Seitz cell (voids). Place an atom at the vertex with the largest distance to neighbors.

### Data Flow
1.  **User** runs `mlip-auto generate`.
2.  **App** loads `GeneratorConfig`.
3.  **`StructureBuilder`** initializes the base crystal structure (fcc/bcc) based on `TargetSystem`.
4.  **`StructureBuilder`** loops $N$ times:
    -   Clone the base structure.
    -   Apply `SQSStrategy` (if alloy).
    -   Apply `StrainStrategy` (randomly sample $\epsilon$ from range).
    -   Apply `RattleStrategy` (randomly sample displacements).
    -   (Optional) Apply `DefectStrategy` (with some probability).
5.  **`StructureBuilder`** returns a list of `Atoms`.
6.  **App** converts `Atoms` to `CandidateData` objects (adding metadata like `config_type="sqs_strained"` and `uuid`).
7.  **`DatabaseManager`** saves them to SQLite.

## 4. Implementation Approach

1.  **Implement Strategies**:
    -   Start with `transformations.py`. Implement `apply_strain(atoms, strain_tensor)` and `apply_rattle(atoms, sigma)`. These are pure functions. Use `atoms.set_cell(..., scale_atoms=True)` for strain.
    -   Implement `defects.py`. Write a simple `create_vacancy(atoms, count=1)` function.
    -   Implement `sqs.py`. Check for `icet` import. If `ImportError`, warn user and fall back to random shuffle.
2.  **Build the Facade**:
    -   Create `generator/builder.py`. The `StructureBuilder` class should take the config in `__init__`.
    -   Method `build_batch() -> List[CandidateData]`. This manages the loop and applies the transformations sequentially. It generates unique IDs for each structure.
3.  **Update Config**:
    -   Add `generator` section to `config/models.py`. Ensure it's optional (default values provided).
4.  **CLI Integration**:
    -   Add `generate` command to `app.py`. It should call `builder.build_batch()` and then `db.save_candidates()`.
    -   Add a flag `--dry-run` to print what would be generated without saving.

## 5. Test Strategy

### Unit Testing
-   **Strain Transformation**: Create a unit cube (1x1x1). Apply 10% hydrostatic strain. Assert new volume is $1.1^3 \approx 1.331$. Assert angles are still 90 degrees.
-   **Shear Transformation**: Apply shear. Assert volume is roughly constant (for small strains) but angles change.
-   **Rattle**: Create a chain of atoms. Rattle with $\sigma=0.1$. Assert positions changed. Assert `norm(old - new)` is approx 0.1 (statistically).
-   **Vacancy**: Create 100 atoms. Apply 1 vacancy. Assert len is 99.
-   **SQS**: Generate an Fe50Ni50 structure. Verify atom counts are equal.

### Integration Testing
-   **End-to-End Generation**:
    -   Run `mlip-auto generate` with `n=10`.
    -   Check DB count is 10.
    -   Inspect metadata: `config_type` should be present and correct.
    -   Check uniqueness: Generated structures should not be identical (check positions or simple hash).
    -   Check persistence: Close and reopen DB, ensure data is still there.
