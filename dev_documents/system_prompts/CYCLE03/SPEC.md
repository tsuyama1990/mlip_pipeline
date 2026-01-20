# Cycle 03 Specification: Physics-Informed Generator

## 1. Summary

Cycle 03 builds **Module A: Physics-Informed Generator**. In Cycle 01, we built the infrastructure; in Cycle 02, the calculation engine. Now we need something to calculate. The goal of this module is to generate a diverse, physically relevant set of atomic structures that define the "Phase Space" we wish to model.

Standard random structure generation (randomly placing atoms in a box) is inefficient because most configurations are extremely high energy (overlapping atoms) and physically irrelevant. Instead, we use **Physics-Informed Heuristics**:
1.  **Alloys**: We use **Special Quasirandom Structures (SQS)** to model disordered solid solutions in small supercells. This minimizes the error introduced by periodic boundaries when modeling random alloys.
2.  **Molecules**: We use **Normal Mode Sampling (NMS)** to distort molecules along their natural vibrational modes, exploring the harmonic and slightly anharmonic wells. This is far more efficient than random displacement.
3.  **Defects**: We systematically introduce vacancies, interstitials, and surfaces.
4.  **Distortions**: We apply affine strain (volume/shear) and random thermal displacements ("rattling") to all structures to learn elastic constants and phonon properties.

By the end of this cycle, the system will be able to take a target description (e.g., "Fe-Ni") and produce thousands of unique atomic configurations ready for screening.

## 2. System Architecture

New components are added to `src/generator`.

```ascii
mlip_autopipec/
├── src/
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── **alloy.py**        # SQS, Strain, Rattling logic
│   │   ├── **molecule.py**     # Normal Mode Sampling
│   │   ├── **defect.py**       # Vacancy/Interstitial generation
│   │   └── **builder.py**      # Main facade
│   └── config/
│       └── models.py           # Updated with GeneratorConfig
├── tests/
│   └── generator/
│       ├── **test_alloy.py**
│       ├── **test_molecule.py**
│       └── **test_defect.py**
```

### Key Components

1.  **`AlloyGenerator`**: A class that wraps `icet` (if available) or internal logic to generate SQS. It also handles "rattling" (adding Gaussian noise to positions) and "straining" (multiplying the cell matrix). It manages the combinatorics (e.g., 5 strains $\times$ 5 rattles = 25 structures).
2.  **`MoleculeGenerator`**: Generates distorted molecules. It requires a method to estimate the Hessian (force constants). We can use a simple Calculator (like EMT) to get the modes, then scale them to simulate different temperatures (300K, 1000K).
3.  **`DefectGenerator`**: Takes a bulk structure and systematically removes atoms (Vacancy) or inserts them (Interstitial). It uses `pymatgen` or internal symmetry analysis to avoid generating identical defects.
4.  **`StructureBuilder`**: The high-level facade. It accepts the `SystemConfig` and delegates to the appropriate generator, returning a list of `Atoms` objects with metadata tags.

## 3. Design Architecture

### Domain Concepts

**The "Unphysical" Benefit**:
We intentionally generate structures that are slightly "too close" or "too distorted".
-   **Repulsive Wall**: By pushing atoms closer than equilibrium ($< 2 \text{\AA}$), we teach the potential that energy rises steeply. If we only training on equilibrium data, the potential might predict that atoms can overlap with zero energy penalty, causing MD simulations to explode.

**SQS (Special Quasirandom Structures)**:
For an alloy $A_{x}B_{1-x}$, an SQS is a small periodic cell (e.g., 32 atoms) whose correlation functions best match the theoretical random alloy. This is far superior to random shuffling.

**Data Tagging**:
Every generated structure must be tagged. This is critical for data provenance.
-   `config_type`: `sqs`, `strain`, `rattle`, `dimer`.
-   `distortion`: `0.05` (amount of strain/rattle).
-   `origin`: `sqs_fe70ni30`.

### Data Models

```python
class GeneratorConfig(BaseModel):
    # SQS settings
    supercell_matrix: List[List[int]] = [[2,0,0], [0,2,0], [0,0,2]]

    # Distortion settings
    rattling_amplitude: float = 0.05  # Angstroms
    strain_range: Tuple[float, float] = (-0.05, 0.05) # +/- 5%
    n_strain_steps: int = 5
    n_rattle_steps: int = 3

    # NMS settings
    temperatures: List[int] = [300, 600, 1000] # Kelvin
```

## 4. Implementation Approach

1.  **Step 1: Alloy Generator**:
    -   Implement `generate_sqs(prim_cell, composition)`. Try importing `icet`. If missing, implement a fallback "Random Shuffle" and log a warning.
    -   Implement `apply_strain(atoms, strain_tensor)`. The tensor should be symmetric.
    -   Implement `apply_rattle(atoms, sigma)`. Use `numpy.random.normal`.
    -   Combine them: `generate_batch()` which returns the product of SQS $\times$ Strains $\times$ Rattles.
2.  **Step 2: Molecule Generator**:
    -   Implement `normal_mode_sampling(atoms, T)`. Use `ase.vibrations.Vibrations` with a cheap calculator (EMT/LennardJones) to get eigenmodes. Displace along these modes.
3.  **Step 3: Defect Generator**:
    -   Implement `create_vacancy(atoms)`. Iterate through unique sites, remove one, return list of unique defects.
    -   Implement `create_interstitial(atoms)`. Use Voronoi tessellation or simple geometry to find holes.
4.  **Step 4: Integration (`builder.py`)**:
    -   In `builder.py`, iterate through the config.
    -   Ensure all metadata (`info` dict) is populated.
    -   Assign UUIDs to every structure immediately.

## 5. Test Strategy

### Unit Testing Approach (Min 300 words)
-   **Stoichiometry Check**: We will test that `AlloyGenerator` produces structures with the exact requested composition. For an $A_{0.5}B_{0.5}$ alloy with 32 atoms, we assert there are exactly 16 A and 16 B. We will checks edge cases (e.g., 33% of 32 atoms) to ensure rounding is handled consistently.
-   **Strain verification**: We will apply a known hydrostatic strain (e.g., +10% volume). We will assert that `atoms.get_volume()` of the output is exactly 1.10 times the input. We will also test shear strains to ensure the cell shape changes correctly.
-   **Rattling Statistics**: We will generate 1000 rattled structures and verify that the mean displacement of atoms is close to 0 and the standard deviation matches the requested `rattling_amplitude`.
-   **Defect Uniqueness**: For a simple FCC crystal, removing any atom results in the same vacancy structure (by symmetry). We will verify that `DefectGenerator` (if implementing symmetry analysis) returns only 1 unique vacancy, or (if naive) returns N vacancies that are structurally equivalent.
-   **Metadata Presence**: We will verify that *every* generated atom object has a `uuid` and `config_type` in its `info` dict.

### Integration Testing Approach (Min 300 words)
-   **Pipeline Flow**: We will setup a `GeneratorConfig` that requests SQS, Strain, and Rattling. We will call the `StructureBuilder` facade. We will assert that:
    1.  A list of `Atoms` is returned.
    2.  The list length matches expectations (e.g., 1 SQS $\times$ 5 strains $\times$ 5 rattles = 25 structures).
    3.  Every `Atoms` object has the correct `info['config_type']` tags.
-   **ASE Database Compatibility**: We will take the generated list and try to write it to an ASE database using the `DatabaseManager` from Cycle 01. This ensures that our generated objects (and their metadata) are serializable and compatible with the persistence layer. We will verify that arrays (like initial magnetic moments) are preserved.
-   **Physical Validity**: We will check that no two atoms are closer than a hard limit (e.g., 0.5 Angstrom) after rattling. If they are, the generator should have flagged or removed them (or we accept them as high-energy data, but we should know).
