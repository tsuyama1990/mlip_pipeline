# Cycle 03: Physics-Informed Generator

## 1. Summary

Cycle 03 introduces the "Creative" component of the system: **Module A: Physics-Informed Generator**. In an Active Learning loop, the quality of the final potential is strictly limited by the diversity of the training data. If we only train on equilibrium structures, the potential will fail catastrophically when simulation conditions push the material far from equilibrium (e.g., during melting or high-strain deformation).

Therefore, this module is designed to proactively generate "difficult" and physically diverse structures *before* any simulation runs. It does not rely on random noise, which is inefficient. Instead, it uses domain knowledge to target specific regions of the configuration space:
1.  **Chemical Disorder**: Using Special Quasirandom Structures (SQS) to model random alloys efficiently.
2.  **Elastic Deformation**: Applying affine strain tensors to the simulation box to teach the potential about bulk modulus and shear strength.
3.  **Vibrational Entropy**: Using Normal Mode Sampling (NMS) to generate distorted structures that mimic high-temperature phonon distributions, ensuring the potential is stable at finite temperatures.
4.  **Defects**: Systematically introducing vacancies and interstitials to allow the potential to describe diffusion mechanisms.

## 2. System Architecture

We add the `generators` package.

```ascii
mlip_autopipec/
├── config/
├── core/
├── generators/
│   ├── __init__.py
│   ├── base.py             # The interface contract.
│   ├── alloy.py            # SQS and Lattice Strain logic.
│   ├── molecule.py         # Normal Mode Sampling logic.
│   └── defect.py           # Point defects logic.
└── tests/
    └── test_generators.py  # Validates structural properties.
```

### 2.1 Code Blueprints

This section details the exact class structures and algorithms.

#### 2.1.1 Base Generator (`generators/base.py`)

This abstract base class defines the contract for all generators.

**Class `StructureGenerator(ABC)`**
*   **Attributes**:
    *   `config` (`SystemConfig`): Access to global settings (like element list).
*   **Methods**:
    *   `__init__(self, config: SystemConfig)`
    *   `generate(self, n_samples: int) -> Iterator[Atoms]`:
        *   *Abstract Method*: Must be implemented by subclasses.
        *   Yields `ase.Atoms` objects.
    *   `_tag(self, atoms: Atoms, generator_type: str, params: dict)`:
        *   Helper method.
        *   Sets `atoms.info['config_type'] = generator_type`.
        *   Sets `atoms.info['generator_params'] = params`.
        *   Sets `atoms.info['uuid']` = new UUID.

#### 2.1.2 Alloy Generator (`generators/alloy.py`)

Handles multicomponent crystal generation and strain.

**Class `AlloyGenerator(StructureGenerator)`**
*   **Methods**:
    *   `generate(self, n_samples: int) -> Iterator[Atoms]`:
        *   Logic:
            1.  Determine supercell size (e.g., 32 or 64 atoms).
            2.  Call `self._generate_sqs()` to get base structure.
            3.  Loop `n_samples` times:
                *   Create strain tensor `eps` (random normal).
                *   Call `self._apply_strain(base, eps)`.
                *   Yield structure.
    *   `_generate_sqs(self, size: int) -> Atoms`:
        *   Uses `icet.tools.structure_generation.generate_sqs` if available.
        *   Fallback: `ase.build.bulk` -> make supercell -> `atoms.set_chemical_symbols(shuffled_symbols)`.
    *   `_apply_strain(self, atoms: Atoms, strain_tensor: np.ndarray) -> Atoms`:
        *   Input: `strain_tensor` is 3x3 (Voigt compatible).
        *   Logic:
            *   $\mathbf{h}_{new} = (\mathbf{I} + \mathbf{\epsilon}) \mathbf{h}_{old}$.
            *   `atoms.set_cell(new_cell, scale_atoms=True)`.
            *   Also apply `rattling`: `atoms.positions += np.random.normal(0, 0.01)`.

#### 2.1.3 Molecule Generator (`generators/molecule.py`)

Handles vibrational sampling using Normal Mode Sampling (NMS).

**Class `MoleculeGenerator(StructureGenerator)`**
*   **Methods**:
    *   `generate(self, n_samples: int, temperature: float = 300.0) -> Iterator[Atoms]`:
        *   Logic:
            1.  Create base structure (or load from config).
            2.  Calculate Hessian $H$ (using cheap calculator e.g. EMT or LJ).
            3.  Diagonalize $H \to \omega_i^2, \mathbf{v}_i$ (eigenvalues, eigenvectors).
            4.  Loop `n_samples` times:
                *   Sample displacements $A_i \sim \mathcal{N}(0, k_B T / \omega_i^2)$.
                *   $\mathbf{R}_{new} = \mathbf{R}_{eq} + \sum A_i \mathbf{v}_i$.
                *   Yield structure.

#### 2.1.4 Defect Generator (`generators/defect.py`)

Handles point defects.

**Class `DefectGenerator(StructureGenerator)`**
*   **Methods**:
    *   `generate(self, n_samples: int) -> Iterator[Atoms]`:
        *   Logic:
            1.  Base supercell.
            2.  Alternates between `_create_vacancy` and `_create_interstitial`.
    *   `_create_vacancy(self, atoms: Atoms) -> Atoms`:
        *   Pick random index `i`.
        *   `del atoms[i]`.
        *   Tag as `vacancy`.
    *   `_create_interstitial(self, atoms: Atoms) -> Atoms`:
        *   Find Voronoi vertices (largest voids).
        *   Pick one vertex.
        *   Insert atom (random species from config).
        *   Check `get_distances` to ensure no overlap < 1.5A.
        *   Tag as `interstitial`.

#### 2.1.5 Data Flow Diagram (Cycle 03)

```mermaid
graph TD
    Config[SystemConfig] --> Factory[GeneratorFactory]
    Factory -->|Type=Alloy| AlloyGen[AlloyGenerator]
    Factory -->|Type=Molecule| MolGen[MoleculeGenerator]
    Factory -->|Type=Defect| DefectGen[DefectGenerator]

    AlloyGen -->|SQS + Strain| Stream[Iterator[Atoms]]
    MolGen -->|NMS| Stream
    DefectGen -->|Vac/Int| Stream

    Stream --> DB[DatabaseManager]
```

## 3. Design Architecture

### 3.1 Factory Pattern for Extensibility

We use a Factory Pattern to instantiate generators.
*   **Why**: The Workflow Manager (Cycle 06) will read a list of desired generators from the config (e.g., `["alloy", "defect"]`). It shouldn't know the implementation details of each.
*   **Implementation**: `GeneratorFactory.create(type, config)` returns an instance of `StructureGenerator`.

### 3.2 Physics-Based vs Random

A key design choice is to prefer **Physics-Based** perturbations over random noise.
*   **Random Noise**: Moving atoms randomly (`rattle`) creates high-energy structures that are often unphysical (atoms too close).
*   **Physics-Based**:
    *   **Strain**: Explores the Elastic/EOS regime.
    *   **NMS**: Explores the thermal regime (phonon DOS).
    *   **SQS**: Explores the configurational entropy regime.
    *   This ensures the MLIP learns the "right" physics, not just how to handle garbage data.

### 3.3 Provenance Tracking

Every generated structure is a "Synthetic Datum". It is crucial to tag it.
*   **Metadata**: `config_type` is the primary tag.
    *   `sqs_strain`: Useful for bulk modulus, elasticity.
    *   `nms_300K`: Useful for thermal properties.
    *   `vacancy`: Useful for diffusion.
*   **Usage**: Later, when analyzing model failure, we can say "The model fails on vacancies". We can then generate *more* vacancies specifically. This is the foundation of the Active Learning strategy.

## 4. Implementation Approach

1.  **Dependency Check**:
    *   Check for `icet`. If missing, implement `RandomAlloyGenerator` fallback in `alloy.py`.
    *   Check for `scipy` (needed for Voronoi in defects).

2.  **Implementation Order**:
    *   `base.py`: Define the ABC.
    *   `alloy.py`: Implement `apply_strain`. This is pure numpy.
    *   `defect.py`: Implement `create_vacancy`.
    *   `molecule.py`: Implement NMS. Use `ase.calculators.emt` for the Hessian if available, or a simple spring model.

3.  **Visualization**:
    *   Write a temporary script `debug_viz.py`.
    *   Generate 10 structures of each type.
    *   Write to `debug.xyz`.
    *   Open in OVITO/VESTA to visually confirm:
        *   Strained cells are actually non-cubic.
        *   Vacancies are actually missing atoms.

## 5. Test Strategy

### 5.1 Unit Testing

*   **Alloy Tests**:
    *   Generate Fe50Ni50. Count atoms. Assert 50% are Fe, 50% are Ni.
    *   Check that `pbc` is True.

*   **Strain Tests**:
    *   Create a cubic 10x10x10 box.
    *   Apply 10% hydrostatic expansion.
    *   Assert new volume is $11^3 = 1331$.
    *   Assert atomic fractional coordinates are unchanged.

*   **Defect Tests**:
    *   Start with 100 atoms.
    *   Request 1 vacancy.
    *   Assert result has 99 atoms.
    *   Request 1 interstitial.
    *   Assert result has 101 atoms.
    *   Check for minimum distance violations (ensure we didn't place the interstitial 0.1 Å from another atom).

### 5.2 Integration Testing

*   **Generator -> DB**:
    *   Generate a batch of 50 mixed structures (strained, defective, SQS).
    *   Use `DatabaseManager` (from Cycle 01) to save them.
    *   Query the DB for `config_type="interstitial"`. Assert we get the correct number back.
