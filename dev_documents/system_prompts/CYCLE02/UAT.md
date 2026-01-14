# MLIP-AutoPipe: Cycle 02 User Acceptance Testing

- **Cycle**: 02
- **Title**: The Seed - Physics-Informed Generator
- **Status**: Design

---

## 1. Test Scenarios

User Acceptance Testing for Cycle 02 provides the first exciting glimpse of the system's intelligent data generation capabilities. The UAT is designed as a hands-on tutorial, showcasing how the **Physics-Informed Generator** can create a diverse, physically meaningful dataset from scratch, purely from a high-level user configuration. This process is designed to amaze the user by demonstrating the solution to the "cold-start" problem in a tangible way. The UAT will be delivered as a single, well-documented Jupyter Notebook (`02_physics_informed_generator.ipynb`). This interactive format allows the user (a developer or materials scientist) to not only trigger the generation process but also to immediately visualise and inspect the resulting atomic structures, providing a clear and intuitive verification of the module's success.

---

### **Scenario ID: UAT-C02-01**
- **Title**: Generating and Visualising a Diverse Alloy Dataset
- **Priority**: Critical

**Description:**
This scenario provides a complete walkthrough of the alloy generation workflow. The user will define a simple binary alloy system (e.g., CuAu) in the configuration, execute the generator, and then analyse the output. The notebook will guide the user to inspect the different categories of generated structures: the initial SQS cell, the elastically strained structures, and the thermally "rattled" structures. By visualising the structures and printing their properties (like cell volume and atomic positions), the user can directly confirm that the generator is producing a dataset with the intended diversity, perfectly priming the system for training a potential that understands both elastic and thermal behaviour.

**UAT Steps via Jupyter Notebook (`02_physics_informed_generator.ipynb`):**

**Part 1: Configuration**
*   The notebook will start by importing the necessary components: `SystemConfig`, `PhysicsInformedGenerator`, and a visualisation tool like `ase.visualize.view`.
*   **Step 1.1:** The user will be guided to create a `SystemConfig` object specifically for generating a Copper-Gold (CuAu) alloy dataset. The configuration will specify parameters for SQS, a list of strains (e.g., `[0.95, 1.0, 1.05]`), and a list of rattle standard deviations (e.g., `[0.05, 0.1]`).

**Part 2: Generation**
*   **Step 2.1:** The user will instantiate the `PhysicsInformedGenerator` with the config object.
*   **Step 2.2:** A single notebook cell will call `generated_structures = generator.generate()`. The notebook will explain that for this UAT, the underlying `icet` SQS calculation is mocked to ensure the process runs quickly and deterministically, but that the subsequent strain and rattle transformations are the real implementation.
*   **Step 2.3:** The notebook will print the total number of structures generated, allowing the user to verify that the combinatorial logic (SQS x strains x rattles) produced the expected quantity of `Atoms` objects.

**Part 3: Verification and Visualisation**
*   **Step 3.1 (SQS):** The notebook will select the first structure (the base SQS cell), print its chemical formula to verify the composition is correct (e.g., Cu16Au16), and use an ASE-compatible viewer to render the structure. The user will visually see the random-like distribution of Cu and Au atoms.
*   **Step 3.2 (Strain):** The notebook will select a structure known to have been strained. It will print the volume of this cell and compare it to the original SQS cell's volume. The user will see a clear, quantitative difference (e.g., `Volume_strained = 0.95**3 * Volume_original`), providing direct proof that the elastic deformation was applied correctly.
*   **Step 3.3 (Rattle):** The notebook will select a "rattled" structure and visualise it. While visually subtle, the accompanying text will explain that the atomic coordinates are no longer on perfect lattice sites. To prove this quantitatively, the notebook will print the atomic positions of the original SQS and the rattled structure, showing the small, random displacements. This demonstrates the generation of thermally perturbed configurations.

The scenario will conclude by summarising that a rich dataset, containing structural information about chemical ordering, elastic response, and thermal vibrations, has been successfully and automatically generated from a simple input.

---

## 2. Behavior Definitions

These Gherkin-style definitions specify the expected behaviour of the `PhysicsInformedGenerator` from a user's perspective, forming the basis for the acceptance criteria.

**Feature: Physics-Informed Initial Structure Generation**
As a materials scientist, I want to be able to automatically generate a diverse set of atomic structures based on physical principles, so that I can create a high-quality initial dataset for training an MLIP without performing expensive AIMD simulations.

---

**Scenario: Generating a Strained and Rattled Alloy Dataset**

*   **GIVEN** I have a `SystemConfig` object configured for an alloy with the composition "Cu50Au50".
*   **AND** the configuration specifies `3` distinct strain levels and `2` distinct rattle magnitudes.
*   **AND** the underlying SQS generation library (`icet`) is mocked to return a single, valid 32-atom Cu16Au16 SQS structure.
*   **WHEN** I create a `PhysicsInformedGenerator` with this configuration and call the `generate()` method.
*   **THEN** the method should return a list containing `(1 + 3) * (1 + 2) = 12` unique ASE `Atoms` objects (assuming a combinatorial logic that includes the base structures).
*   **AND** at least one returned structure should have a cell volume that is significantly different from the base SQS structure, corresponding to the applied strain.
*   **AND** at least one returned structure should have atomic positions that are slightly displaced from the original lattice sites, corresponding to the applied rattle.
*   **AND** every structure in the returned list must have the correct chemical composition (16 Copper and 16 Gold atoms).

---

**Scenario: Generating Crystal Structures with Point Defects**

*   **GIVEN** I have a `SystemConfig` object configured for a crystalline material, Silicon (Si).
*   **AND** the configuration specifies the creation of a `3x3x3` supercell.
*   **AND** the configuration requests the generation of a `vacancy`.
*   **AND** the underlying structure generation library (`pymatgen`) is mocked to return a pristine Si supercell containing `216` atoms.
*   **WHEN** I create a `PhysicsInformedGenerator` with this configuration and call the `generate()` method.
*   **THEN** the method should return a list of `Atoms` objects.
*   **AND** one of the objects in the list must represent the structure with a vacancy.
*   **AND** this vacancy structure must contain exactly `215` atoms, which is one less than the pristine supercell.

---

**Scenario: Verifying Deterministic Generation**

*   **GIVEN** I have a `SystemConfig` object with a fixed random seed specified for the rattling procedure.
*   **WHEN** I instantiate the `PhysicsInformedGenerator` and call `generate()` to get a first list of structures.
*   **AND** I then instantiate a new `PhysicsInformedGenerator` with the *exact same configuration* and call `generate()` a second time.
*   **THEN** the second list of `Atoms` objects must be identical to the first list.
*   **AND** the atomic positions of a specific rattled structure from the first list must be exactly equal to the positions of the corresponding structure in the second list.
