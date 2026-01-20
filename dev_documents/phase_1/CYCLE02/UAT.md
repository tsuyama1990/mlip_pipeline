# CYCLE02: Surrogate-First Exploration (UAT.md)

## 1. Test Scenarios

User Acceptance Testing (UAT) for Cycle 2 is designed to demonstrate the significant value added by the Surrogate-First Exploration module. The key "wow" factor for the user is seeing how the system intelligently and automatically filters a large, noisy set of candidate structures into a small, high-quality, and diverse set, thereby saving enormous computational cost. These scenarios will be presented in a Jupyter Notebook to visually contrast the "before" and "after" states, making the benefits tangible and easy to understand.

| Scenario ID   | Test Scenario                                             | Priority |
|---------------|-----------------------------------------------------------|----------|
| UAT-C2-001    | **Filtering of Unphysical Structures**                    | **High**     |
| UAT-C2-002    | **Intelligent Selection of Diverse Structures (FPS)**       | **High**     |
| UAT-C2-003    | **End-to-End Curation Pipeline**                          | **Medium**   |

---

### **Scenario UAT-C2-001: Filtering of Unphysical Structures**

**(Min 300 words)**

**Description:**
This scenario provides a dramatic and easily understandable demonstration of the value of the surrogate pre-screening. When generating initial structures, especially with methods like large "rattling," it is common to produce configurations that are physically impossible, such as atoms being far too close together. Sending these to the DFT Factory would, at best, waste computational time on a calculation that is likely to fail, and at worst, produce nonsensical high-energy results that could corrupt the training of the MLIP. This UAT will show the user how the surrogate model acts as a crucial "sanity check."

**UAT Steps in Jupyter Notebook:**
1.  **Setup:** The notebook will import the `SurrogateExplorer` and `ase.build`.
2.  **Create Structures:** A list of `ase.Atoms` objects will be created.
    -   First, a "good" structure: a standard, equilibrium crystal cell of copper (Cu).
    -   Second, a visibly "bad" structure: a copy of the copper cell where two atoms have been manually moved to be only 0.6 Å apart.
    -   The notebook will use a 3D visualiser to display both structures side-by-side, clearly labelling them "Physically Plausible" and "Physically Implausible." The overlapping atoms in the bad structure will be obvious.
3.  **Instantiate Explorer:** An instance of the `SurrogateExplorer` is created, configured with a path to a real MACE model and a reasonable force threshold (e.g., 10 eV/Å).
4.  **Execute Screening:** The notebook will call the `_pre_screen_structures` method (or a public wrapper for it) with the list containing both structures.
5.  **Display Results:** The notebook will print the input list (containing two structures) and the output list returned by the method. The output will contain only the single "good" structure.
6.  **Explanation:** A clear markdown explanation will follow, stating: "The surrogate model instantly calculated the forces on the 'bad' structure and found them to be extremely high, indicating an unphysical configuration. It was therefore automatically and cheaply discarded *before* it could be sent for an expensive DFT calculation." This provides a clear and compelling demonstration of cost savings and improved data quality.

---

### **Scenario UAT-C2-002: Intelligent Selection of Diverse Structures (FPS)**

**(Min 300 words)**

**Description:**
This scenario is designed to showcase the intelligence of the Farthest Point Sampling (FPS) algorithm. A common issue in data generation is redundancy; for example, taking many snapshots from a simulation can result in thousands of structures that are only marginally different from one another. This UAT will create a synthetic dataset representing this exact problem and show how FPS elegantly selects a small, representative, and diverse subset. This amazes the user by showing the system's ability to "understand" the geometry of the data and select only the most informative points.

**UAT Steps in Jupyter Notebook:**
1.  **Setup:** The notebook will import the `SurrogateExplorer` and plotting libraries like `matplotlib` or `plotly`.
2.  **Create a Redundant Dataset:**
    -   A base structure, like a distorted silicon cell, is created.
    -   A loop will generate 200 new structures by applying tiny, random perturbations to the atomic positions of the base structure. This creates a dense "cloud" of very similar points.
    -   A few (e.g., 3-4) "outlier" structures will be created by applying a much larger, distinct deformation (e.g., a large shear strain) to the base structure.
    -   The total list now contains a dense cluster and a few distant outliers.
3.  **Visualise the "Problem":** The notebook will calculate the fingerprints for all 200+ structures. Since fingerprints are high-dimensional, it will use a dimensionality reduction technique like PCA or t-SNE to plot the fingerprints in 2D. The plot will visually confirm the data structure: a large, dense blob of points and a few isolated points far away. This plot is captioned: "The Problem: A Redundant Dataset".
4.  **Execute FPS:** The `SurrogateExplorer`'s selection method (specifically the FPS part) is called on this dataset, with a request to select, for example, 10 structures.
5.  **Visualise the "Solution":** The same 2D plot will be displayed again, but this time the 10 points selected by FPS will be highlighted in a different colour. The user will immediately see that the selected points are not all clustered in the middle of the blob. Instead, the algorithm will have picked the outliers and a few points spread out around the periphery of the main cluster. This is visually intuitive and powerful.
6.  **Explanation:** A markdown cell will explain: "As you can see, instead of picking 10 random but similar structures from the dense cloud, the Farthest Point Sampling algorithm intelligently identified the most unique structures—the outliers—and then spread its remaining selections to best cover the entire dataset. This ensures that every DFT calculation provides the maximum possible new information to the model."

---

## 2. Behavior Definitions

This section defines the expected behaviors of the system in the Gherkin-style Given/When/Then format.

### **UAT-C2-001: Filtering of Unphysical Structures**

```gherkin
Feature: Pre-screening of Candidate Structures
  As a user focused on computational efficiency,
  I want the system to automatically discard physically impossible structures,
  So that I do not waste expensive DFT resources on useless calculations.

  Scenario: A structure with overlapping atoms is rejected
    Given a list of two candidate structures
    And one structure is physically plausible
    And the other structure has atoms closer than 0.8 Å
    When the system pre-screens the list using a surrogate model
    Then the returned list of structures should contain exactly one structure
    And the rejected structure should be the one with the overlapping atoms.
```

### **UAT-C2-002: Intelligent Selection of Diverse Structures (FPS)**

```gherkin
Feature: Diverse Structure Selection
  As a researcher building a comprehensive training set,
  I want the system to select a small but diverse set of structures from a large pool of candidates,
  So that the resulting MLIP is trained on a wide range of atomic environments.

  Scenario: FPS selects a diverse subset from a redundant dataset
    Given a large dataset of 200 candidate structures where 95% are very similar and 5% are unique outliers
    And I ask the system to select the 10 most diverse structures
    When the system runs the Farthest Point Sampling selection algorithm
    Then the returned list should contain exactly 10 structures
    And the majority of the unique outlier structures should be included in the selection.
```

### **UAT-C2-003: End-to-End Curation Pipeline**

```gherkin
Feature: Full Data Curation Workflow
  As a user of the MLIP-AutoPipe system,
  I want a simple interface to curate a large, raw set of structures,
  So that I can be confident the input to the training module is of high quality.

  Scenario: A raw list of structures is filtered and down-selected
    Given a list of 500 raw candidate structures, including 20 known-unphysical ones and many redundant ones
    And I configure the explorer to select 50 final structures
    When I run the full `select` method of the `SurrogateExplorer`
    Then the returned list should contain exactly 50 structures
    And none of the 20 known-unphysical structures should be in the returned list
    And the structures in the returned list should be more geometrically diverse than a random sample of 50 from the input.
```
