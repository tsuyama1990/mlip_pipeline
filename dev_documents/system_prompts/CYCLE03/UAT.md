# MLIP-AutoPipe: Cycle 03 User Acceptance Testing

- **Cycle**: 03
- **Title**: The Filter - Surrogate Explorer and Selector
- **Status**: Design

---

## 1. Test Scenarios

User Acceptance Testing for Cycle 03 is a crucial demonstration of the system's intelligence and efficiency. It provides a compelling, visual answer to the question: "How do you avoid wasting supercomputer time on useless calculations?" The UAT will showcase the **Surrogate Explorer's** two-stage filtering process, making it tangible and easy to understand. The user will be amazed to see a large, noisy dataset of thousands of candidate structures being automatically refined into a small, information-rich subset. This UAT will be presented as an interactive Jupyter Notebook (`03_intelligent_selection.ipynb`), which will use 2D scatter plots to visually represent the high-dimensional descriptor space, making the concept of "diversity" immediately intuitive.

---

### **Scenario ID: UAT-C03-01**
- **Title**: Visualising Intelligent Data Selection with FPS
- **Priority**: Critical

**Description:**
This scenario provides a step-by-step visualisation of the intelligent selection process. The user will start with a large set of candidate structures (represented as points in a 2D plot for clarity), apply the surrogate screening to remove "bad" candidates, and then run the Farthest Point Sampling algorithm to select a small, diverse subset. The notebook will generate a series of plots that show the state of the dataset at each stage: the initial random cloud, the dataset after high-energy points are removed, and the final selection with the chosen points highlighted. This powerful visualisation will provide undeniable proof that the system is not just selecting points randomly but is actively seeking out the most diverse, and therefore most valuable, structures for training.

**UAT Steps via Jupyter Notebook (`03_intelligent_selection.ipynb`):**

**Part 1: Generating a Candidate Set**
*   The notebook will begin by importing `SystemConfig`, `SurrogateExplorer`, and plotting libraries like `matplotlib` and `seaborn`.
*   **Step 1.1:** Instead of running the full generator from Cycle 02, the notebook will create a synthetic dataset for demonstration purposes. It will generate a 2D NumPy array of 1000 points, representing 1000 structures in a simplified descriptor space. To make the scenario interesting, the points will be generated from two distinct clusters, simulating a dataset with significant redundancy.
*   **Step 1.2:** A first scatter plot will be created, visualising all 1000 initial "candidate" points. The user will see a dense cloud, making it obvious that selecting points at random would be inefficient.

**Part 2: Surrogate Screening**
*   **Step 2.1:** The notebook will instantiate the `SurrogateExplorer`. The UAT will explain that we are simulating the energy screening step. A synthetic "energy" value will be assigned to each of the 1000 points, with points in a certain region of the plot being assigned artificially high energies.
*   **Step 2.2:** The notebook will call a (conceptual) `_screen_with_surrogate` method on this data.
*   **Step 2.3:** A new scatter plot will be generated. The user will see that a number of points have been removed from the plot—these are the "unphysical" structures that the surrogate model would have filtered out. This demonstrates the first layer of cost-saving.

**Part 3: Farthest Point Sampling (FPS)**
*   **Step 3.1:** The notebook will now run the core of the UAT. It will call the `_farthest_point_sampling` method on the remaining points, asking it to select, for instance, 15 points.
*   **Step 3.2 (The "Wow" Moment):** A final, multi-layered plot will be generated. It will show all the original points in a light grey, the points removed by screening in red, and the 15 points selected by FPS in a vibrant blue, with their selection order annotated and connected by a line. The user will see with their own eyes that FPS ignored the dense clusters and instead picked points around the periphery of the data cloud, maximising the coverage of the descriptor space. The visual is undeniable: the algorithm works and it is intelligent.

This scenario delivers a powerful and intuitive demonstration of a complex process, building trust and excitement about the system's capability to optimise the use of expensive computational resources.

---

## 2. Behavior Definitions

These Gherkin-style definitions specify the expected behaviour of the `SurrogateExplorer` from a user's (in this case, the system's) perspective.

**Feature: Intelligent Candidate Selection**
As the MLIP-AutoPipe system, I want to intelligently select a small, diverse subset of structures from a large pool of candidates, so that I can maximise the information gained from each expensive DFT calculation.

---

**Scenario: Screening Unphysical Structures**

*   **GIVEN** a list of 100 candidate `Atoms` objects.
*   **AND** a surrogate model is configured to predict the energy of each structure.
*   **AND** the model predicts that 10 of these structures have an energy per atom exceeding the configured `energy_threshold_ev`.
*   **AND** the remaining 90 structures are below the threshold.
*   **WHEN** the `SurrogateExplorer`'s `select` method is called on this list.
*   **THEN** the initial screening step must discard the 10 high-energy structures.
*   **AND** only the 90 low-energy structures are passed to the Farthest Point Sampling stage.

---

**Scenario: Selecting a Diverse Subset with FPS**

*   **GIVEN** a set of 100 candidate structures that have passed the energy screening.
*   **AND** their corresponding descriptor vectors have been calculated.
*   **AND** a visual inspection of the descriptor vectors shows two dense clusters and 5 isolated, outlying points.
*   **AND** the `SystemConfig` requests a final selection of `n_select = 5`.
*   **WHEN** the Farthest Point Sampling algorithm is executed on these descriptor vectors.
*   **THEN** the 5 indices returned by the algorithm must correspond to the 5 isolated, outlying points.
*   **AND** the final list of `Atoms` objects returned by the `SurrogateExplorer` must contain exactly these 5 structures.

---

**Scenario: Handling an Empty Candidate List**

*   **GIVEN** the Physics-Informed Generator produces an empty list of candidate structures.
*   **WHEN** the `SurrogateExplorer`'s `select` method is called with this empty list.
*   **THEN** the method must handle this gracefully and return an empty list.
*   **AND** it must not raise an error or exception.

---

**Scenario: Requesting More Structures Than Available**

*   **GIVEN** a list of 50 candidate structures remains after the energy screening step.
*   **AND** the `SystemConfig` requests a final selection of `n_select = 100`.
*   **WHEN** the `SurrogateExplorer`'s `select` method is called.
*   **THEN** the system should intelligently handle this case by returning all 50 available structures.
*   **AND** it must log a warning message indicating that the requested selection size exceeded the number of available candidates.
