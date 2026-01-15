# CYCLE04: On-The-Fly (OTF) Inference and Embedding (UAT.md)

## 1. Test Scenarios

User Acceptance Testing (UAT) for Cycle 4 is designed to showcase the "intelligence" of the active learning loop. The user experience should be one of amazement, seeing the system autonomously detect a weakness in its own knowledge and then precisely extract the necessary information to correct it. These scenarios will be presented in a Jupyter Notebook, which is perfect for visualising the atomic structures and the logic of the extraction process.

| Scenario ID   | Test Scenario                                             | Priority |
|---------------|-----------------------------------------------------------|----------|
| UAT-C4-001    | **Uncertainty Detection in a Live Simulation**            | **High**     |
| UAT-C4-002    | **Visualisation of Periodic Embedding and Force Masking** | **High**     |
| UAT-C4-003    | **Full Loop Data Handoff**                                | **Medium**   |

---

### **Scenario UAT-C4-001: Uncertainty Detection in a Live Simulation**

**(Min 300 words)**

**Description:**
This scenario provides the user with a direct view into the system's "mind" as it performs a simulation. The goal is to demonstrate that the `LammpsRunner` isn't just blindly running dynamics but is actively self-monitoring. The user will see a simulation start, run for a short time, and then automatically stop at the exact moment the MLIP's uncertainty exceeds a set threshold. This provides tangible proof of the active learning trigger mechanism.

**UAT Steps in Jupyter Notebook:**
1.  **Setup:** The notebook will import the `LammpsRunner` and load a pre-trained potential from the previous cycle's UAT. An initial, stable crystal structure will be created.
2.  **Configuration:** An `InferenceConfig` Pydantic model will be instantiated. To make the test deterministic and fast, the `uncertainty_threshold` will be set to an artificially low, sensitive value (e.g., 2.5). This ensures the trigger fires quickly. The configuration object will be displayed in the notebook.
3.  **The "Controlled Experiment":** To create a predictable uncertainty event, the notebook will give one of the atoms in the initial structure a very large initial velocity, pointing it directly at a neighbour. This "kick" will force the simulation to explore a high-energy, non-equilibrium configuration that is likely to be outside the model's training distribution. The modified structure will be visualised.
4.  **Execute Simulation:** The `lammps_runner.run()` method is called. To enhance the user experience, the notebook will be configured to stream and display the `stdout` from the LAMMPS process. The user will see the simulation timesteps printing to the screen (`Step, Temp, E_pair, ...`).
5.  **The "Aha!" Moment:** After a few timesteps, the LAMMPS output will suddenly stop. The `run()` method will return a result. The notebook will then print a clear message: "✅ **Uncertainty Detected!** The simulation was automatically stopped at timestep `X` because atom `Y` exceeded the uncertainty threshold. The system is now extracting this configuration for retraining." This confirms the primary goal of the scenario: the system can successfully monitor itself and react to uncertainty.

---

### **Scenario UAT-C4-02: Visualisation of Periodic Embedding and Force Masking**

**(Min 300 words)**

**Description:**
This scenario is designed to be visually impressive and educational. It directly follows the previous scenario and explains the innovative and scientifically crucial part of the process: *how* the system extracts data. The user will be shown the difference between a naive "cluster" cutout and the superior Periodic Embedding method. They will also see a clear visualisation of the force mask, helping them understand how the system avoids learning from corrupted "boundary" forces. This builds confidence not just that the system works, but that it works in a scientifically rigorous way.

**UAT Steps in Jupyter Notebook:**
1.  **Prerequisite:** This scenario uses the `UncertainStructure` object returned from UAT-C4-001.
2.  **Load the Full Frame:** The notebook will first load the complete atomic structure from the simulation frame where uncertainty was detected. This "large cell" (e.g., 200+ atoms) will be visualised. The uncertain atom will be highlighted in red.
3.  **Visualise Periodic Embedding:** The notebook will display the extracted, smaller `ase.Atoms` object from the `UncertainStructure` result. This "embedded cell" (e.g., ~60 atoms) will be shown next to the large cell. A caption will explain: "Instead of just cutting a sphere and creating artificial surfaces, the system has extracted a small, fully periodic box from the larger simulation. Notice how atoms that were on the left side of the original box have been wrapped around to the right side of the new box, preserving the correct bulk environment."
4.  **Visualise Force Masking:** The most important visualisation comes next. The notebook will display the embedded cell again, but this time, the atoms will be coloured based on the `force_mask`.
    -   Atoms with a mask value of `1` (the "core" region) will be coloured blue.
    -   Atoms with a mask value of `0` (the "buffer" region) will be coloured grey.
    -   The original uncertain atom at the center will remain red.
    The user will see a clear sphere of blue atoms at the center, surrounded by a shell of grey atoms.
5.  **Explanation:** A clear markdown explanation will tie everything together: "The image above shows the force mask. When this structure is sent for a DFT calculation, the learning algorithm will be instructed to **only** learn from the forces on the **blue** atoms. The grey 'buffer' atoms are crucial for providing the correct chemical environment, but their own forces are ignored to prevent learning from artificial boundary effects. This ensures the model learns the true physics of the bulk material."

---

## 2. Behavior Definitions

This section defines the expected behaviors of the system in the Gherkin-style Given/When/Then format.

### **UAT-C4-001: Uncertainty Detection in a Live Simulation**

```gherkin
Feature: On-The-Fly Uncertainty Detection
  As a researcher seeking to improve my MLIP,
  I want the system to automatically stop a simulation when the model is uncertain,
  So that I can capture novel atomic configurations for retraining.

  Scenario: A simulation exploring a new configuration exceeds the uncertainty threshold
    Given a trained MLIP and a starting atomic structure
    And an MD simulation is configured with an uncertainty threshold of 3.0
    When I run the OTF simulation which is known to generate a configuration with a maximum uncertainty of 4.5
    Then the simulation should stop automatically before reaching its total specified duration
    And the system should return a valid "Uncertain Structure" object as a result.
```

### **UAT-C4-002: Visualisation of Periodic Embedding and Force Masking**

```gherkin
Feature: Scientifically Rigorous Data Extraction
  As a scientist,
  I want to be sure that the data extracted for retraining is physically meaningful and free of artifacts,
  So that I can trust the improvements made to the model.

  Scenario: Extracting an embedded structure around an uncertain atom
    Given a large periodic simulation cell and the index of a target atom
    When I request an embedded structure with a radius of 8.0 Angstroms
    Then the system should return a new, smaller `ase.Atoms` object that is also periodic
    And the new object should contain the target atom and all its neighbours within the 8.0 Angstrom radius, correctly wrapped across the original periodic boundaries.

  Scenario: Generating a force mask for an embedded structure
    Given a small, embedded atomic structure
    And a masking cutoff radius of 5.0 Angstroms
    When I request a force mask for this structure
    Then the system should return an array of 0s and 1s with the same length as the number of atoms
    And the atoms within 5.0 Angstroms of the center should have a mask value of 1
    And the atoms outside 5.0 Angstroms of the center should have a mask value of 0.
```

### **UAT-C4-003: Full Loop Data Handoff**

```gherkin
Feature: Closing the Active Learning Loop
  As a user of the end-to-end system,
  I want the data structure produced by the inference engine to be seamlessly usable by the DFT and training modules,
  So that the active learning loop is fully closed.

  Scenario: The output of the inference engine is ready for the next stage
    Given a valid `UncertainStructure` object containing an embedded `ase.Atoms` object and a `force_mask` array
    When this object is passed back to the start of the workflow
    Then the embedded `ase.Atoms` object can be successfully used as an input for the DFT Factory
    And the `force_mask` array can be stored in the database alongside the DFT result to be used by the Training Engine.
```
