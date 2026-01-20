# CYCLE05: Data Modelling and Configuration (UAT.md)

## 1. Test Scenarios

User Acceptance Testing (UAT) for Cycle 5 is focused on the user's interaction with the system's configuration and the integrity of the data it produces. The key user experience is one of **simplicity and confidence**. The user should be amazed at how a very simple, human-readable input file can control such a complex workflow, and they should be confident that the data being stored is rich, traceable, and correct. A Jupyter Notebook is the perfect environment to demonstrate these two aspects side-by-side.

| Scenario ID   | Test Scenario                                             | Priority |
|---------------|-----------------------------------------------------------|----------|
| UAT-C5-001    | **Simple and Intuitive User Configuration**               | **High**     |
| UAT-C5-002    | **Robust Validation of User Input**                       | **High**     |
| UAT-C5-003    | **Verification of Rich Metadata in Database**             | **Medium**   |

---

### **Scenario UAT-C5-001: Simple and Intuitive User Configuration**

**(Min 300 words)**

**Description:**
This scenario is designed to showcase the "Zero-Human" philosophy from the user's point of view. It will demonstrate how the user can define a complex materials science project using a minimal, declarative, and human-readable `input.yaml` file. The user will then see how the `WorkflowManager` takes this simple input and expands it into a comprehensive, detailed `SystemConfig` object, filling in hundreds of sensible default parameters. This "magical" expansion is a core feature, showing the user that they only need to specify their high-level goals, and the system will handle the expert-level details.

**UAT Steps in Jupyter Notebook:**
1.  **Setup:** The notebook will import the `UserInputConfig` model and the `WorkflowManager` class.
2.  **The User's View:** A string representing a simple `input.yaml` file will be defined in a notebook cell. The contents will be clean and easy to read:
    ```yaml
    project_name: "My FeNi Alloy Study"
    target_system:
      elements: ["Fe", "Ni"]
      composition: { "Fe": 0.7, "Ni": 0.3 }
      crystal_structure: "fcc"
    simulation_goal:
      type: "melt_quench"
      temperature_range: [300, 2000]
    ```
    The notebook will print this string under the heading: "This is all you need to provide."
3.  **Parsing and Validation:** The notebook will load this YAML string and parse it into the `UserInputConfig` Pydantic model. The success of this step, with no errors, is the first confirmation.
4.  **The System's View:** An instance of the `WorkflowManager` will be created using the parsed user config. The notebook will then access the `manager.system_config` attribute. To avoid overwhelming the user, instead of printing the entire massive object, it will selectively display a few key expanded parameters. For example:
    -   `system_config.dft_config.magnetism`: "Automatically enabled because 'Fe' and 'Ni' are magnetic."
    -   `system_config.explorer_config.fingerprint.species`: "Set to `['Fe', 'Ni']` based on your `target_system`."
    -   `system_config.inference_config.md_params.temperature`: "Set to 2000 K, the maximum of your specified `temperature_range`."
5.  **Explanation:** A markdown cell will conclude the demonstration: "As you can see, your simple, high-level request was automatically expanded into a detailed and consistent execution plan. The system applied expert heuristics for magnetism, fingerprinting, simulation parameters, and more, without requiring any detailed input from you. This is the power of the 'Zero-Human' protocol."

---

### **Scenario UAT-C5-002: Robust Validation of User Input**

**(Min 300 words)**

**Description:**
A good user experience isn't just about making things easy; it's also about preventing mistakes. This scenario demonstrates the robustness of the system's input validation. The user will be shown several examples of common, simple mistakes in an `input.yaml` file. They will see how the system, thanks to its schema-first design with Pydantic, catches these errors immediately and provides clear, helpful error messages, rather than failing cryptically in the middle of a long run. This builds trust and shows that the system is designed to be user-friendly and safe.

**UAT Steps in Jupyter Notebook:**
1.  **Setup:** The notebook will import the `UserInputConfig` model and `pydantic.ValidationError`.
2.  **Error Case 1: Composition Mismatch:** An `input.yaml` string will be shown where the `composition` percentages do not sum to 1.0 (e.g., `{'Fe': 0.7, 'Ni': 0.2}`). The notebook will then contain a `try...except` block that attempts to parse this configuration. The `except` block will catch the `ValidationError` and print the formatted error message from Pydantic, which will clearly state something like: "Composition fractions must sum to 1.0."
3.  **Error Case 2: Invalid Element:** Another `input.yaml` string will be shown, this time with a typo in an element symbol (e.g., `elements: ["Fe", "Nx"]`). Again, the `try...except` block will catch the error, and the user will see a clear message: "'Nx' is not a valid chemical symbol."
4.  **Error Case 3: Typo in a Key:** A third example will show a typo in one of the keys of the YAML file (e.g., `project_nam: "My Project"`). The user will see that the system catches this and reports an error like: "Extra inputs are not permitted. Did you mean `project_name`?"
5.  **Explanation:** A final markdown cell will summarise the results: "The system's strict data validation acts as a safety net. It checks your input for common mistakes *before* launching the workflow, providing immediate and helpful feedback. This saves you time and prevents wasted computational resources on incorrectly configured runs."

---

## 2. Behavior Definitions

### **UAT-C5-001: Simple and Intuitive User Configuration**

```gherkin
Feature: User-Friendly Workflow Configuration
  As a user,
  I want to define my project using a simple, high-level configuration file,
  So that I can focus on my scientific goals instead of low-level parameters.

  Scenario: A minimal user configuration is expanded into a full system configuration
    Given a valid `input.yaml` file specifying only the project name, target elements, and simulation goal
    When the WorkflowManager processes this input
    Then it should successfully generate a complete internal `SystemConfig` object without errors
    And the `SystemConfig` should contain detailed, non-empty configurations for all modules (DFT, training, etc.)
    And the parameters in the `SystemConfig` should be consistent with the user's high-level request (e.g., element species match, temperatures match).
```

### **UAT-C5-002: Robust Validation of User Input**

```gherkin
Feature: Input Configuration Validation
  As a user,
  I want the system to validate my input configuration and provide clear error messages,
  So that I can quickly correct mistakes.

  Scenario: User provides a configuration with an invalid chemical element
    Given an `input.yaml` file where the `elements` list contains an invalid symbol like "Xx"
    When the system attempts to parse this configuration
    Then it should raise a `ValidationError`
    And the error message should clearly indicate that "Xx" is not a valid element.

  Scenario: User provides a configuration where composition fractions do not sum to 1.0
    Given an `input.yaml` file where the `composition` fractions sum to 0.9
    When the system attempts to parse this configuration
    Then it should raise a `ValidationError`
    And the error message should clearly state that the composition must sum to 1.0.
```

### **UAT-C5-003: Verification of Rich Metadata in Database**

```gherkin
Feature: Data Provenance and Traceability
  As a researcher,
  I want every piece of data in the training database to be tagged with rich metadata,
  So that I can trace its origin and understand how it was generated.

  Scenario: A structure generated by the active learning loop is saved with its metadata
    Given a DFT result for a structure that was extracted by the inference engine
    And this structure has a `force_mask` associated with it
    And the metadata indicates it is from 'active_learning_gen_4'
    When the system saves this result to the database
    Then a new entry should be created in the database
    And when I read this entry back, its `info` dictionary should contain the key 'config_type' with the value 'active_learning_gen_4'
    And its `arrays` dictionary should contain the key 'force_mask' with the correct numerical data.
```
