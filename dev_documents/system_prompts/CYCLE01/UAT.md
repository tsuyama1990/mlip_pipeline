# CYCLE01: The Foundation - Automated DFT Factory (UAT.md)

## 1. Test Scenarios

User Acceptance Testing (UAT) for Cycle 1 is designed to verify that the core functionality of the Automated DFT Factory is robust, reliable, and delivers the expected outcomes for a foundational component. The scenarios are designed from the perspective of a user (or a component acting on the user's behalf) who wants to perform a DFT calculation on a given atomic structure without needing to understand the intricate details of the underlying DFT engine. The primary vehicle for this UAT will be a Jupyter Notebook, which provides an ideal interactive environment to demonstrate each feature, show the inputs and outputs clearly, and serve as a living document and tutorial for the system's capabilities.

| Scenario ID   | Test Scenario                                             | Priority |
|---------------|-----------------------------------------------------------|----------|
| UAT-C1-001    | **Successful "Happy Path" Calculation**                   | **High**     |
| UAT-C1-002    | **Automatic Parameter Heuristics Verification**           | **High**     |
| UAT-C1-003    | **Resilience to Convergence Failure**                     | **High**     |
| UAT-C1-004    | **Data Persistence and Retrieval**                        | **Medium**   |

---

### **Scenario UAT-C1-001: Successful "Happy Path" Calculation**

**(Min 300 words)**

**Description:**
This is the most fundamental test case. Its purpose is to verify that the system can successfully perform a standard, end-to-end DFT calculation for a well-behaved atomic structure and return a physically plausible result. This scenario confirms that all the basic components—input file generation, process execution, and output parsing—are correctly integrated and functioning as expected. It serves as the baseline for all other tests. A simple, well-understood material like a bulk Silicon (Si) crystal provides a perfect test case, as its properties are widely known, and DFT calculations on it are typically stable and fast.

**UAT Steps in Jupyter Notebook:**
1.  **Setup:** The notebook will begin by importing the necessary libraries (`ase`, `pathlib`) and the `DFTFactory` class from the project's source code. It will also define the path to the ASE database file that will be used for the tests.
2.  **Create Structure:** An `ase.Atoms` object for a standard 2-atom conventional cell of Silicon will be created. The cell parameters and atomic positions will be explicitly defined. The notebook cell will display a 3D visualisation of this structure to provide clear visual confirmation of the input.
3.  **Instantiate Factory:** An instance of the `DFTFactory` will be created. The configuration passed to it will be minimal, primarily specifying the location of the Quantum Espresso executable.
4.  **Execute Calculation:** The `dft_factory.run(si_atoms)` method will be called. The notebook cell will print a message like "Running DFT calculation for Si...". The execution will be timed to ensure it completes within an expected timeframe (e.g., under 60 seconds for a simple system).
5.  **Display Results:** Upon successful completion, the returned `DFTResult` object will be captured. The notebook will then display its contents in a clean, readable format. This includes printing the calculated total energy (in eV), the forces on each atom (as a NumPy array), and the virial stress tensor.
6.  **Assertion:** The final step will involve a simple check to confirm the results are reasonable. The test will assert that the calculated energy is negative (as expected for a bound system) and within a known, wide range for bulk silicon. It will also verify that the forces are very close to zero, as the input structure is already at its equilibrium position. This demonstrates that the calculation was physically meaningful.

---

### **Scenario UAT-C1-002: Automatic Parameter Heuristics Verification**

**(Min 300 words)**

**Description:**
A core value proposition of the `DFTFactory` is its ability to automatically determine sensible calculation parameters, removing this burden from the user. This UAT scenario is designed to amaze the user by demonstrating this "magic" in action. It will show that by providing two vastly different atomic structures—a simple metal and a more complex, magnetic alloy—the factory intelligently adapts the DFT parameters to suit each case without requiring any specific instructions. This directly validates the system's codified domain knowledge.

**UAT Steps in Jupyter Notebook:**
1.  **Setup:** As before, the `DFTFactory` is imported. This test will also require a way to "spy" on the parameters being used. A temporary logging configuration will be added to the notebook to print the generated DFT input file to the screen.
2.  **Case 1: Simple Metal (Aluminium):**
    -   An `ase.Atoms` object for a face-centred cubic (FCC) Aluminium (Al) cell is created.
    -   The `dft_factory.run(al_atoms)` method is called.
    -   The test will then capture and display the generated Quantum Espresso input file. The user will be shown that the system automatically selected an appropriate k-point mesh (e.g., 8x8x8) and correctly identified the system as a metal by including `'smearing'` parameters.
3.  **Case 2: Magnetic Alloy (Iron):**
    -   An `ase.Atoms` object for a body-centred cubic (BCC) Iron (Fe) cell is created.
    -   The `dft_factory.run(fe_atoms)` method is called.
    -   Again, the generated input file is displayed. The user will be shown two key differences:
        1.  The system has correctly identified Iron as a magnetic element and has automatically enabled a spin-polarised calculation (`nspin = 2`).
        2.  It has also set an initial magnetic moment (`starting_magnetization(1) = 0.5`) to ensure the calculation converges to the correct magnetic state.
4.  **Assertion:** The notebook will explicitly point out these automatically generated parameters, contrasting the two cases. The success of this scenario is not in the final energy value but in demonstrating that the system's internal heuristics are working correctly and adapting to the chemical nature of the input structure. This provides confidence that the factory is not just a simple wrapper but an intelligent agent.

---

## 2. Behavior Definitions

This section defines the expected behaviors of the system in the Gherkin-style Given/When/Then format. These definitions provide a clear and unambiguous specification of the system's requirements for each scenario.

### **UAT-C1-001: Successful "Happy Path" Calculation**

```gherkin
Feature: Basic DFT Calculation
  As a materials science researcher,
  I want to calculate the energy and forces of an atomic structure,
  So that I can obtain its fundamental physical properties.

  Scenario: Calculate properties for a standard Silicon crystal
    Given a valid atomic structure for a 2-atom Silicon conventional cell
    And a correctly configured DFT Factory
    When I run a DFT calculation for the Silicon structure
    Then the process should complete successfully within 90 seconds
    And the returned result should contain a total energy, forces, and stress
    And the total energy should be a negative floating-point number
    And the forces on each atom should be close to zero (e.g., magnitude < 1e-4 eV/Angstrom)
```

### **UAT-C1-002: Automatic Parameter Heuristics Verification**

```gherkin
Feature: Intelligent DFT Parameter Generation
  As a user with minimal DFT expertise,
  I want the system to automatically choose correct and safe DFT parameters,
  So that I can run reliable calculations for different types of materials.

  Scenario: System automatically detects a metal and applies smearing
    Given an atomic structure for a metallic element like Aluminium
    And a DFT Factory with basic configuration
    When I run a DFT calculation for the Aluminium structure
    Then the system should generate a DFT input file
    And that file must contain parameters for metallic smearing (e.g., occupations = 'smearing' and degauss > 0).

  Scenario: System automatically detects a magnetic element and enables spin-polarization
    Given an atomic structure for a magnetic element like Iron
    And a DFT Factory with basic configuration
    When I run a DFT calculation for the Iron structure
    Then the system should generate a DFT input file
    And that file must contain the parameter to enable spin-polarization (e.g., nspin = 2)
    And that file must contain a non-zero starting magnetization.
```

### **UAT-C1-003: Resilience to Convergence Failure**

```gherkin
Feature: Automated Error Recovery
  As a researcher running a large-scale automated workflow,
  I want the system to automatically recover from common DFT convergence errors,
  So that my workflow does not crash and require manual intervention.

  Scenario: Calculation fails to converge but recovers by adjusting parameters
    Given a computationally "difficult" atomic structure known to cause convergence issues
    And a DFT Factory configured with a maximum of 3 retry attempts
    When I run a DFT calculation for the difficult structure
    Then the system should initially fail the first DFT run
    And the system should log that it is attempting a recovery (e.g., "Convergence failed. Retrying with modified parameters...")
    And the system should automatically modify a relevant parameter (e.g., reduce 'mixing_beta')
    And the system should eventually succeed within the allowed retry attempts
    And the final result returned should be a valid DFT result object.
```

### **UAT-C1-004: Data Persistence and Retrieval**

```gherkin
Feature: Storing Calculation Results
  As a user running multiple calculations,
  I want the system to save every successful result to a database,
  So that I have a persistent and queryable record of my training data.

  Scenario: Save a successful DFT result to an ASE database
    Given a valid atomic structure and its corresponding successful DFT result
    And the path to a new, empty ASE database file
    When I use the database utility to save the result
    Then the database file should be created on disk
    And the database should contain exactly one entry
    And when I read that entry back from the database, its stored energy and forces should match the original DFT result.
```
