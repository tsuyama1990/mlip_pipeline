# Cycle 02 UAT: Data Generation & DFT Integration

## 1. Test Scenarios

### Scenario 02: "The First Principles"
**Priority**: High
**Description**: Verify that the system can generate valid Quantum Espresso input files and parse the output correctly. This tests the `DFTManager` and `OracleConfig`.

**Pre-conditions**:
-   A valid `pseudo_dir` with at least one UPF file (e.g., Si.upf).
-   `pw.x` is accessible in the path (or a mock script that mimics it).

**Steps**:
1.  User creates a `config.yaml` with `oracle.type: qe`.
2.  User provides a `structure.xyz` file containing a Silicon crystal.
3.  User runs `pyacemaker compute --structure structure.xyz --config config.yaml` (New CLI command for debugging).

**Expected Outcome**:
-   A directory `dft_calc/` is created.
-   `pw.in` file exists and contains correct `ATOMIC_SPECIES`, `CELL_PARAMETERS`, and `K_POINTS`.
-   The command outputs the calculated Energy and Forces to the console.

## 2. Behavior Definitions (Gherkin)

```gherkin
Feature: DFT Calculation Management

  Scenario: Generate Input and Run QE
    Given a Silicon crystal structure
    And an Oracle configuration with "kspacing: 0.04"
    When I request a single point calculation
    Then a "pw.in" file should be generated
    And the k-point grid should be approximately "4 4 4" (for a standard cell)
    And if the calculation fails with "convergence not achieved"
    Then the Oracle should retry with "mixing_beta" reduced by 50%
```
