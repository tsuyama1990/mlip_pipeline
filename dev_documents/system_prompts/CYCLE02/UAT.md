# Cycle 02 UAT: Data Management & Structure Generation

## 1. Test Scenarios

### Scenario 01: "The Architect" (Structure Generation)
**Priority**: High
**Description**: Verify that the system can generate physically reasonable structures for a given composition.

**Gherkin Definition**:
```gherkin
GIVEN a generator configuration with composition "MgO"
WHEN I execute the structure generation command
THEN the system should produce 10 valid structures
AND each structure should contain Mg and O atoms
AND the minimum distance between atoms should be > 1.5 Angstrom
```

### Scenario 02: "The Librarian" (Database Persistence)
**Priority**: Critical
**Description**: Verify that generated structures can be saved to disk and reloaded without data loss.

**Gherkin Definition**:
```gherkin
GIVEN a list of 10 generated structures
WHEN I save them to "dataset.pckl.gzip"
AND I load them back from "dataset.pckl.gzip"
THEN the loaded list should contain 10 structures
AND the atomic positions should match the original ones
```

## 2. Verification Steps

1.  **Unit Test Execution**: Run `pytest tests/infrastructure/generator`.
2.  **Manual Script**: Create `scripts/test_gen.py` that:
    *   Instantiates `RandomStructureGenerator`.
    *   Generates 5 structures.
    *   Prints their details (formula, volume).
    *   Saves them using `Dataset`.
    *   Loads and asserts equality.
3.  **Visual Inspection**: Use `ase gui` or `view_image` (if implemented) to check a generated structure.
