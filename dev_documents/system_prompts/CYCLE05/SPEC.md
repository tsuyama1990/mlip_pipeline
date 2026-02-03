# Specification: Cycle 05 - The Validator (Quality Assurance)

## 1. Summary

Cycle 05 is about building "Trust". An MLIP that has low RMSE on the training set can still be physically nonsensical (e.g., have imaginary phonon modes or negative bulk modulus). This cycle implements the **Validation Suite**, a series of rigorous physical tests that every potential must pass before it is accepted.

The output of this cycle is the `Validator` module. It runs post-training and acts as a gatekeeper. If the potential fails critical tests (like stability), the Orchestrator flags it, and potentially triggers a "Repair" strategy (though repair logic is part of the Adaptive Policy, the *detection* happens here). Furthermore, this cycle implements the **Report Generator**, which creates a human-readable HTML dashboard visualizing the potential's performance.

## 2. System Architecture

### 2.1 File Structure

Files to be created or modified (bold):

```
mlip-autopipec/
├── src/
│   └── mlip_autopipec/
│       ├── config/
│       │   └── **config_model.py**         # Update with ValidationConfig
│       ├── validation/
│       │   ├── __init__.py
│       │   ├── **runner.py**               # Main execution logic
│       │   ├── **metrics.py**              # Individual tests (Phonon, Elastic)
│       │   └── **report_generator.py**     # HTML creation
│       └── utils/
│           └── **plotting.py**             # Matplotlib/Plotly helpers
├── tests/
│   ├── unit/
│   │   └── **test_metrics.py**
│   └── integration/
│       └── **test_validation_flow.py**
└── templates/
    └── **report_template.html**            # Jinja2 template
```

### 2.2 Component Blueprints

#### `src/mlip_autopipec/validation/runner.py`

```python
class ValidationRunner:
    def validate(self, potential_path: Path, work_dir: Path) -> ValidationResult:
        """
        Runs all enabled tests.
        Returns a result object containing metrics and a global Pass/Fail boolean.
        """
        # Load potential and structure...
        potential = ...
        structure = ...

        results = []
        if self.config.check_phonons:
            results.append(PhononValidator.run(potential, structure))
        if self.config.check_elastic:
            results.append(ElasticValidator.run(potential, structure))

        return self._aggregate_results(results, work_dir)
```

#### `src/mlip_autopipec/validation/metrics.py`

```python
class PhononValidator:
    @staticmethod
    def run(potential, structure) -> MetricResult:
        """
        Calculates force constants using Phonopy.
        Checks for imaginary frequencies (freq < -0.01 THz).
        """
        # ... logic to interface with phonopy API ...
```

## 3. Design Architecture

### 3.1 Pluggable Metrics
The architecture treats each validation test as a plugin.
*   **Interface**: Each validator implements a `run(potential, structure) -> MetricResult` method.
*   **MetricResult**: A Pydantic model containing:
    *   `name`: str
    *   `passed`: bool
    *   `score`: float (e.g., RMSE, Max Imaginary Freq)
    *   `details`: dict (e.g., the full tensor)
    *   `plot_path`: Optional[str]

### 3.2 HTML Reporting
We use **Jinja2** templating to generate reports.
*   **Why HTML?**: It's portable, interactive (with Plotly), and can be hosted on static servers (GitHub Pages) for team collaboration.
*   **Content**: The report displays the Config, Training Curves, and the results of all Validation tests side-by-side with the Training Set distribution.

## 4. Implementation Approach

### Step 1: Metric Infrastructure
1.  Define the `MetricResult` and `ValidationResult` models.
2.  Implement `ValidationConfig` to allow users to toggle specific tests (e.g., `check_phonons: false` for liquids).

### Step 2: Elastic Validator
1.  Implement `ElasticValidator`.
2.  Use ASE's `ElasticModel` or implement a simple Finite Difference method: deform cell by $\pm 1\%$, measure stress, fit stiffness tensor $C_{ij}$.
3.  Check Born stability conditions based on the crystal symmetry.

### Step 3: Phonon Validator
1.  Implement `PhononValidator`.
2.  Use `phonopy` library (if available) or call `phonopy` CLI.
3.  Calculate the band structure on a standard path (e.g., G-X-M-G).
4.  Check for $\omega < -\epsilon$.

### Step 4: Report Generation
1.  Create a nice HTML template using Bootstrap or simple CSS.
2.  Implement `ReportGenerator`. It should accept the `ValidationResult` and render the HTML.
3.  Integrate `matplotlib` or `plotly` to save figures to `active_learning/iter_XXX/plots/`.

## 5. Test Strategy

### 5.1 Unit Testing Approach (Min 300 words)
*   **Elastic Logic**: We will create a test using a perfect Lennard-Jones FCC crystal (where analytical elastic constants are known). We will verify that `ElasticValidator` computes $C_{11}, C_{12}, C_{44}$ within 5% of the analytical values.
*   **Stability Check**: We will construct a "broken" stiffness matrix (e.g., $C_{11} < 0$) and assert that the validator returns `passed=False`.
*   **Reporting**: We will feed a dummy `ValidationResult` into the `ReportGenerator` and assert that the output string contains the expected HTML tags (`<table>`, `<h1>`) and the specific values from the result.

### 5.2 Integration Testing Approach (Min 300 words)
*   **Full Validation Run**: We will set up an integration test where the `ValidationRunner` is called on a dummy potential.
    *   Since we can't easily train a potential in the test environment, we will mock the `potential` object (or use a simple calculator like EMT/LJ disguised as the potential).
    *   We will verify that `phonopy` (if installed) runs without crashing. If `phonopy` is not installed, the system should skip the test gracefully (logging a warning) rather than crashing the whole pipeline.
*   **Visual Inspection**: For UAT, we will generate a sample report and manually inspect it to ensure the plots are readable and the layout is correct.
