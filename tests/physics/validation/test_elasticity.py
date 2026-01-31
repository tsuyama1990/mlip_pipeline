import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from mlip_autopipec.domain_models.config import ValidationConfig
from mlip_autopipec.physics.validation.elasticity import ElasticityValidator


class ElasticityMockCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, ref_cell, C_matrix):
        super().__init__()
        self.ref_cell = ref_cell
        self.C = C_matrix  # 6x6 in GPa

    def calculate(self, atoms=None, properties=["stress"], system_changes=None):
        super().calculate(atoms, properties, system_changes)

        # Calculate strain relative to ref_cell
        cell = atoms.get_cell()

        # h_new = h_old @ (I + eps) => (I + eps) = h_old_inv @ h_new
        # If cell vectors are rows.

        deformation = np.linalg.inv(self.ref_cell) @ cell
        eps_tensor = deformation - np.eye(3)
        # Symmetrize
        eps_tensor = 0.5 * (eps_tensor + eps_tensor.T)

        # Voigt strain
        strain_voigt = np.array(
            [
                eps_tensor[0, 0],
                eps_tensor[1, 1],
                eps_tensor[2, 2],
                2 * eps_tensor[1, 2],
                2 * eps_tensor[0, 2],
                2 * eps_tensor[0, 1],
            ]
        )

        stress_gpa = self.C @ strain_voigt
        self.results["stress"] = stress_gpa / 160.21766208
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))


def test_elasticity_validation_success(tmp_path):
    """Test that Elasticity validation passes for stable stiffness matrix."""
    structure = Atoms(
        "Si2",
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )

    # Stable Cubic Matrix
    # C11=100, C12=20, C44=40
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = 100
    C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = 20
    C[3, 3] = C[4, 4] = C[5, 5] = 40

    calc = ElasticityMockCalculator(structure.get_cell(), C)
    config = ValidationConfig(elastic_strain=0.01)

    validator = ElasticityValidator(structure, calc, config, tmp_path, "test_pot")
    result = validator.validate()

    assert result.overall_status == "PASS"
    assert result.metrics[0].name == "Elastic Stability (Min Eigenvalue)"
    assert result.metrics[0].value > 0
    # Check if calculated C matches input C (approx)
    # Since we use finite difference on linear stress-strain, it should be exact.

    # But wait, min eigenvalue of C:
    # Eigenvalues of cubic C: C11+2C12 (140), C11-C12 (80, twice), C44 (40, thrice)
    # Min is 40.
    assert np.isclose(result.metrics[0].value, 40.0, atol=1.0)
    assert (tmp_path / "elasticity_matrix.png").exists()


def test_elasticity_validation_failure(tmp_path):
    """Test that Elasticity validation fails for unstable stiffness matrix."""
    structure = Atoms(
        "Si2",
        positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )

    # Unstable Matrix (Negative Eigenvalue)
    # C11=10, C12=50 -> C11-C12 = -40
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = 10
    C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = 50
    C[3, 3] = C[4, 4] = C[5, 5] = 40

    calc = ElasticityMockCalculator(structure.get_cell(), C)
    config = ValidationConfig(elastic_strain=0.01)

    validator = ElasticityValidator(structure, calc, config, tmp_path, "test_pot")
    result = validator.validate()

    assert result.overall_status == "FAIL"
    assert result.metrics[0].passed is False
    assert result.metrics[0].value < 0
