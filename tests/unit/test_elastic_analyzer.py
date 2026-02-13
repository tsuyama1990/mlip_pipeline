from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.validator.elastic import ElasticAnalyzer, ElasticResults


class MockElasticCalculator(Calculator):
    """Mock calculator that returns stress based on strain and known Cij."""
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"] # type: ignore[misc]

    def __init__(self, c11: float, c12: float, c44: float, cell0: np.ndarray) -> None:
        super().__init__() # type: ignore[no-untyped-call]
        self.c11 = c11
        self.c12 = c12
        self.c44 = c44
        self.cell0 = cell0
        self.results = {}

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes
    ) -> None:
        super().calculate(atoms, properties, system_changes) # type: ignore[no-untyped-call]

        if atoms is None:
             msg = "Atoms must be provided"
             raise ValueError(msg)

        cell = atoms.get_cell() # type: ignore[no-untyped-call]
        # simple strain calculation assuming no rotation
        strain = np.dot(cell, np.linalg.inv(self.cell0)) - np.eye(3)
        eps = 0.5 * (strain + strain.T)

        stress = np.zeros(6)

        # Cubic system (Voigt: xx, yy, zz, yz, xz, xy)
        stress[0] = self.c11 * eps[0,0] + self.c12 * eps[1,1] + self.c12 * eps[2,2]
        stress[1] = self.c12 * eps[0,0] + self.c11 * eps[1,1] + self.c12 * eps[2,2]
        stress[2] = self.c12 * eps[0,0] + self.c12 * eps[1,1] + self.c11 * eps[2,2]

        # Shear components (engineering shear strain gamma = 2*epsilon)
        stress[3] = self.c44 * 2 * eps[1,2]
        stress[4] = self.c44 * 2 * eps[0,2]
        stress[5] = self.c44 * 2 * eps[0,1]

        # ASE returns -stress tensor (virial stress convention often used)

        self.results["stress"] = -stress
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))

def test_elastic_analyzer_cubic() -> None:
    """Test elastic constants calculation for a cubic system."""
    # Target values in GPa
    c11_target = 200.0
    c12_target = 100.0
    c44_target = 50.0

    # Create a mock structure (Cu fcc)
    # 1 eV/A^3 = 160.2 GPa
    # So 200 GPa = 1.24 eV/A^3
    conv = 160.21766208
    c11_ev = c11_target / conv
    c12_ev = c12_target / conv
    c44_ev = c44_target / conv

    cell0 = np.eye(3) * 3.6
    # ase.Atoms is typed or has type stubs?
    # If using recent ASE, it might.
    # The previous error said "Unused type ignore", so let's remove it.
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=cell0, pbc=True)

    potential = MagicMock(spec=Potential)
    potential.path = Path("mock_potential.pot")

    calc = MockElasticCalculator(c11_ev, c12_ev, c44_ev, cell0)

    with patch("mlip_autopipec.validator.elastic.MLIPCalculatorFactory") as MockFactoryCls:
        mock_instance = MockFactoryCls.return_value
        mock_instance.create.return_value = calc

        analyzer = ElasticAnalyzer()
        results = analyzer.calculate_elastic_constants(atoms, potential)

        assert isinstance(results, ElasticResults)
        assert pytest.approx(c11_target, rel=0.1) == results.C11
        assert pytest.approx(c12_target, rel=0.1) == results.C12
        assert pytest.approx(c44_target, rel=0.1) == results.C44

        B_expected = (c11_target + 2*c12_target) / 3
        assert pytest.approx(B_expected, rel=0.1) == results.B
