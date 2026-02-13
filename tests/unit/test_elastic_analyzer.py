from unittest.mock import patch

import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.validator.elastic import ElasticAnalyzer


def test_elastic_analyzer_lj():
    # Use LJ calculator (mocking factory to return LJ)
    with patch("mlip_autopipec.validator.elastic.MLIPCalculatorFactory") as MockFactory:
        mock_factory_instance = MockFactory.return_value
        # Return a new LJ calculator each time
        mock_factory_instance.create.side_effect = lambda path: LennardJones() # type: ignore[no-untyped-call]

        analyzer = ElasticAnalyzer(strain_magnitude=0.01)

        # Create FCC "X" structure with small lattice constant suitable for LJ (sigma=1.0)
        atoms = bulk("X", "fcc", a=1.6)
        structure = Structure(atoms=atoms, provenance="test")
        potential = Potential(path="dummy.yace", format="yace")

        results = analyzer.analyze(potential, structure)

        # Check if keys exist
        assert "C11" in results
        assert "C12" in results
        assert "C44" in results
        assert "bulk_modulus" in results
        assert "shear_modulus" in results

        # LJ should give non-zero elastic constants
        assert results["C11"] > 0.01
        assert results["C12"] > 0.01
        assert results["bulk_modulus"] > 0.01

        # Check consistency B = (C11 + 2C12)/3 (approximately for cubic)
        assert results["bulk_modulus"] == pytest.approx((results["C11"] + 2*results["C12"])/3, rel=1e-5)

def test_elastic_analyzer_relaxation_failure():
    # Test that analyzer proceeds even if relaxation fails
    with patch("mlip_autopipec.validator.elastic.MLIPCalculatorFactory") as MockFactory:
        mock_factory_instance = MockFactory.return_value
        mock_factory_instance.create.return_value = LennardJones() # type: ignore[no-untyped-call]

        # Mock LBFGS to raise exception
        with patch("mlip_autopipec.validator.elastic.LBFGS") as MockLBFGS:
            mock_opt = MockLBFGS.return_value
            mock_opt.run.side_effect = Exception("Relaxation exploded")

            analyzer = ElasticAnalyzer()
            atoms = bulk("X", "fcc", a=1.6)
            structure = Structure(atoms=atoms, provenance="test")
            potential = Potential(path="dummy.yace", format="yace")

            # Should not raise exception
            results = analyzer.analyze(potential, structure)
            assert "C11" in results
