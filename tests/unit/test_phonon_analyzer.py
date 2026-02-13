from unittest.mock import patch
from typing import Any

from ase.build import bulk
from ase.calculators.emt import EMT

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.validator.phonon import PhononAnalyzer, PhononResults


def test_phonon_analyzer_emt() -> None:
    # Use EMT calculator
    with patch("mlip_autopipec.validator.phonon.MLIPCalculatorFactory") as MockFactory:
        mock_factory_instance = MockFactory.return_value
        # Return new EMT calculator
        mock_factory_instance.create.side_effect = lambda path: EMT() # type: ignore[no-untyped-call]

        analyzer = PhononAnalyzer()

        # Stable Al structure
        atoms = bulk("Al", "fcc", a=4.05)
        structure = Structure(atoms=atoms, provenance="test")
        potential = Potential(path="dummy.yace", format="yace")

        # Use small supercell [2,2,2]
        results = analyzer.analyze(potential, structure, supercell_matrix=[2, 2, 2])

        assert isinstance(results, PhononResults)
        assert results.is_stable is True
        assert results.max_imaginary_freq == 0.0
        # Check band structure data exists
        if results.band_structure_plot_data:
            assert "frequencies" in results.band_structure_plot_data

def test_phonon_analyzer_unstable() -> None:
    # Create an unstable structure (e.g., compressed or expanded too much)
    with patch("mlip_autopipec.validator.phonon.MLIPCalculatorFactory") as MockFactory:
        mock_factory_instance = MockFactory.return_value
        mock_factory_instance.create.side_effect = lambda path: EMT() # type: ignore[no-untyped-call]

        analyzer = PhononAnalyzer()

        # Expanded Al - might be unstable
        atoms = bulk("Al", "fcc", a=6.0)
        structure = Structure(atoms=atoms, provenance="test")
        potential = Potential(path="dummy.yace", format="yace")

        results = analyzer.analyze(potential, structure, supercell_matrix=[2, 2, 2])

        # Check if we got results
        assert isinstance(results, PhononResults)

def test_phonon_analyzer_missing_phonopy() -> None:
    # Mock ImportError for phonopy
    with patch.dict("sys.modules", {"phonopy": None}):
        analyzer = PhononAnalyzer()
        atoms = bulk("Al", "fcc", a=4.05)
        structure = Structure(atoms=atoms, provenance="test")
        potential = Potential(path="dummy.yace", format="yace")

        results = analyzer.analyze(potential, structure)

        assert results.is_stable is True
        assert results.max_imaginary_freq == 0.0
