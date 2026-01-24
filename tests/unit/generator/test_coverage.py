from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.config.schemas.core import TargetSystem
from mlip_autopipec.config.schemas.generator import GeneratorConfig
from mlip_autopipec.exceptions import GeneratorError
from mlip_autopipec.generator.builder import StructureBuilder
from mlip_autopipec.generator.transformations import apply_rattle, apply_strain

# --- StructureBuilder Coverage ---

def test_builder_molecule_generation() -> None:
    target = TargetSystem(
        name="molecule_H2O",
        elements=["H", "O"],
        composition={"H": 2/3, "O": 1/3},
    )

    config = SystemConfig(target_system=target)
    builder = StructureBuilder(config)

    # Mock ase.build.molecule to return a fake molecule
    with patch("mlip_autopipec.generator.builder.molecule") as mock_mol:
        mock_mol.return_value = Atoms("H2O", positions=[[0,0,0], [0,0,1], [0,1,0]])

        structures = list(builder.build())
        assert len(structures) >= 1
        assert len(structures[0]) == 3
        mock_mol.assert_called()

def test_builder_molecule_failure() -> None:
    target = TargetSystem(
        name="molecule_Invalid",
        elements=["H"],
        composition={"H": 1.0}
    )

    config = SystemConfig(target_system=target)
    builder = StructureBuilder(config)

    with patch("mlip_autopipec.generator.builder.molecule", side_effect=Exception("Mol fail")):
        # Should warn and return empty list
        structures = list(builder.build())
        assert len(structures) == 0

def test_builder_bulk_fallback() -> None:
    target = TargetSystem(
        name="Al",
        elements=["Al"],
        composition={"Al": 1.0}
    )
    # Disable SQS to avoid overwriting the fallback element
    gen_config = GeneratorConfig(sqs={"enabled": False})
    config = SystemConfig(target_system=target, generator_config=gen_config)
    builder = StructureBuilder(config)

    def bulk_side_effect(name, *args, **kwargs):
        if name == "Al":
            raise Exception("Bulk fail")
        return Atoms("Fe", positions=[[0,0,0]], cell=[2,2,2], pbc=True)

    with patch("mlip_autopipec.generator.builder.bulk", side_effect=bulk_side_effect):
        # Should fallback to Fe
        structures = list(builder.build())
        assert len(structures) > 0
        assert structures[0].symbols[0] == "Fe"

def test_builder_critical_failure() -> None:
    # Mock _generate_base to raise generic exception
    target = TargetSystem(
            name="Al",
            elements=["Al"],
            composition={"Al": 1.0}
    )
    config = SystemConfig(target_system=target)
    builder = StructureBuilder(config)

    with patch.object(builder, '_generate_base', side_effect=Exception("Critical")):
        with pytest.raises(GeneratorError) as excinfo:
            list(builder.build())
        assert "Structure generation failed" in str(excinfo.value)

# --- Transformations Coverage ---

def test_apply_strain_exception() -> None:
    atoms = Atoms("H")
    strain = np.eye(3)

    # Mock atoms.copy() to raise exception
    with patch.object(atoms, 'copy', side_effect=Exception("Copy failed")):
        with pytest.raises(GeneratorError) as exc:
            apply_strain(atoms, strain)
        assert "Failed to apply strain" in str(exc.value)

def test_apply_rattle_exception() -> None:
    atoms = Atoms("H")

    # Mock atoms.copy() to raise exception
    with patch.object(atoms, 'copy', side_effect=Exception("Copy failed")):
        with pytest.raises(GeneratorError) as exc:
            apply_rattle(atoms, 0.1)
        assert "Failed to apply rattle" in str(exc.value)
