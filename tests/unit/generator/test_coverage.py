from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.models import SystemConfig
from mlip_autopipec.config.schemas.core import TargetSystem
from mlip_autopipec.config.schemas.generator import GeneratorConfig, SQSConfig
from mlip_autopipec.exceptions import GeneratorError
from mlip_autopipec.generator.builder import StructureBuilder
from mlip_autopipec.generator.transformations import apply_rattle, apply_strain

# --- StructureBuilder Coverage ---

def test_builder_generation_flow():
    target = TargetSystem(
        name="Al",
        elements=["Al"],
        composition={"Al": 1.0},
        crystal_structure="fcc"
    )
    config = SystemConfig(target_system=target)
    builder = StructureBuilder(config)

    with patch("mlip_autopipec.generator.builder.bulk") as mock_bulk:
        mock_bulk.return_value = Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)

        # Act
        structures = list(builder.build())

        # Assert
        assert len(structures) > 0
        mock_bulk.assert_called()

def test_builder_bulk_fallback():
    # If target system fails to build (e.g. invalid structure string), it should fallback
    target = TargetSystem(name="Al", elements=["Al"], composition={"Al": 1.0}, crystal_structure="invalid_struct")

    gen_config = GeneratorConfig(sqs=SQSConfig(enabled=False))
    config = SystemConfig(target_system=target, generator_config=gen_config)
    builder = StructureBuilder(config)

    # We mock bulk to fail on first call, succeed on second (fallback)
    # The code calls bulk(primary, structure_type) first.
    # Then fallback calls bulk("Al", "fcc", ...)

    with patch("mlip_autopipec.generator.builder.bulk") as mock_bulk:
        def side_effect(element, structure=None, **kwargs):
            if structure == "invalid_struct":
                raise ValueError("Invalid")
            return Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)

        mock_bulk.side_effect = side_effect

        structures = list(builder.build())
        assert len(structures) > 0
        # Verify fallback was used
        assert mock_bulk.call_count >= 2

# --- Transformations Coverage ---

def test_apply_strain_exception():
    atoms = Atoms("H")
    strain = np.eye(3)

    # Mock atoms.copy() to raise exception
    with patch.object(atoms, "copy", side_effect=Exception("Copy failed")):
        with pytest.raises(GeneratorError) as exc:
            apply_strain(atoms, strain)
        assert "Failed to apply strain" in str(exc.value)


def test_apply_rattle_exception():
    atoms = Atoms("H")

    # Mock atoms.copy() to raise exception
    with patch.object(atoms, "copy", side_effect=Exception("Copy failed")):
        with pytest.raises(GeneratorError) as exc:
            apply_rattle(atoms, 0.1)
        assert "Failed to apply rattle" in str(exc.value)
