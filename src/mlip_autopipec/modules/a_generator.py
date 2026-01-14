# Copyright (C) 2024-present by the LICENSE file authors.
#
# This file is part of MLIP-AutoPipe.
#
# MLIP-AutoPipe is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MLIP-AutoPipe is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MLIP-AutoPipe.  If not, see <https://www.gnu.org/licenses/>.
"""This module provides functions for generating atomic structures."""
import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.vibrations import Vibrations
from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs

from mlip_autopipec.schemas.system_config import SystemConfig


def _generate_alloy_sqs(config: SystemConfig) -> list[Atoms]:
    """Generates SQS for alloys, then applies strain and rattle."""
    composition = config.user_config.target_system.composition
    crystal_structure = config.user_config.target_system.crystal_structure
    supercell_size = config.generator_params.sqs_supercell_size

    lattice_constant = config.generator_params.lattice_constant or 3.6
    primitive_cell = bulk(next(iter(composition.keys())), crystal_structure, a=lattice_constant)

    cutoffs = config.generator_params.cutoffs or [6.0]
    chemical_symbols = list(composition.keys())
    cs = ClusterSpace(primitive_cell, cutoffs, chemical_symbols)

    sqs = generate_sqs(
        cluster_space=cs,
        max_size=int(np.prod(supercell_size)),
        target_concentrations=composition,
    )

    generated_structures = []
    for strain in config.generator_params.strain_magnitudes:
        strained_sqs = sqs.copy()  # type: ignore[no-untyped-call]
        strained_sqs.set_cell(strained_sqs.cell * (1 + strain), scale_atoms=True)

        rattled_sqs = strained_sqs.copy()  # type: ignore[no-untyped-call]
        rattled_sqs.rattle(stdev=config.generator_params.rattle_standard_deviation, seed=42)

        generated_structures.append(rattled_sqs)

    return generated_structures


def _generate_eos_strain(config: SystemConfig) -> list[Atoms]:
    """Generates a set of structures with varying volumetric strain."""
    composition = config.user_config.target_system.composition
    crystal_structure = config.user_config.target_system.crystal_structure
    lattice_constant = config.generator_params.lattice_constant or 5.43
    primitive_cell = bulk(
        next(iter(composition.keys())),
        crystal_structure,
        a=lattice_constant,
        cubic=True,
    )

    generated_structures = []
    for strain in config.generator_params.strain_magnitudes:
        strained_cell = primitive_cell.copy()
        strained_cell.set_cell(strained_cell.cell * (1 + strain), scale_atoms=True)
        generated_structures.append(strained_cell)
    return generated_structures


def _generate_nms(config: SystemConfig) -> list[Atoms]:
    """Generates structures by displacing atoms along normal modes."""
    composition = config.user_config.target_system.composition
    crystal_structure = config.user_config.target_system.crystal_structure
    primitive_cell = bulk(next(iter(composition.keys())), crystal_structure, a=3.6, cubic=True)

    # A real implementation would use a calculator to get forces
    # For now, we use the EMT calculator for testing
    from ase.calculators.emt import EMT

    primitive_cell.calc = EMT()

    vib = Vibrations(primitive_cell, name="dummy_vib")  # type: ignore[no-untyped-call]
    vib.run()

    generated_structures = []
    for disp, atoms in vib.iterdisplace():
        if disp.sign != 0:
            generated_structures.append(atoms.copy())
    return generated_structures


def _generate_melt_quench(config: SystemConfig) -> list[Atoms]:
    """Generates structures by melting and quenching the system."""
    composition = config.user_config.target_system.composition
    crystal_structure = config.user_config.target_system.crystal_structure
    atoms = bulk(next(iter(composition.keys())), crystal_structure, a=3.6, cubic=True)
    atoms.rattle(stdev=0.1)

    # A real implementation would use a calculator
    from ase.calculators.emt import EMT

    atoms.calc = EMT()

    # Melt
    dyn = Langevin(atoms, 5 * units.fs, 1000 * units.kB, 0.02)  # type: ignore[no-untyped-call]
    dyn.run(100)

    # Quench
    dyn = Langevin(atoms, 5 * units.fs, 10 * units.kB, 0.1)  # type: ignore[no-untyped-call]
    dyn.run(100)

    return [atoms]


def generate_structures(config: SystemConfig) -> list[Atoms]:
    """
    Generates a list of ASE Atoms objects based on the system configuration.

    Args:
        config: The system configuration.

    Returns:
        A list of generated ASE Atoms objects.

    Raises:
        ValueError: If the generation type is unknown or the parameters are invalid.
        TypeError: If the strain magnitudes are not a list of floats.
    """
    params = config.generator_params
    if not isinstance(params.sqs_supercell_size, list) or len(params.sqs_supercell_size) != 3:
        sqs_supercell_size_error = "sqs_supercell_size must be a list of three integers."
        raise ValueError(sqs_supercell_size_error)
    if not isinstance(params.strain_magnitudes, list):
        strain_magnitudes_error = "strain_magnitudes must be a list of floats."
        raise TypeError(strain_magnitudes_error)

    generation_type = config.user_config.generation_config.generation_type
    if generation_type == "alloy_sqs":
        return _generate_alloy_sqs(config)
    if generation_type == "eos_strain":
        return _generate_eos_strain(config)
    if generation_type == "nms":
        return _generate_nms(config)
    if generation_type == "melt_quench":
        return _generate_melt_quench(config)

    unknown_type_error = f"Unknown generation type: {generation_type}"
    raise ValueError(unknown_type_error)
