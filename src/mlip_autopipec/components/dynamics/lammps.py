import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from mlip_autopipec.components.dynamics.base import BaseDynamics
from mlip_autopipec.components.dynamics.lammps_driver import LAMMPSDriver
from mlip_autopipec.domain_models.config import LAMMPSDynamicsConfig
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure

logger = logging.getLogger(__name__)


class LAMMPSDynamics(BaseDynamics):
    """
    LAMMPS implementation of the Dynamics component.
    """

    def __init__(self, config: LAMMPSDynamicsConfig) -> None:
        super().__init__(config)
        self.config: LAMMPSDynamicsConfig = config

    @property
    def name(self) -> str:
        return self.config.name

    def explore(
        self,
        potential: Potential,
        start_structures: Iterable[Structure],
        workdir: Path | None = None,
        physics_baseline: dict[str, Any] | None = None,
    ) -> Iterator[Structure]:
        """
        Explore the PES using LAMMPS MD simulations.
        """
        base_workdir = workdir or Path.cwd()

        # We iterate through start structures
        # Note: enumerate consumes the iterable.
        for idx, structure in enumerate(start_structures):
            run_dir = base_workdir / f"lammps_run_{idx:05d}"
            driver = LAMMPSDriver(workdir=run_dir, binary="lmp")

            try:
                driver.write_input_files(
                    structure, potential, self.config, physics_baseline=physics_baseline
                )
                driver.run_md()
                result = driver.parse_log()

                if result["halted"]:
                    logger.info(f"Structure {idx} halted at step {result['final_step']}")
                    try:
                        final_struct = driver.read_dump(potential)
                        final_struct.uncertainty = 100.0  # Flag as uncertain
                        final_struct.tags["provenance"] = "dynamics_halted"

                        yield final_struct

                    except Exception as e:
                        logger.exception(f"Failed to recover halted structure {idx}: {e}") # noqa: TRY401
                else:
                    logger.debug(f"Structure {idx} finished without halt.")

            except Exception:
                logger.exception(f"LAMMPS run failed for structure {idx}")
                # Continue to next structure
                continue

    def __repr__(self) -> str:
        return f"<LAMMPSDynamics(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"LAMMPSDynamics({self.name})"
