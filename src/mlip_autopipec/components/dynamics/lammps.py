import concurrent.futures
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


def _run_single_lammps_simulation(
    idx: int,
    structure: Structure,
    potential: Potential,
    config: LAMMPSDynamicsConfig,
    base_workdir: Path,
    physics_baseline: dict[str, Any] | None,
) -> Structure | None:
    """
    Run a single LAMMPS simulation in a separate process.
    Returns the halted structure if found, else None.
    """
    try:
        run_dir = base_workdir / f"lammps_run_{idx:05d}"
        driver = LAMMPSDriver(workdir=run_dir, config=config, binary="lmp")

        driver.write_input_files(
            structure, potential, physics_baseline=physics_baseline
        )
        driver.run_md()
        result = driver.parse_log()

        if result["halted"]:
            logger.info(f"Structure {idx} halted at step {result['final_step']}")
            try:
                final_struct = driver.read_dump(potential)
                final_struct.uncertainty = 100.0  # Flag as uncertain
                final_struct.tags["provenance"] = "dynamics_halted"
            except Exception:
                logger.exception(f"Failed to recover halted structure {idx}")
                return None
            else:
                return final_struct
        else:
            logger.debug(f"Structure {idx} finished without halt.")
            return None
    except Exception:
        logger.exception(f"LAMMPS run failed for structure {idx}")
        return None


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
        cycle: int = 0,
        metrics: dict[str, Any] | None = None,
    ) -> Iterator[Structure]:
        """
        Explore the PES using LAMMPS MD simulations.
        """
        base_workdir = workdir or Path.cwd()
        max_workers = self.config.max_workers

        # Update config based on schedule if necessary
        run_config = self.config
        if self.config.temperature_schedule and cycle in self.config.temperature_schedule:
            target_temp = self.config.temperature_schedule[cycle]
            logger.info(f"Using scheduled temperature {target_temp}K for cycle {cycle}")
            # Create a copy with updated temperature
            # Since Pydantic V2 we use model_copy(update=...)
            run_config = self.config.model_copy(update={"temperature": target_temp})

        # We submit tasks to executor.
        # Since start_structures is an iterable, we iterate and submit.
        # To avoid submitting infinite tasks if iterable is infinite (unlikely here but good practice),
        # we usually just consume it.

        # We also need to handle potential pickling. Potential object should be pickleable.

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, structure in enumerate(start_structures):
                future = executor.submit(
                    _run_single_lammps_simulation,
                    idx,
                    structure,
                    potential,
                    run_config,
                    base_workdir,
                    physics_baseline,
                )
                futures.append(future)

            # Yield results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        yield result
                except Exception:
                    logger.exception("Simulation task failed")

    def __repr__(self) -> str:
        return f"<LAMMPSDynamics(name={self.name}, config={self.config})>"

    def __str__(self) -> str:
        return f"LAMMPSDynamics({self.name})"
