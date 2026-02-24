import marimo

__generated_with = "0.1.0"
app = marimo.App(width="medium")


@app.cell
def setup_marimo():
    import marimo as mo
    return mo,


@app.cell
def intro_md(mo):
    return mo.md(
        """
        # PYACEMAKER Cycle 06 UAT: End-to-End MACE Distillation

        This tutorial demonstrates the full **MACE Knowledge Distillation** workflow using the **SN2 Reaction** scenario.

        **Workflow Steps:**
        1.  **Structure Generation**: Define a custom generator for the SN2 reaction pathway ($CH_3Cl + OH^- \\rightarrow CH_3OH + Cl^-$).
        2.  **Configuration**: Setup the `Orchestrator` with a hybrid Mock/Real configuration.
        3.  **Execution**: Run the 7-step pipeline (Direct Sampling -> Active Learning -> Surrogate Generation -> Labeling -> Training -> Delta Learning).
        4.  **Validation**: Verify that artifacts (potentials, datasets) are created and physics checks pass.

        The goal is to automatically train a potential that captures the transition state of the reaction by distilling knowledge from a MACE foundation model.
        """
    )


@app.cell
def setup_constants():
    import os
    import sys
    from loguru import logger

    # Configure Logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Mode Detection
    IS_MOCK = (
        os.environ.get("PYACEMAKER_MODE", "").upper() == "MOCK"
        or os.environ.get("CI", "false").lower() == "true"
        or os.environ.get("MOCK_MODE", "false").lower() == "true"
    )
    MODE_NAME = "MOCK" if IS_MOCK else "REAL"
    print(f"Running in {MODE_NAME} Mode")
    return IS_MOCK, MODE_NAME, logger


@app.cell
def generator_md(mo):
    return mo.md(
        """
        ## 1. Define Structure Generator

        We define a custom `SN2StructureGenerator` that interpolates structures along the reaction coordinate.
        This simulates the "Exploration" phase of the active learning loop.
        It generates a sequence of structures from reactant to product, providing a diverse initial pool for the distillation process.
        """
    )


@app.cell
def define_sn2_generator(IS_MOCK, logger):
    from typing import Iterator, Any, Iterable
    import numpy as np
    from ase.build import molecule
    from pyacemaker.core.interfaces import StructureGenerator
    from pyacemaker.core.base import ModuleResult as _ModuleResult, Metrics
    from pyacemaker.domain_models.models import (
        StructureMetadata,
        StructureStatus,
        MaterialDNA,
    )

    class SN2StructureGenerator(StructureGenerator):
        """Generates SN2 Reaction Pathway Structures (CH3Cl + OH- -> CH3OH + Cl-)."""

        def __init__(self, config):
            super().__init__(config)
            self.logger = logger
            self.rattle_amp = 0.1

        def run(self) -> _ModuleResult:
            return _ModuleResult(status="success", metrics=Metrics())

        def get_strategy_info(self) -> dict[str, Any]:
            return {
                "strategy": "sn2_custom",
                "parameters": {"rattle_amp": self.rattle_amp},
            }

        def generate_initial_structures(self) -> Iterator[StructureMetadata]:
            # Provide basic endpoints
            yield from self._generate_path(n_points=2)

        def generate_direct_samples(
            self, n_samples: int, objective: str = "maximize_entropy"
        ) -> Iterator[StructureMetadata]:
            self.logger.info(f"Generating {n_samples} SN2 pathway structures.")
            yield from self._generate_path(n_samples)

        def generate_local_candidates(
            self,
            seed_structure: StructureMetadata,
            n_candidates: int,
            cycle: int = 1,
        ) -> Iterator[StructureMetadata]:
            atoms = seed_structure.features.get("atoms")
            if not atoms:
                return
            for _ in range(n_candidates):
                new_atoms = atoms.copy()
                new_atoms.rattle(self.rattle_amp)
                yield StructureMetadata(
                    features={"atoms": new_atoms},
                    material_dna=seed_structure.material_dna,
                    status=StructureStatus.NEW,
                    tags=["candidate", "local"],
                )

        def generate_batch_candidates(
            self,
            seed_structures: Iterable[StructureMetadata],
            n_candidates_per_seed: int,
            cycle: int = 1,
        ) -> Iterator[StructureMetadata]:
            # Simple rattle for local candidates
            for seed in seed_structures:
                yield from self.generate_local_candidates(
                    seed, n_candidates_per_seed, cycle
                )

        def _generate_path(self, n_points: int):
            # 1. Define Reactant: CH3Cl + OH-
            try:
                mol1 = molecule("CH3Cl")
                # Center C at 0
                c_idx = [a.index for a in mol1 if a.symbol == "C"][0]
                mol1.translate(-mol1.positions[c_idx])

                # Rotate so Cl is on Z axis (approx)
                cl_idx = [a.index for a in mol1 if a.symbol == "Cl"][0]
                vec = mol1.positions[cl_idx] - mol1.positions[c_idx]
                if np.linalg.norm(np.cross(vec, [0, 0, 1])) > 1e-3:
                    mol1.rotate(vec, [0, 0, 1])

                # OH-
                mol2 = molecule("OH")
                mol2.translate([0, 0, -4.0])

                reactant = mol1.copy()
                reactant.extend(mol2)
                reactant.set_cell([12, 12, 12])
                reactant.center()
                reactant.pbc = True
            except Exception as e:
                self.logger.warning(
                    f"Failed to build SN2 molecules: {e}. Using dummies."
                )
                from ase import Atoms

                reactant = Atoms(
                    "C", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True
                )

            for i in range(n_points):
                alpha = i / (n_points - 1) if n_points > 1 else 0.0
                frame = reactant.copy()

                try:
                    o_indices = [a.index for a in frame if a.symbol == "O"]
                    cl_indices = [a.index for a in frame if a.symbol == "Cl"]
                    if o_indices and cl_indices:
                        o_idx = o_indices[0]
                        cl_idx = cl_indices[0]

                        # Move O from -4 to -1.5
                        # Move Cl from ~1.8 to 4.0
                        current_o_z = frame.positions[o_idx][2]
                        target_o_z = -1.5
                        shift_o = (target_o_z - (-4.0)) * alpha
                        frame.positions[o_idx][2] += shift_o

                        frame.positions[cl_idx][2] += 2.0 * alpha
                except Exception:
                    pass

                # Add some noise
                if IS_MOCK:
                    frame.rattle(0.05)

                yield StructureMetadata(
                    features={"atoms": frame},
                    material_dna=MaterialDNA(
                        composition={"C": 1 / 7, "H": 4 / 7, "O": 1 / 7, "Cl": 1 / 7}
                    ),
                    status=StructureStatus.NEW,
                    tags=["sn2_path", f"alpha_{alpha:.2f}".replace(".", "p")],
                )

    return SN2StructureGenerator


@app.cell
def workflow_md(mo):
    return mo.md(
        """
        ## 2. Configure & Run Workflow

        The `run_uat_workflow` function initializes the `Orchestrator` with our custom generator and runs the full pipeline.
        It handles cleaning up the output directory and creating necessary mock files if needed.

        We configure the `Orchestrator` to use:
        - **MACE Distillation**: Enabled.
        - **Direct Sampling**: Uses our custom SN2 generator via `random` strategy (which delegates to generator).
        - **Oracle**: Mock or Real MACE.
        - **Trainer**: Mock or Real Pacemaker.
        """
    )


@app.cell
def define_workflow(IS_MOCK, SN2StructureGenerator):
    from pathlib import Path
    import shutil
    import yaml
    from pyacemaker.core.config_loader import load_config
    from pyacemaker.orchestrator import Orchestrator
    from pyacemaker.core.base import ModuleResult

    def run_uat_workflow():
        output_dir = Path("uat_sn2_reaction").absolute()

        # Safety Check: Ensure we are not deleting something important
        # We only delete if it looks like our output dir
        if output_dir.exists():
            if "uat_sn2_reaction" in output_dir.name:
                shutil.rmtree(output_dir)
            else:
                raise RuntimeError(f"Safety check failed: Refusing to delete {output_dir}")

        output_dir.mkdir(parents=True)

        config_dict = {
            "project": {"name": "SN2_UAT", "root_dir": str(output_dir)},
            "structure_generator": {"strategy": "random"},
            "oracle": {
                "mock": IS_MOCK,
                "dft": {
                    "code": "mock",
                    "pseudopotentials": {
                        "C": "C.upf",
                        "H": "H.upf",
                        "O": "O.upf",
                        "Cl": "Cl.upf",
                    },
                },
                "mace": {
                    "model_path": "medium" if not IS_MOCK else "mock_model.model",
                    "mock": IS_MOCK,
                },
            },
            "trainer": {"potential_type": "pace", "mock": IS_MOCK},
            "distillation": {
                "enable_mace_distillation": True,
                "step1_direct_sampling": {
                    "target_points": 10 if IS_MOCK else 50,
                    "objective": "random",
                },
                "step4_surrogate_sampling": {
                    "target_points": 20 if IS_MOCK else 500,
                    "method": "md",
                },
                "step7_pacemaker_finetune": {"enable": True, "weight_dft": 10.0},
                "pool_file": "step1_pool.xyz",
                "surrogate_file": "step4_surrogate.xyz",
                "surrogate_dataset_file": "step5_labeled.xyz",
            },
            "validator": {"phonon_supercell": [1, 1, 1], "eos_strain": 0.01},
        }

        # Write config with error handling
        config_path = output_dir / "config.yaml"
        try:
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f)
        except Exception as e:
            return ModuleResult(status="failed", error=f"Config write failed: {e}"), output_dir

        try:
            config = load_config(config_path)
        except Exception as e:
             return ModuleResult(status="failed", error=f"Config load failed: {e}"), output_dir

        # Create dummy pseudopotentials for validation if missing (Mock requirement)
        dummy_files = []
        if IS_MOCK:
            for el in ["C", "H", "O", "Cl"]:
                p = output_dir / f"{el}.upf"
                p.touch()
                dummy_files.append(p)
                config.oracle.dft.pseudopotentials[el] = str(p)

        # Instantiate Generator
        generator = SN2StructureGenerator(config)

        # Instantiate Orchestrator
        orchestrator = Orchestrator(
            config, base_dir=output_dir, structure_generator=generator
        )

        # Run
        result = orchestrator.run()

        # Cleanup dummy files
        for p in dummy_files:
            if p.exists():
                p.unlink()

        return result, output_dir

    return run_uat_workflow


@app.cell
def run_and_display(run_uat_workflow):
    # This cell executes the workflow when running in Marimo
    result, output_dir = run_uat_workflow()

    print(f"Workflow Completed. Status: {result.status}")
    if result.status == "failed":
        print(f"Error: {result.error}")
    else:
        print(f"Artifacts located in: {output_dir}")

    return output_dir, result


@app.cell
def visualizer(mo, output_dir, result):
    status = getattr(result, "status", "unknown")
    error = getattr(result, "error", None)
    metrics = getattr(result, "metrics", {})

    if status == "success":
        return mo.md(
            f"""
            ## ✅ UAT Passed!

            **Artifacts Directory**: `{output_dir}`

            **Pipeline Metrics**:
            ```json
            {metrics}
            ```
            """
        )

    return mo.md(f"## ❌ UAT Failed: {error}")


if __name__ == "__main__":
    app.run()
