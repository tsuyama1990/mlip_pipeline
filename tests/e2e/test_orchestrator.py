import pytest
from pathlib import Path
from ase.io import write
from ase import Atoms
from mlip_autopipec.orchestrator.simple_orchestrator import SimpleOrchestrator
from mlip_autopipec.domain_models import GlobalConfig


def test_simple_orchestrator_run(tmp_path: Path) -> None:
    # Setup config
    config = GlobalConfig(
        max_cycles=2,
        initial_structure_path=tmp_path / "init.xyz",
        workdir=tmp_path / "work",
        oracle={"type": "mock"},
        trainer={"type": "mock"},
        dynamics={"type": "mock"},
        generator={"type": "mock"},
        validator={"type": "mock"},
        selector={"type": "mock"},
    )
    # Create dummy init file
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    write(tmp_path / "init.xyz", atoms)

    orch = SimpleOrchestrator(config)
    orch.run()

    # Assertions
    assert (tmp_path / "work").exists()
    assert (tmp_path / "work" / "mlip.log").exists()

def test_orchestrator_empty_dataset(tmp_path: Path) -> None:
    config = GlobalConfig(
        max_cycles=1,
        initial_structure_path=tmp_path / "nonexistent.xyz",
        workdir=tmp_path / "work_empty",
        oracle={"type": "mock"},
        trainer={"type": "mock"},
        dynamics={"type": "mock"},
        generator={"type": "mock"},
        validator={"type": "mock"},
        selector={"type": "mock"},
    )
    orch = SimpleOrchestrator(config)
    orch.run()
    assert (tmp_path / "work_empty" / "dataset.jsonl").exists()
    # It should have generated structures
    with (tmp_path / "work_empty" / "dataset.jsonl").open() as f:
        lines = f.readlines()
        assert len(lines) > 0

def test_orchestrator_halted_dynamics(tmp_path: Path) -> None:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    write(tmp_path / "init.xyz", atoms)

    config = GlobalConfig(
        max_cycles=1,
        initial_structure_path=tmp_path / "init.xyz",
        workdir=tmp_path / "work_halt",
        oracle={"type": "mock"},
        trainer={"type": "mock"},
        dynamics={"type": "mock", "prob_halt": 1.0}, # Force halt
        generator={"type": "mock"},
        validator={"type": "mock"},
        selector={"type": "mock"},
    )
    orch = SimpleOrchestrator(config)
    orch.run()
    # If halted, it selects and labels.
    # Dataset should grow.
    # Initial: 1. Halted -> Select 1 -> Label 1 -> Append 1. Total 2.
    with (tmp_path / "work_halt" / "dataset.jsonl").open() as f:
        lines = f.readlines()
        assert len(lines) >= 2
