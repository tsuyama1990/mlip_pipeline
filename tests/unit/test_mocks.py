from ase import Atoms

# We need to import these, but they don't exist yet.
# In strict TDD, we'd create the file and then run tests to see import errors.
# But for this environment, I will create the test file and then implement the code.
# I will use safe imports or just standard imports.
from mlip_autopipec.domain_models.datastructures import Structure
from mlip_autopipec.dynamics.interface import MockDynamics

# Assuming these will be available
from mlip_autopipec.generator.interface import MockGenerator
from mlip_autopipec.oracle.interface import MockOracle
from mlip_autopipec.trainer.interface import MockTrainer
from mlip_autopipec.validator.interface import MockValidator


def test_mock_generator_explore() -> None:
    generator = MockGenerator()
    structures_iter = generator.explore(context={"count": 2})
    # Do not convert to list
    first_structure = next(structures_iter)
    assert isinstance(first_structure, Structure)
    assert isinstance(first_structure.atoms, Atoms)
    assert first_structure.provenance == "mock_generator"
    # Verify content
    assert len(first_structure.atoms) == 2  # He2
    assert first_structure.atoms.get_chemical_symbols() == ["He", "He"]  # type: ignore[no-untyped-call]
    assert first_structure.label_status == "unlabeled"


def test_mock_generator_local_candidates(mock_atoms: Atoms) -> None:
    generator = MockGenerator()
    seed = Structure(atoms=mock_atoms, provenance="seed")

    # Generate 5 local candidates
    candidates = generator.generate_local_candidates(seed, count=5)

    # Verify count and type
    candidate_list = list(candidates)
    assert len(candidate_list) == 5
    for c in candidate_list:
        assert isinstance(c, Structure)
        assert "local_candidate" in c.provenance


def test_mock_oracle_compute(mock_atoms: Atoms) -> None:
    oracle = MockOracle()
    s = Structure(atoms=mock_atoms, provenance="test", label_status="unlabeled")
    # Oracle takes iterable, returns iterator
    labeled_iter = oracle.compute([s])
    first_labeled = next(labeled_iter)

    # Check if labeled
    assert first_labeled.label_status == "labeled"
    assert first_labeled.energy is not None
    assert first_labeled.forces is not None
    assert first_labeled.stress is not None

    # Verify values logic (mock returns negative energy based on atom count)
    # mock_atoms H2 has 2 atoms. Energy ~ -4.0 * 2 = -8.0
    assert first_labeled.energy < -7.0


def test_mock_trainer_train(mock_atoms: Atoms) -> None:
    from pathlib import Path

    trainer = MockTrainer(work_dir=Path())
    s = Structure(
        atoms=mock_atoms,
        provenance="test",
        label_status="labeled",
        energy=-10.0,
        forces=[[0, 0, 0], [0, 0, 0]],
        stress=[0, 0, 0, 0, 0, 0],
    )
    # Trainer accepts Iterable
    structures = [s]
    potential = trainer.train(structures)
    assert potential.path.name.startswith("potential_")
    assert potential.path.name.endswith(".yace")
    assert potential.format == "yace"


def test_mock_trainer_select_active_set(mock_atoms: Atoms) -> None:
    from pathlib import Path

    trainer = MockTrainer(work_dir=Path())

    structures = [Structure(atoms=mock_atoms, provenance=f"cand_{i}") for i in range(10)]

    # Select 3
    selected = trainer.select_active_set(structures, count=3)
    selected_list = list(selected)

    assert len(selected_list) == 3
    # Check provenance modified by mock implementation
    assert "selected" in selected_list[0].provenance


def test_mock_dynamics_simulate(mock_atoms: Atoms) -> None:
    dynamics = MockDynamics()
    # Need a potential
    from pathlib import Path

    from mlip_autopipec.domain_models.datastructures import Potential

    pot = Potential(path=Path("potential.yace"), format="yace")

    s = Structure(atoms=mock_atoms, provenance="test")

    # simulate returns Iterator[Structure]
    trajectory_iter = dynamics.simulate(pot, s)
    first_frame = next(trajectory_iter)
    assert isinstance(first_frame, Structure)

    # Verify we can get more frames
    frames = [first_frame]
    for _ in range(4):
        frames.append(next(trajectory_iter))

    assert len(frames) == 5

    # Verify atoms moved
    first_pos = frames[0].to_ase().positions
    last_pos = frames[-1].to_ase().positions
    import numpy as np

    assert not np.array_equal(first_pos, last_pos)


def test_mock_validator_validate() -> None:
    validator = MockValidator()
    from pathlib import Path

    from mlip_autopipec.domain_models.datastructures import Potential

    pot = Potential(path=Path("potential.yace"), format="yace")

    result = validator.validate(pot)
    assert result.passed is True
    assert "phonon_stability" in result.metrics
