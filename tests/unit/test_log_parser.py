from pathlib import Path
from mlip_autopipec.physics.dynamics.log_parser import LogParser
from mlip_autopipec.domain_models.dynamics import MDStatus


def create_log_file(path: Path, content: str) -> None:
    path.write_text(content)


def test_parse_clean_log(tmp_path: Path) -> None:
    log_file = tmp_path / "clean.log"
    content = """
    LAMMPS (29 Sep 2021)
    ...
    Loop time of 10.0 on 1 procs for 100 steps with 10 atoms
    Performance: 10.0 ns/day, 2.4 hours/ns, 10.0 timesteps/s
    99.9% CPU use with 1 MPI tasks x 1 OpenMP threads
    ...
    Total wall time: 0:00:10
    """
    create_log_file(log_file, content)
    parser = LogParser()
    result = parser.parse(log_file)
    assert result.status == MDStatus.COMPLETED


def test_parse_halted_log(tmp_path: Path) -> None:
    log_file = tmp_path / "halted.log"
    content = """
    LAMMPS (29 Sep 2021)
    ...
    Step Temp E_pair E_mol TotEng Press
           0          300   -1000.0         0   -1000.0       0.0
          10          300   -1000.0         0   -1000.0       0.0
    ERROR: Fix halt condition met (src/fix_halt.cpp:123)
    Last command: run 1000
    """
    create_log_file(log_file, content)
    parser = LogParser()
    result = parser.parse(log_file)
    assert result.status == MDStatus.HALTED
    if result.halt_step is not None:
        assert result.halt_step >= 0


def test_parse_crash_log(tmp_path: Path) -> None:
    log_file = tmp_path / "crash.log"
    content = """
    LAMMPS (29 Sep 2021)
    ...
    Step Temp E_pair E_mol TotEng Press
           0          300   -1000.0         0   -1000.0       0.0
    Segmentation fault (core dumped)
    """
    create_log_file(log_file, content)
    parser = LogParser()
    result = parser.parse(log_file)
    assert result.status == MDStatus.FAILED
