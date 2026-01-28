from mlip_autopipec.inference.parsers import LammpsLogParser


def test_log_parser_no_file(tmp_path):
    log_file = tmp_path / "non_existent.log"
    max_gamma, halted, halt_step = LammpsLogParser.parse(log_file)
    assert max_gamma == 0.0
    assert halted is False
    assert halt_step is None


def test_log_parser_success_run(tmp_path):
    log_file = tmp_path / "log.lammps"
    content = """
Step Temp v_max_gamma
0 300 0.1
10 300 0.5
20 300 1.2
Loop time of 1.23 on 1 procs
    """
    log_file.write_text(content)

    max_gamma, halted, halt_step = LammpsLogParser.parse(log_file)
    assert max_gamma == 1.2
    assert halted is False
    assert halt_step is None


def test_log_parser_halted_run(tmp_path):
    log_file = tmp_path / "log.lammps"
    content = """
Step Temp v_max_gamma
0 300 0.1
10 300 6.5
ERROR: Fix halt condition met (src/fix_halt.cpp:123)
Last command: run 1000
    """
    log_file.write_text(content)

    max_gamma, halted, halt_step = LammpsLogParser.parse(log_file)
    assert max_gamma == 6.5
    assert halted is True
    assert halt_step == 10


def test_log_parser_halted_run_explicit_step(tmp_path):
    log_file = tmp_path / "log.lammps"
    content = """
Step Temp v_max_gamma
0 300 0.1
100 305 2.0
200 310 5.5
ERROR: Fix halt condition met
    """
    log_file.write_text(content)

    max_gamma, halted, halt_step = LammpsLogParser.parse(log_file)
    assert max_gamma == 5.5
    assert halted is True
    assert halt_step == 200
