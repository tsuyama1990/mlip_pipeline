from mlip_autopipec.inference.parsers import LogParser


def test_log_parser_no_file(tmp_path):
    log_file = tmp_path / "non_existent.log"
    max_gamma, halted, halt_step = LogParser.parse(log_file)
    assert max_gamma == 0.0
    assert halted is False
    assert halt_step is None


def test_log_parser_success_run(tmp_path):
    log_file = tmp_path / "log.lammps"
    content = """
Step Temp c_max_gamma
0 300 0.1
10 300 0.5
20 300 1.2
Loop time of 1.23 on 1 procs
    """
    log_file.write_text(content)

    max_gamma, halted, halt_step = LogParser.parse(log_file)
    assert max_gamma == 1.2
    assert halted is False
    assert halt_step is None


def test_log_parser_halted_run(tmp_path):
    log_file = tmp_path / "log.lammps"
    content = """
Step Temp c_max_gamma
0 300 0.1
10 300 6.5
ERROR: Fix halt condition met (src/fix_halt.cpp:123)
Last command: run 1000
    """
    log_file.write_text(content)

    max_gamma, halted, halt_step = LogParser.parse(log_file)
    assert max_gamma == 6.5
    assert halted is True
    # In some LAMMPS versions it might output step, but usually we rely on finding the last step output
    # or the error message doesn't contain the step explicitly unless we parse "Last command" or the last step line.
    # Our parser should probably look at the last printed step.
    # If the parser logic is robust, it finds 10 as the step where it exceeded.

    # Let's assume our parser implementation will look at the last valid Step line if halt detected.


def test_log_parser_halted_run_explicit_step(tmp_path):
    # Some setups might use "fix halt ... error hard" which terminates.
    # The last line before error usually indicates state.
    log_file = tmp_path / "log.lammps"
    content = """
Step Temp c_max_gamma
0 300 0.1
100 305 2.0
200 310 5.5
ERROR: Fix halt condition met
    """
    log_file.write_text(content)

    max_gamma, halted, halt_step = LogParser.parse(log_file)
    assert max_gamma == 5.5
    assert halted is True
    # Ideally halt_step would be 200
