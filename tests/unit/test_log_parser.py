from mlip_autopipec.physics.dynamics.log_parser import LammpsLogParser

def test_parse_halt_detected():
    log_content = """
    Step Temp PotEng
    100 300 -500
    ERROR: Fix halt condition met (src/fix_halt.cpp:45)
    Last command: run 1000
    """

    parser = LammpsLogParser()
    result = parser.parse(log_content)

    assert result.halt_detected is True
    # If we can parse the step, ideally it returns it, but fix halt log format varies.
    # Assuming we can't always get the step from the error line easily without more context.

def test_parse_no_halt():
    log_content = """
    Step Temp PotEng
    100 300 -500
    Loop time of 1.23 on 1 procs
    """

    parser = LammpsLogParser()
    result = parser.parse(log_content)

    assert result.halt_detected is False


def test_parse_max_gamma():
    # If we print gamma in thermo, we might parse it.
    # Suppose we configure 'thermo_style custom step c_pace_gamma'
    log_content = """
    Step c_pace_gamma
    10 0.1
    20 0.5
    30 1.2
    """
    parser = LammpsLogParser()
    result = parser.parse(log_content)

    # We expect the parser to find max gamma if columns are present
    assert result.max_gamma == 1.2
