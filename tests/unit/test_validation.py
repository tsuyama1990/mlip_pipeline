from pathlib import Path

from mlip_autopipec.domain_models.validation import ValidationResult


def test_validation_result_passed() -> None:
    res = ValidationResult(passed=True, metrics={"rmse": 0.005}, report_path=Path("report.html"))
    assert res.passed
    assert res.metrics["rmse"] == 0.005
    assert res.report_path == Path("report.html")


def test_validation_result_failed() -> None:
    res = ValidationResult(passed=False, details={"error": "Too high RMSE"})
    assert not res.passed
    assert res.metrics == {}
    assert res.report_path is None
    assert res.details["error"] == "Too high RMSE"
