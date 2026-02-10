import numpy as np
from mlip_autopipec.domain_models.datastructures import sanitize_value

def test_sanitize_value_nested():
    data = {
        "a": np.int64(1),
        "b": {
            "c": np.float64(2.0),
            "d": [np.bool_(True), 3]
        },
        "e": [np.int32(4)]
    }
    sanitized = sanitize_value(data)
    assert isinstance(sanitized["a"], int)
    assert isinstance(sanitized["b"]["c"], float)
    assert isinstance(sanitized["b"]["d"][0], bool)
    assert sanitized["b"]["d"][0] is True
    assert isinstance(sanitized["e"][0], int)
