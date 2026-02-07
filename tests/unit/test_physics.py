import numpy as np
import pytest

from mlip_autopipec.utils.physics import kspacing_to_grid


def test_kspacing_to_grid_cubic() -> None:
    # Cubic cell 10x10x10
    cell = np.eye(3) * 10.0
    # Reciprocal length: 2*pi / 10 = 0.6283
    # kspacing = 0.2
    # grid = ceil(0.6283 / 0.2) = ceil(3.1415) = 4
    grid = kspacing_to_grid(cell, 0.2)
    assert grid == (4, 4, 4)

def test_kspacing_to_grid_orthorhombic() -> None:
    # 5x10x20
    cell = np.diag([5.0, 10.0, 20.0])
    # b1 = 2pi/5 ~ 1.2566. grid = ceil(1.2566/0.2) = 7
    # b2 = 2pi/10 ~ 0.6283. grid = ceil(0.6283/0.2) = 4
    # b3 = 2pi/20 ~ 0.3141. grid = ceil(0.3141/0.2) = 2
    grid = kspacing_to_grid(cell, 0.2)
    assert grid == (7, 4, 2)

def test_kspacing_to_grid_small_spacing() -> None:
    cell = np.eye(3) * 5.0 # b = 1.256
    # kspacing = 2.0 (very large)
    # grid = ceil(1.256/2.0) = 1
    grid = kspacing_to_grid(cell, 2.0)
    assert grid == (1, 1, 1)

def test_kspacing_invalid() -> None:
    cell = np.eye(3)
    with pytest.raises(ValueError, match="kspacing must be positive"):
        kspacing_to_grid(cell, 0.0)
