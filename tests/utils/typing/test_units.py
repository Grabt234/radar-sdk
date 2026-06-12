import numpy as np
import pytest

from radar.utils.typing.units import Frequency, Phase, Angle, Distance, Length
from radar.utils.typing.enums import FrequencyUnit, PhaseUnit, DistanceUnit


def test_frequency():
    # Scaling tests
    f_hz = Frequency(150.0, FrequencyUnit.HERTZ)
    assert f_hz.Hz == 150.0
    assert f_hz.KHz == 0.150
    assert f_hz.MHz == 0.000150
    assert f_hz.GHz == 0.000000150

    f_ghz = Frequency(3.0, FrequencyUnit.GIGAHERTZ)
    assert f_ghz.Hz == 3e9
    assert f_ghz.KHz == 3e6
    assert f_ghz.MHz == 3000.0
    assert f_ghz.GHz == 3.0

    # Comparison tests
    f_mhz = Frequency(3000.0, FrequencyUnit.MEGAHERTZ)
    assert f_ghz == f_mhz
    assert f_hz < f_ghz
    assert f_ghz > f_hz
    assert f_ghz != f_hz

    # Check comparison with non-Frequency raises TypeError
    with pytest.raises(TypeError):
        _ = f_hz < 150.0


def test_phase_and_angle():
    # Angle is just an alias for Phase
    assert Angle is Phase

    # Conversions
    p_deg = Phase(180.0, PhaseUnit.DEGREE)
    assert p_deg.deg == 180.0
    assert np.allclose(p_deg.rad, np.pi)

    p_rad = Phase(np.pi / 2.0, PhaseUnit.RADIAN)
    assert np.allclose(p_rad.deg, 90.0)
    assert p_rad.rad == np.pi / 2.0

    # Comparison
    p_rad_2 = Angle(90.0, PhaseUnit.DEGREE)
    assert p_rad == p_rad_2
    assert p_rad < p_deg
    assert p_deg > p_rad
    assert p_deg != p_rad

    # Hashing
    assert hash(p_rad) == hash(p_rad_2)

    with pytest.raises(TypeError):
        _ = p_deg < 3.14


def test_distance_and_length():
    assert Length is Distance

    # Conversions
    d_m = Distance(10.0, DistanceUnit.METER)
    assert d_m.m == 10.0
    assert d_m.km == 0.01
    assert np.allclose(d_m.miles, 10.0 / 1609.34)
    assert np.allclose(d_m.ft, 10.0 / 0.3048)

    d_km = Distance(1.5, DistanceUnit.KILOMETER)
    assert d_km.m == 1500.0
    assert d_km.km == 1.5

    d_cm = Distance(250.0, DistanceUnit.CENTIMETER)
    assert d_cm.m == 2.5

    # Comparison
    d_m_2 = Distance(10.0, DistanceUnit.METER)
    assert d_m == d_m_2
    assert d_cm < d_m
    assert d_km > d_m
    assert d_km != d_m

    # Repr
    assert repr(d_m) == "Distance(10.0m)"

    with pytest.raises(TypeError):
        _ = d_m < 10.0
