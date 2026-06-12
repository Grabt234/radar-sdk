import warnings
import numpy as np
import polars as pl

from radar.utils.calculate.pattern import Isotropic
from radar.utils.typing import (
    Angle,
    PhaseUnit,
    Frequency,
    FrequencyUnit,
    AmplitudeDomain,
    AmplitudeUnit,
    DataHeader,
    DirectionDomain,
    FigureType,
)
from radar.components.element import Element
from radar.components.response import FrequencyResponse


# Helpers
def make_element(pattern=None, over_sample_factor=1):
    az = (Angle(-45, PhaseUnit.DEGREE), Angle(45, PhaseUnit.DEGREE))
    el = (Angle(-45, PhaseUnit.DEGREE), Angle(45, PhaseUnit.DEGREE))
    freq = Frequency(3.0, FrequencyUnit.GIGAHERTZ)
    pat = pattern or Isotropic()
    return Element(pat, az, el, freq, over_sample_factor=over_sample_factor), freq


def test_element_init_bounds():
    elem, _ = make_element()

    az = elem.azimuth_bound
    el = elem.elevation_bound

    assert np.allclose(az[0].deg, -45.0)
    assert np.allclose(az[1].deg, 45.0)
    assert np.allclose(el[0].deg, -45.0)
    assert np.allclose(el[1].deg, 45.0)


def test_element_beam_pattern_columns():
    elem, freq = make_element()
    bp = elem.beam_pattern(freq)

    expected_cols = [
        DataHeader.AZIMUTH_RAD,
        DataHeader.ELEVATION_RAD,
        DataHeader.AZIMUTH_DEG,
        DataHeader.ELEVATION_DEG,
        DataHeader.U,
        DataHeader.V,
        DataHeader.UV_MASK,
        DataHeader.BEAM_GAIN_DB,
        DataHeader.BEAM_GAIN_LINEAR,
    ]
    for col in expected_cols:
        assert col in bp.columns, f"Missing column: {col}"


def test_element_beam_pattern_isotropic_gains():
    # Isotropic: 0 dB / 1.0 linear everywhere
    elem, freq = make_element(Isotropic())
    bp = elem.beam_pattern(freq)

    assert (bp[DataHeader.BEAM_GAIN_DB] == 0).all()
    assert (bp[DataHeader.BEAM_GAIN_LINEAR] == 1).all()


def test_element_beam_pattern_with_frequency_response():
    az = (Angle(-45, PhaseUnit.DEGREE), Angle(45, PhaseUnit.DEGREE))
    el = (Angle(-45, PhaseUnit.DEGREE), Angle(45, PhaseUnit.DEGREE))

    freq = Frequency(3.0, FrequencyUnit.GIGAHERTZ)
    # -6 dB flat frequency response
    resp = FrequencyResponse(
        df=pl.DataFrame(
            {
                DataHeader.FREQ_FREQS: [freq.Hz],
                DataHeader.FREQ_GAIN_DB: [-6.0],
            }
        )
    )
    elem = Element(Isotropic(), az, el, resp)
    bp = elem.beam_pattern(freq)

    # Isotropic gain is 0 dB, response adds -6 dB → net -6 dB
    assert (bp[DataHeader.BEAM_GAIN_DB] == -6.0).all()
    # Linear: 0 dB isotropic = 1.0, response lin = 10^(-6/20) ~= 0.501
    expected_lin = 10.0 ** (-6.0 / 20.0)
    assert np.allclose(bp[DataHeader.BEAM_GAIN_LINEAR].to_list(), expected_lin)


def test_element_domain_methods():
    elem, _ = make_element()

    az_rad = elem.azimuth_domain(PhaseUnit.RADIAN)
    assert DataHeader.AZIMUTH_RAD in az_rad.columns

    az_deg = elem.azimuth_domain(PhaseUnit.DEGREE)
    assert DataHeader.AZIMUTH_DEG in az_deg.columns

    el_rad = elem.elevation_domain(PhaseUnit.RADIAN)
    assert DataHeader.ELEVATION_RAD in el_rad.columns

    el_deg = elem.elevation_domain(PhaseUnit.DEGREE)
    assert DataHeader.ELEVATION_DEG in el_deg.columns


def test_element_oversampling_increases_grid():
    elem_1x, _ = make_element(over_sample_factor=1)
    elem_2x, _ = make_element(over_sample_factor=2)

    bp_1x = elem_1x.beam_pattern(Frequency(3.0, FrequencyUnit.GIGAHERTZ))
    bp_2x = elem_2x.beam_pattern(Frequency(3.0, FrequencyUnit.GIGAHERTZ))

    assert bp_2x.shape[0] > bp_1x.shape[0]


def test_element_plot_antenna_factor_warning():
    elem, freq = make_element()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            elem.plot.beam(
                DirectionDomain.ANGLE,
                PhaseUnit.DEGREE,
                AmplitudeDomain.AntennaFactor,
                AmplitudeUnit.DECIBEL,
                FigureType.IMAGE,
                freq,
            )
        except Exception:
            pass  # Plotting may fail in headless, but the warning should have fired
        antenna_factor_warnings = [
            str(warning.message)
            for warning in w
            if "Antenna factor" in str(warning.message)
        ]
        assert len(antenna_factor_warnings) == 1


def test_element_plot_steer_warning():
    elem, freq = make_element()
    steer = (Angle(5, PhaseUnit.DEGREE), Angle(5, PhaseUnit.DEGREE))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            elem.plot.beam(
                DirectionDomain.ANGLE,
                PhaseUnit.DEGREE,
                AmplitudeDomain.Gain,
                AmplitudeUnit.DECIBEL,
                FigureType.IMAGE,
                freq,
                steer=steer,
            )
        except Exception:
            pass
        steer_warnings = [
            str(warning.message) for warning in w if "Steering" in str(warning.message)
        ]
        assert len(steer_warnings) == 1
