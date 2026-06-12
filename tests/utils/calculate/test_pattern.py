import polars as pl
import numpy as np
import pytest

from radar.utils.calculate.pattern import (
    CustomPattern,
    Isotropic,
    Cosine,
    Gaussian,
    Sinc,
)
from radar.utils.typing import DataHeader, Angle, PhaseUnit


def test_isotropic():
    iso = Isotropic()
    df = pl.DataFrame(
        {
            DataHeader.AZIMUTH_RAD: [0.0, 1.0],
            DataHeader.ELEVATION_RAD: [0.0, -1.0],
        }
    )
    result = iso.calculate_pattern(df)
    assert DataHeader.BEAM_GAIN_DB in result.columns
    assert DataHeader.BEAM_GAIN_LINEAR in result.columns
    assert result[DataHeader.BEAM_GAIN_DB].to_list() == [0, 0]
    assert result[DataHeader.BEAM_GAIN_LINEAR].to_list() == [1, 1]


def test_cosine():
    cos_pat = Cosine(order=1)
    df = pl.DataFrame(
        {
            DataHeader.AZIMUTH_RAD: [0.0, np.pi / 3, np.pi / 2],
            DataHeader.ELEVATION_RAD: [0.0, 0.0, 0.0],
        }
    )
    result = cos_pat.calculate_pattern(df)
    assert np.allclose(result[DataHeader.BEAM_GAIN_LINEAR].to_list(), [1.0, 0.5, 0.0])
    assert np.allclose(
        result[DataHeader.BEAM_GAIN_DB].to_list(), [0.0, 10.0 * np.log10(0.5), -200.0]
    )

    cos_pat_2 = Cosine(order=2)
    result_2 = cos_pat_2.calculate_pattern(df)
    assert np.allclose(
        result_2[DataHeader.BEAM_GAIN_LINEAR].to_list(), [1.0, 0.25, 0.0]
    )


def test_gaussian():
    bw_az = Angle(10.0, PhaseUnit.DEGREE)
    bw_el = Angle(20.0, PhaseUnit.DEGREE)
    gauss = Gaussian((bw_az, bw_el))

    df_half = pl.DataFrame(
        {
            DataHeader.AZIMUTH_RAD: [0.0, bw_az.rad / 2.0],
            DataHeader.ELEVATION_RAD: [0.0, 0.0],
        }
    )
    result_half = gauss.calculate_pattern(df_half)
    assert np.allclose(result_half[DataHeader.BEAM_GAIN_LINEAR].to_list(), [1.0, 0.5])
    assert np.allclose(
        result_half[DataHeader.BEAM_GAIN_DB].to_list(), [0.0, 10.0 * np.log10(0.5)]
    )


def test_sinc():
    bw_az = Angle(10.0, PhaseUnit.DEGREE)
    bw_el = Angle(20.0, PhaseUnit.DEGREE)
    sinc_pat = Sinc((bw_az, bw_el))

    df = pl.DataFrame(
        {
            DataHeader.AZIMUTH_RAD: [0.0, bw_az.rad / 2.0],
            DataHeader.ELEVATION_RAD: [0.0, 0.0],
        }
    )
    result = sinc_pat.calculate_pattern(df)
    assert np.allclose(result[DataHeader.BEAM_GAIN_LINEAR].to_list()[0], 1.0)
    assert np.allclose(
        result[DataHeader.BEAM_GAIN_LINEAR].to_list()[1], np.sqrt(0.5), atol=1e-4
    )


def test_custom_pattern():
    ref_df = pl.DataFrame(
        {
            DataHeader.AZIMUTH_DEG: [-5.0, -5.0, 5.0, 5.0],
            DataHeader.ELEVATION_DEG: [-10.0, 10.0, -10.0, 10.0],
            DataHeader.BEAM_GAIN_LINEAR: [1.0, 2.0, 3.0, 4.0],
        }
    )

    pat = CustomPattern(ref_df)
    assert DataHeader.BEAM_GAIN_DB in pat.df.columns

    input_df = pl.DataFrame(
        {
            DataHeader.AZIMUTH_DEG: [-5.0, -5.0, 5.0, 5.0],
            DataHeader.ELEVATION_DEG: [-10.0, 10.0, -10.0, 10.0],
        }
    )
    res = pat.calculate_pattern(input_df)
    assert len(res) == 4
    assert res[DataHeader.BEAM_GAIN_LINEAR].to_list() == [1.0, 2.0, 3.0, 4.0]

    invalid_input = pl.DataFrame({DataHeader.AZIMUTH_DEG: [-5.0, 5.0]})
    with pytest.raises(ValueError, match="Required columns missing"):
        pat.calculate_pattern(invalid_input)

    mismatched_bounds = pl.DataFrame(
        {
            DataHeader.AZIMUTH_DEG: [0.0, 0.0, 5.0, 5.0],
            DataHeader.ELEVATION_DEG: [-10.0, 10.0, -10.0, 10.0],
        }
    )
    with pytest.raises(ValueError, match="Surface mismatch"):
        pat.calculate_pattern(mismatched_bounds)

    incomplete = pl.DataFrame(
        {DataHeader.AZIMUTH_DEG: [-5.0, 5.0], DataHeader.ELEVATION_DEG: [-10.0, 10.0]}
    )
    with pytest.raises(ValueError, match="Incomplete surface"):
        pat.calculate_pattern(incomplete)
