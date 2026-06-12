import polars as pl
import pytest
import numpy as np

from radar.components.response import FrequencyResponse
from radar.utils.typing import Frequency, FrequencyUnit, AmplitudeUnit, DataHeader


def test_frequency_response_init():
    # Test flat initialization using Frequency
    freq = Frequency(3.0, FrequencyUnit.GIGAHERTZ)
    resp = FrequencyResponse(freq=freq)
    assert resp.response(freq) == 0.0

    # Test initialization with custom df
    df = pl.DataFrame(
        {
            DataHeader.FREQ_FREQS: [2.0e9, 4.0e9],
            DataHeader.FREQ_GAIN_DB: [-3.0, -1.0],
        }
    )
    resp_df = FrequencyResponse(df=df)
    assert resp_df._response[DataHeader.FREQ_FREQS].to_list() == [4.0e9, 2.0e9]

    # Test AssertionError if both or neither are defined
    with pytest.raises(AssertionError):
        FrequencyResponse(freq=freq, df=df)

    with pytest.raises(AssertionError):
        FrequencyResponse()


def test_frequency_response_exact_and_bounds():
    df = pl.DataFrame(
        {
            DataHeader.FREQ_FREQS: [2.0e9, 4.0e9],
            DataHeader.FREQ_GAIN_DB: [-3.0, -1.0],
        }
    )
    resp = FrequencyResponse(df=df)

    # Exact match in dB and Linear
    freq_2 = Frequency(2.0, FrequencyUnit.GIGAHERTZ)
    assert resp.response(freq_2, AmplitudeUnit.DECIBEL) == -3.0
    assert np.allclose(
        resp.response(freq_2, AmplitudeUnit.LINEAR), 10.0 ** (-3.0 / 20.0)
    )

    # Out of bounds raises ValueError
    freq_low = Frequency(1.9, FrequencyUnit.GIGAHERTZ)
    with pytest.raises(ValueError, match="is out of bounds"):
        resp.response(freq_low)

    freq_high = Frequency(4.1, FrequencyUnit.GIGAHERTZ)
    with pytest.raises(ValueError, match="is out of bounds"):
        resp.response(freq_high)


def test_frequency_response_interpolation():
    df = pl.DataFrame(
        {
            DataHeader.FREQ_FREQS: [2.0e9, 4.0e9],
            DataHeader.FREQ_GAIN_DB: [-3.0, -1.0],
        }
    )
    resp = FrequencyResponse(df=df)

    # Intermediate frequency (3 GHz is halfway between 2 GHz and 4 GHz)
    freq_mid = Frequency(3.0, FrequencyUnit.GIGAHERTZ)

    # Expected gain in dB is halfway: -2.0 dB
    assert resp.response(freq_mid, AmplitudeUnit.DECIBEL) == -2.0

    # Expected linear gain: 10 ** (-2.0 / 20)
    assert np.allclose(
        resp.response(freq_mid, AmplitudeUnit.LINEAR), 10.0 ** (-2.0 / 20.0)
    )
