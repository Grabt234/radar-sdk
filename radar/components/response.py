from radar.utils.typing import Frequency, AmplitudeUnit, DataHeader
import polars as pl
from radar.utils.calculate.convert import from_db


class FrequencyResponse:
    def __init__(self, freq: Frequency | None = None, df: pl.DataFrame | None = None):

        if freq is not None and df is not None:
            raise AssertionError(
                "Either frequency or frequency response df should be defined but not both"
            )
        elif freq is None and df is None:
            raise AssertionError(
                "Either frequency or frequency response df should be defined but not both"
            )

        if freq:
            df = pl.DataFrame(
                {DataHeader.FREQ_GAIN_DB: 0, DataHeader.FREQ_FREQS: freq.Hz}
            )

        assert isinstance(df, pl.DataFrame)

        self._response = df.sort(DataHeader.FREQ_FREQS, descending=True)

    def response(
        self, freq: Frequency, unit: AmplitudeUnit = AmplitudeUnit.DECIBEL
    ) -> float:
        """
        Retrieves or interpolates the amplitude gain for a given frequency.

        This method queries the underlying DataFrame for the exact frequency requested.
        If the exact frequency does not exist but falls within the range of the dataset,
        it performs a linear interpolation between the two closest neighboring frequencies.

        Parameters
        ----------
        freq : float
            The frequency to query, typically specified in Hertz (Hz).
        unit : AmplitudeUnit, default AmplitudeUnit.DECIBEL
            The unit format for the returned gain value. Supports DECIBEL (dB)
            or LINEAR ratio.

        Returns
        -------
        float
            The calculated or interpolated gain value in the requested unit.

        Raises
        ------
        ValueError
            If the requested frequency is out of the dataset bounds.
        """
        # 1. Exact Match Look-up
        exact_match = self._response.filter(pl.col(DataHeader.FREQ_FREQS) == freq.Hz)
        if not exact_match.is_empty():
            if unit == AmplitudeUnit.DECIBEL:
                expr = pl.col(DataHeader.FREQ_GAIN_DB)
            else:
                expr = from_db(DataHeader.FREQ_GAIN_DB)
            return float(exact_match.select(expr).item(0, 0))

        # 2. Boundary Guards (Strict Exception Handling)
        min_freq = self._response[DataHeader.FREQ_FREQS][-1]
        max_freq = self._response[DataHeader.FREQ_FREQS][0]

        assert isinstance(min_freq, float)
        assert isinstance(max_freq, float)

        if freq.Hz < min_freq or freq.Hz > max_freq:
            raise ValueError(
                f"Requested frequency {freq.Hz} Hz is out of bounds. "
                f"Available range is [{min_freq}, {max_freq}] Hz."
            )

        # 3. Linear Interpolation Bounding
        lower_row = self._response.filter(pl.col(DataHeader.FREQ_FREQS) < freq.Hz)[-1]
        upper_row = self._response.filter(pl.col(DataHeader.FREQ_FREQS) > freq.Hz)[0]

        f0 = lower_row.item(0, DataHeader.FREQ_FREQS)
        y0 = lower_row.item(0, DataHeader.FREQ_GAIN_DB)

        f1 = upper_row.item(0, DataHeader.FREQ_FREQS)
        y1 = upper_row.item(0, DataHeader.FREQ_GAIN_DB)

        # Linear interpolation math
        gain_db = y0 + (freq.Hz - f0) * ((y1 - y0) / (f1 - f0))

        if unit == AmplitudeUnit.DECIBEL:
            return float(gain_db)
        else:
            expr = from_db("gain_db_col")
            return float(
                pl.DataFrame({"gain_db_col": [gain_db]}).select(expr).item(0, 0)
            )
