from typing import Union

import numpy as np
import polars as pl

from radar.utils.typing.constants import RadarConstants
from radar.utils.typing.enums import DistanceUnit
from radar.utils.typing.units import Distance, Frequency

__all__ = ["to_db"]


def to_db(arr: Union[np.typing.NDArray, pl.Series]) -> np.typing.NDArray:
    """dB relative to peak for a normalized field amplitude pattern (``|E|/|E_max|``)."""
    # np.maximum prevents log10(0) down to a -200 dB floor
    return 20 * np.log10(np.maximum(arr, 1e-10))


def from_db(arr: Union[np.typing.NDArray, pl.Series]) -> np.typing.NDArray:
    """Converts dB back to a normalized linear field amplitude ratio (``|E|/|E_max|``)."""
    # Converts a Polars Series to a NumPy array if necessary to maintain type consistency
    array_input = arr.to_numpy() if isinstance(arr, pl.Series) else arr
    return 10 ** (array_input / 20.0)


def cf_to_min_dist(frequency: Frequency) -> Distance:
    return Distance(RadarConstants.c / (2 * frequency.Hz), DistanceUnit.METER)
