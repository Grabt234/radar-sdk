from radar.components import geometry
from radar.utils.typing.enums import FrequencyUnit
from radar.utils.typing.units import Frequency

from radar.components import Element
from radar.utils.calculate import pattern
from radar.utils.typing import (
    PhaseUnit,
    Angle,
)

from radar.components.array import Array
import polars as pl

from radar.utils.typing.constants import DataHeader

from radar.optimiser.biology import Organism
from radar.utils.typing.enums import BoundType


def fitness_function(org: Organism, generation: int | None = None, plot=False) -> dict:

    x = org._chromosomes[0]
    y = org._chromosomes[1]
    g = org._chromosomes[2]

    # print(g.df)

    az_bound = 90
    el_bound = 90
    az_bound_tuple = (
        Angle(-az_bound, PhaseUnit.DEGREE),
        Angle(az_bound, PhaseUnit.DEGREE),
    )
    el_bound_tuple = (
        Angle(-el_bound, PhaseUnit.DEGREE),
        Angle(el_bound, PhaseUnit.DEGREE),
    )

    element_pattern = pattern.Isotropic()
    freq = Frequency(1, FrequencyUnit.GIGAHERTZ)
    antenna_element = Element(element_pattern, az_bound_tuple, el_bound_tuple, freq, 1)

    cf = Frequency(1, FrequencyUnit.GIGAHERTZ)
    array_geometry = geometry.CustomGeometry(
        x.df.to_numpy().ravel(), y.df.to_numpy().ravel()
    )

    # Geometry
    df = array_geometry.df
    updated_df = df.with_columns(
        g.df.to_series().alias(DataHeader.GEOM_AMP_GAIN_DB)
    ).drop(DataHeader.GEOM_AMP_GAIN_LIN)
    array_geometry.gains = updated_df

    arr = Array(antenna_element, array_geometry)

    beam = arr.beam_pattern(cf, None)
    beam2 = beam.clone()

    beam2 = beam2.with_columns(
        pl.when(
            (pl.col(DataHeader.AZIMUTH_DEG).abs() > 10)
            & (pl.col(DataHeader.ELEVATION_DEG).abs() > 10)
        )
        .then(1)  # If condition is True -> 0
        .otherwise(0.01)  # If condition is False -> -20
        .alias(DataHeader.BEAM_GAIN_LINEAR)
    )

    average_diff = (
        beam2.select(
            (
                pl.col(DataHeader.BEAM_GAIN_LINEAR)
                - pl.lit(beam[DataHeader.BEAM_GAIN_LINEAR])
            )
            .abs()
            .mean()
        ).item()  # Extracts the single value from the resulting 1x1 DataFrame
    )

    filters = [
        (DataHeader.AZIMUTH_DEG, -20.0, 20, BoundType.EXCLUSIVE),
        (DataHeader.ELEVATION_DEG, -20, 20, BoundType.EXCLUSIVE),
    ]
    ave_db = arr.statistic.mean(DataHeader.ANTENNA_FACTOR_DB, freq, filters, None, "OR")
    std_db = arr.statistic.std(DataHeader.ANTENNA_FACTOR_DB, freq, filters, None, "OR")
    max_db = arr.statistic.max(DataHeader.ANTENNA_FACTOR_DB, freq, filters, None, "OR")
    min_db = arr.statistic.max(DataHeader.ANTENNA_FACTOR_DB, freq, filters, None, "OR")

    ave_prop = 200 * (10 ** (ave_db / 10.0))
    std_dev_prop = 40 * (10 ** (std_db / 10.0))
    diff_prop = 8 * average_diff
    max_prop = 4 * (10 ** (max_db / 10))

    diff = 10 ** ((max_db - min_db) / 10)
    fitness = (ave_prop + std_dev_prop + diff_prop) * max_prop * diff

    return {
        "fitness": fitness,
        "average_db": ave_db,
        "standard_deviation_db": std_db,
        "ave_diff_lin": average_diff,
        "ave_prop": ave_prop,
        "std_dev_prop": std_dev_prop,
        "diff_prop": diff_prop,
        "max_db": max_db,
        "max_prop": max_prop,
    }  # arr.statistic.ave(cf)
