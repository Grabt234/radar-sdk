import polars as pl

from radar.utils.typing.constants import RadarConstants
from radar.utils.typing.enums import DistanceUnit
from radar.utils.typing.units import Distance, Frequency

__all__ = ["from_db", "to_db", "cf_to_min_dist"]


def from_db(col_name: str) -> pl.Expr:
    """Polars expression equivalent of your from_db function."""
    return 10.0 ** (pl.col(col_name) / 10.0)


def to_db(col_expr: pl.Expr) -> pl.Expr:
    """Polars expression equivalent of your to_db function.

    Uses pl.when().then().otherwise() to mimic np.maximum(arr, 1e-10)
    without breaking the Polars execution graph.
    """
    return pl.when(col_expr > 1e-10).then(10.0 * col_expr.log10()).otherwise(-200.0)


def cf_to_min_dist(frequency: Frequency) -> Distance:
    return Distance(RadarConstants.c / (2 * frequency.Hz), DistanceUnit.METER)
