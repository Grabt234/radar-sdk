import polars as pl
import numpy as np

from radar.utils.calculate.convert import from_db, to_db, cf_to_min_dist
from radar.utils.typing import Frequency, FrequencyUnit, Distance


def test_from_db():
    # Test from_db expression from string
    df = pl.DataFrame({"db_col": [0.0, 20.0, -20.0]})
    expr = from_db("db_col")
    result = df.select(expr.alias("lin_col"))
    assert np.allclose(result["lin_col"].to_list(), [1.0, 10.0, 0.1])


def test_to_db():
    # Test to_db expression with thresholds
    df = pl.DataFrame({"lin_col": [1.0, 10.0, 1e-12]})
    expr = to_db(pl.col("lin_col"))
    result = df.select(expr.alias("db_col"))
    assert np.allclose(result["db_col"].to_list(), [0.0, 20.0, -200.0])


def test_cf_to_min_dist():
    # Test frequency mapping (distance = c / 2f)
    freq = Frequency(300.0, FrequencyUnit.MEGAHERTZ)
    dist = cf_to_min_dist(freq)
    assert isinstance(dist, Distance)
    assert np.allclose(dist.m, 299792458 / (2 * 300e6))
