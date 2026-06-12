import polars as pl

from radar.utils.typing import (
    Frequency,
)
from radar.utils.typing.constants import DataHeader, RadarConstants

import numpy as np


class Monopulse:
    @classmethod
    def _preprocess_geometry(
        cls, geometry: pl.DataFrame, condition: pl.Expr
    ) -> pl.DataFrame:
        """Computes spatial splitting weights based on a dynamic condition."""
        return geometry.with_columns(
            pl.when(condition).then(1.0).otherwise(-1.0).alias("diff_weight")
        )

    @classmethod
    def _compute_element_responses(
        cls, array_beam: pl.DataFrame, geometry_processed: pl.DataFrame, k: float
    ) -> pl.DataFrame:
        """Performs cross-join and computes complex voltage contributions per element."""
        return (
            array_beam.join(geometry_processed, how="cross")
            .with_columns(
                [
                    # FIX: Use sine for Azimuth so dir_x = 0 at boresight
                    (
                        pl.col(DataHeader.ELEVATION_RAD).cos()
                        * pl.col(DataHeader.AZIMUTH_RAD).sin()
                    ).alias("dir_x"),
                    # FIX: Use sine for Elevation so dir_y = 0 at boresight
                    (
                        pl.col(DataHeader.ELEVATION_RAD).sin()
                        * pl.col(DataHeader.AZIMUTH_RAD).cos()
                    ).alias("dir_y"),
                    (10.0 ** (pl.col(DataHeader.BEAM_GAIN_DB) / 20.0)).alias(
                        "element_field"
                    ),
                ]
            )
            .with_columns(
                phase=-k
                * (
                    pl.col("dir_x") * pl.col(DataHeader.X_POS_M)
                    + pl.col("dir_y") * pl.col(DataHeader.Y_POS_M)
                )
            )
            .with_columns(
                elem_re=pl.col("element_field") * pl.col("phase").cos(),
                elem_im=pl.col("element_field") * pl.col("phase").sin(),
                elem_diff_re=pl.col("element_field")
                * pl.col("phase").cos()
                * pl.col("diff_weight"),
                elem_diff_im=pl.col("element_field")
                * pl.col("phase").sin()
                * pl.col("diff_weight"),
            )
        )

    @classmethod
    def _aggregate_and_compute_ratios(
        cls, working_df: pl.DataFrame, array_beam_id: pl.DataFrame
    ) -> pl.DataFrame:
        """Aggregates element responses back to the angular grid and calculates channels/ratios."""
        aggregated = working_df.group_by("grid_id").agg(
            sum_re=pl.sum("elem_re"),
            sum_im=pl.sum("elem_im"),
            diff_re=pl.sum("elem_diff_re"),
            diff_im=pl.sum("elem_diff_im"),
        )

        # Re-join and execute math pipeline cleanly
        calc_df = array_beam_id.join(aggregated, on="grid_id").with_columns(
            sum_mag_sq=pl.col("sum_re") ** 2 + pl.col("sum_im") ** 2,
            sum_mag_lin=(pl.col("sum_re") ** 2 + pl.col("sum_im") ** 2).sqrt(),
            diff_mag_lin=(pl.col("diff_re") ** 2 + pl.col("diff_im") ** 2).sqrt(),
        )

        # Inject safe denominator for complex division
        calc_df = calc_df.with_columns(
            safe_denom=pl.when(pl.col("sum_mag_sq") > 1e-15)
            .then(pl.col("sum_mag_sq"))
            .otherwise(1e-15)
        )

        # Ratios and Decibel metrics
        return calc_df.with_columns(
            ratio_re=(
                pl.col("diff_re") * pl.col("sum_re")
                + pl.col("diff_im") * pl.col("sum_im")
            )
            / pl.col("safe_denom"),
            ratio_im=(
                pl.col("diff_im") * pl.col("sum_re")
                - pl.col("diff_re") * pl.col("sum_im")
            )
            / pl.col("safe_denom"),
        ).with_columns(
            sum_db=pl.when(pl.col("sum_mag_lin") > 1e-15)
            .then(20.0 * pl.col("sum_mag_lin").log10())
            .otherwise(-300.0),
            diff_db=pl.when(pl.col("diff_mag_lin") > 1e-15)
            .then(20.0 * pl.col("diff_mag_lin").log10())
            .otherwise(-300.0),
            ratio_db=pl.when(
                (pl.col("ratio_re") ** 2 + pl.col("ratio_im") ** 2) > 1e-30
            )
            .then(
                20.0
                * (pl.col("ratio_re") ** 2 + pl.col("ratio_im") ** 2).sqrt().log10()
            )
            .otherwise(-300.0),
        )

    @classmethod
    def _compute_sensitivity(cls, df: pl.DataFrame) -> pl.DataFrame:
        """Calculates derivative of monopulse ratio using explicitly ordered window intervals."""
        # Enforce strict window context sorting rule
        ev_col = DataHeader.ELEVATION_RAD
        az_col = DataHeader.AZIMUTH_RAD

        return (
            df.sort([ev_col, az_col])
            .with_columns(
                d_ratio=(
                    pl.col("ratio_im").shift(-1).over(ev_col, order_by=az_col)
                    - pl.col("ratio_im").shift(1).over(ev_col, order_by=az_col)
                ),
                d_theta=(
                    pl.col(az_col).shift(-1).over(ev_col, order_by=az_col)
                    - pl.col(az_col).shift(1).over(ev_col, order_by=az_col)
                ),
            )
            .with_columns(
                sensitivity_per_rad=pl.when(
                    pl.col("d_theta").is_not_null() & (pl.col("d_theta").abs() > 1e-12)
                )
                .then(pl.col("d_ratio") / pl.col("d_theta"))
                .otherwise(None)
            )
        )

    @classmethod
    def calculate_monopulse(
        cls,
        frequency: Frequency,
        array_beam: pl.DataFrame,
        geometry: pl.DataFrame,
        condition: pl.Expr,
    ) -> pl.DataFrame:
        """Main pipeline execution for radar monopulse evaluation."""
        k = (2 * np.pi) * frequency.Hz / RadarConstants.c
        array_beam_with_id = array_beam.with_row_index("grid_id")

        # Pipeline Processing Steps
        geometry_processed = cls._preprocess_geometry(geometry, condition)
        working_df = cls._compute_element_responses(
            array_beam_with_id, geometry_processed, k
        )
        calculated_df = cls._aggregate_and_compute_ratios(
            working_df, array_beam_with_id
        )
        final_calc = cls._compute_sensitivity(calculated_df)

        # Format output DataFrame schema matching original architecture
        return (
            final_calc.with_columns(
                [
                    pl.col("sum_db").alias(DataHeader.MONOPULSE_SUMATION_DB),
                    pl.col("diff_db").alias(DataHeader.MONOPULSE_DIFFERENCE_DB),
                    pl.col("ratio_db").alias(DataHeader.MONOPULSE_RAIO_DB),
                    pl.col("sensitivity_per_rad").alias(
                        DataHeader.MONOPULSE_SENSITIVITY
                    ),
                ]
            )
            .select(
                [
                    "grid_id",
                    *array_beam.columns,
                    DataHeader.MONOPULSE_SUMATION_DB,
                    DataHeader.MONOPULSE_DIFFERENCE_DB,
                    DataHeader.MONOPULSE_RAIO_DB,
                    DataHeader.MONOPULSE_SENSITIVITY,
                ]
            )
            .sort("grid_id")
            .drop("grid_id")
        )
