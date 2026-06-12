from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
import polars as pl

from radar.utils.typing import DataHeader, Angle
from radar.utils.calculate.convert import to_db
from radar.utils.typing.enums import PhaseUnit
from radar.utils.typing.validator import AngleBound


class Pattern(ABC):
    """Abstract base class defining the required interface for radar beam patterns."""

    @abstractmethod
    def calculate_pattern(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculates the pattern's gain metrics and appends them to the DataFrame.

        Args:
            df (pl.DataFrame): The input spatial coordinate dataset.

        Returns:
            pl.DataFrame: The modified DataFrame including linear and dB gains.
        """
        pass


class CustomPattern(Pattern):
    """A customizable data-driven antenna pattern built from empirical lookup data.

    Attributes:
        df (pl.DataFrame): The underlying reference pattern dataset.
    """

    def __init__(self, df: pl.DataFrame):
        """Initializes the CustomPattern lookup table and calculates boundaries.

        Args:
            df (pl.DataFrame): Dataframe containing the reference pattern mapping.
                Must include azimuth, elevation, and linear gain columns.

        Raises:
            ValueError: If any required configuration headers are missing.
        """
        required_init = [
            DataHeader.AZIMUTH_DEG,
            DataHeader.ELEVATION_DEG,
            DataHeader.BEAM_GAIN_LINEAR,
        ]
        self._validate_presence(df, required_init)

        # 1. Use .item() to extract single scalar metrics cleanly
        self._az_bound = AngleBound(
            (
                Angle(
                    df.select(pl.col(DataHeader.AZIMUTH_DEG).min()).item(),
                    PhaseUnit.DEGREE,
                ),
                Angle(
                    df.select(pl.col(DataHeader.AZIMUTH_DEG).max()).item(),
                    PhaseUnit.DEGREE,
                ),
            )
        )
        self._el_bound = AngleBound(
            (
                Angle(
                    df.select(pl.col(DataHeader.ELEVATION_DEG).min()).item(),
                    PhaseUnit.DEGREE,
                ),
                Angle(
                    df.select(pl.col(DataHeader.ELEVATION_DEG).max()).item(),
                    PhaseUnit.DEGREE,
                ),
            )
        )

        # 2. Append the dB column if it doesn't exist yet
        if DataHeader.BEAM_GAIN_DB not in df.columns:
            df = df.with_columns(
                to_db(pl.col(DataHeader.BEAM_GAIN_LINEAR)).alias(
                    DataHeader.BEAM_GAIN_DB
                )
            )

        self.df = df

    def _validate_presence(self, df: pl.DataFrame, columns: list[str]) -> None:
        """Internal helper to ensure columns exist before executing operations.

        Args:
            df (pl.DataFrame): Target Polars DataFrame to inspect.
            columns (list[str]): List of expected column names.

        Raises:
            ValueError: If one or more columns are not present in the dataframe.
        """
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Required columns missing from Polars DataFrame: {missing}"
            )

    def calculate_pattern(self, df: pl.DataFrame) -> pl.DataFrame:
        """Maps input coordinates against the custom reference pattern.

        Performs strict shape, coordinate, and boundary matches before executing
        an inner join to assign calculated gains to incoming data rows.

        Args:
            df (pl.DataFrame): The spatial grid coordinates to compute patterns for.

        Returns:
            pl.DataFrame: Joined dataframe incorporating historical reference gains.

        Raises:
            ValueError: If required headers are missing, coordinate boundaries
                do not match, or coordinates are missing from the lookup reference.
        """
        lookup_keys = [DataHeader.AZIMUTH_DEG, DataHeader.ELEVATION_DEG]
        self._validate_presence(df, lookup_keys)

        unique_az = df[DataHeader.AZIMUTH_DEG].unique()
        unique_el = df[DataHeader.ELEVATION_DEG].unique()

        actual_az_min, actual_az_max = unique_az.min(), unique_az.max()
        actual_el_min, actual_el_max = unique_el.min(), unique_el.max()

        # Validates that grid bounds align perfectly with the source data boundaries
        if (
            actual_az_min != self._az_bound[0].deg
            or actual_az_max != self._az_bound[1].deg
            or actual_el_min != self._el_bound[0].deg
            or actual_el_max != self._el_bound[1].deg
        ):
            raise ValueError(
                f"Surface mismatch. Input corners must exactly match pattern corners: "
                f"Az ({self._az_bound[0].deg}, {self._az_bound[1].deg}), "
                f"El ({self._el_bound[0].deg}, {self._el_bound[1].deg})"
            )

        expected_row_count = len(unique_az) * len(unique_el)
        if len(df) != expected_row_count:
            raise ValueError(
                f"Incomplete surface. Expected {expected_row_count} points "
                f"({len(unique_az)} az x {len(unique_el)} el), but got {len(df)}."
            )

        result = df.join(
            self.df.select(
                [*lookup_keys, DataHeader.BEAM_GAIN_LINEAR, DataHeader.BEAM_GAIN_DB]
            ),
            on=lookup_keys,
            how="inner",
        )

        if len(result) < len(df):
            missing_count = len(df) - len(result)
            raise ValueError(
                f"Surface mapping failed: {missing_count} coordinate pairs are "
                f"missing from the beam pattern lookup."
            )

        return result


class Isotropic(Pattern):
    """An ideal isotropic antenna pattern radiating uniformly with 0 dB gain."""

    def calculate_pattern(self, df: pl.DataFrame) -> pl.DataFrame:
        """Appends static isotropic gains (0 dB / 1.0 Linear) to the DataFrame.

        Args:
            df (pl.DataFrame): Input dataset.

        Returns:
            pl.DataFrame: Modified DataFrame with uniform gain entries.
        """
        return df.with_columns(
            [
                pl.lit(0).alias(DataHeader.BEAM_GAIN_DB),
                pl.lit(1).alias(DataHeader.BEAM_GAIN_LINEAR),
            ]
        )


class Cosine(Pattern):
    """A hemispherical cosine-power beam pattern model."""

    def __init__(self, order: int = 1) -> None:
        """Initializes the Cosine model with a mathematical scaling power factor.

        Args:
            order (int, optional): The exponential factor modifying the cosine window.
                Higher values yield narrower main beams. Defaults to 1.
        """
        self._order = order

    def calculate_pattern(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculates directional cosine gains relative to boresight at (0,0).

        Args:
            df (pl.DataFrame): Input dataset containing `AZIMUTH_RAD` and `ELEVATION_RAD`.

        Returns:
            pl.DataFrame: Dataset containing appended cosine gain metrics.
        """
        cos_theta_expr = (
            pl.col(DataHeader.AZIMUTH_RAD).cos()
            * pl.col(DataHeader.ELEVATION_RAD).cos()
        )

        # 2. Clip at 0 and raise to the power of self._order
        mag_linear_expr = (
            pl.when(cos_theta_expr > 0).then(cos_theta_expr).otherwise(0.0)
            ** self._order
        )

        # 3. Assuming to_db can be expressed as a formula (e.g., 10 * mag_linear.log10())
        # If to_db is a custom complex function, you might need to use .map_batches()
        mag_db_expr = to_db(mag_linear_expr)

        return df.with_columns(
            [
                mag_linear_expr.alias(DataHeader.BEAM_GAIN_LINEAR),
                mag_db_expr.alias(DataHeader.BEAM_GAIN_DB),
            ]
        )


class Gaussian(Pattern):
    """A mathematical Gaussian distribution beam pattern model."""

    def __init__(
        self,
        beam_width: Tuple[Angle, Angle],
    ) -> None:
        """Initializes the Gaussian pattern with designated half-power beamwidths.

        Args:
            beam_width (Tuple[Angle, Angle]): Target sizing bounds configured as
                (Azimuth HPBW, Elevation HPBW).
        """
        self._beam_width = beam_width

    def calculate_pattern(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculates normal Gaussian scaling gains across spatial dimensions.

        Args:
            df (pl.DataFrame): Input dataset containing `AZIMUTH_RAD` and `ELEVATION_RAD`.

        Returns:
            pl.DataFrame: Dataset containing appended Gaussian gain metrics.
        """
        bw_az, bw_el = self._beam_width[0].rad, self._beam_width[1].rad

        sigma_const = -4 * np.log(2)

        mag_linear_expr = (
            sigma_const
            * (
                (pl.col(DataHeader.AZIMUTH_RAD) / bw_az) ** 2
                + (pl.col(DataHeader.ELEVATION_RAD) / bw_el) ** 2
            )
        ).exp()

        mag_db_expr = to_db(mag_linear_expr)

        return df.with_columns(
            [
                mag_linear_expr.alias(DataHeader.BEAM_GAIN_LINEAR),
                mag_db_expr.alias(DataHeader.BEAM_GAIN_DB),
            ]
        )


class Sinc(Pattern):
    """An analytical Sinc (cardinal sine) distribution beam pattern model."""

    def __init__(self, beam_width: Tuple[Angle, Angle]) -> None:
        """Initializes the Sinc pattern with designated half-power beamwidths.

        Args:
            beam_width (Tuple[Angle, Angle]): Target sizing bounds configured as
                (Azimuth HPBW, Elevation HPBW).
        """
        self._beam_width = beam_width

    def calculate_pattern(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculates structural Sinc gains representing uniform aperture characteristics.

        Args:
            df (pl.DataFrame): Input dataset containing `AZIMUTH_RAD` and `ELEVATION_RAD`.

        Returns:
            pl.DataFrame: Dataset containing appended Sinc gain metrics.
        """
        bw_az, bw_el = self._beam_width[0].rad, self._beam_width[1].rad

        # Constant for Sinc Half-Power Beamwidth (HPBW)
        # 1.3915 is the value where sinc^2(x) = 0.5
        k = 1.3915 * 2

        # np.sinc in numpy is sin(pi*x)/(pi*x)
        arg_az_expr = (k * df[DataHeader.AZIMUTH_RAD] / bw_az) / np.pi
        arg_el_expr = (k * df[DataHeader.ELEVATION_RAD] / bw_el) / np.pi

        def polars_sinc(x_expr):
            pi_x = np.pi * x_expr
            return pl.when(x_expr == 0).then(1.0).otherwise(pi_x.sin() / pi_x)

        # 3. Compute the absolute linear magnitude
        mag_linear_expr = (polars_sinc(arg_az_expr) * polars_sinc(arg_el_expr)).abs()

        mag_db_expr = to_db(mag_linear_expr)

        return df.with_columns(
            [
                mag_db_expr.alias(DataHeader.BEAM_GAIN_DB),
                mag_linear_expr.alias(DataHeader.BEAM_GAIN_LINEAR),
            ]
        )
