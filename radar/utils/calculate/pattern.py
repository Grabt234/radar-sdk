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

        metrics = df.select(
            [
                pl.col(DataHeader.AZIMUTH_DEG).min().alias("az_min"),
                pl.col(DataHeader.AZIMUTH_DEG).max().alias("az_max"),
                pl.col(DataHeader.ELEVATION_DEG).min().alias("el_min"),
                pl.col(DataHeader.ELEVATION_DEG).max().alias("el_max"),
            ]
        )

        self._az_bound = AngleBound(
            (
                Angle(metrics["az_min"][0], PhaseUnit.DEGREE),
                Angle(metrics["az_max"][0], PhaseUnit.DEGREE),
            )
        )
        self._el_bound = AngleBound(
            (
                Angle(metrics["el_min"][0], PhaseUnit.DEGREE),
                Angle(metrics["el_max"][0], PhaseUnit.DEGREE),
            )
        )

        if DataHeader.BEAM_GAIN_DB not in df.columns:
            df = df.with_columns(
                pl.col(DataHeader.BEAM_GAIN_LINEAR)
                .map_batches(to_db)
                .alias(DataHeader.BEAM_GAIN_DB)
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
        az = df.select(DataHeader.AZIMUTH_RAD).to_numpy()
        el = df.select(DataHeader.ELEVATION_RAD).to_numpy()

        # Typical implementation: cos(theta) where theta is the angle from boresight
        # Assuming boresight is at (0,0)
        cos_theta = np.cos(az) * np.cos(el)

        # Clip to 0 to ensure no back-lobes (hemispherical)
        mag_linear = np.maximum(0, cos_theta) ** self._order
        mag_db = to_db(mag_linear)

        return df.with_columns(
            [
                pl.Series(DataHeader.BEAM_GAIN_DB, mag_db.ravel()),
                pl.Series(DataHeader.BEAM_GAIN_LINEAR, mag_linear.ravel()),
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
        mag_linear = np.exp(
            sigma_const
            * (
                (df.select(DataHeader.AZIMUTH_RAD).to_numpy() / bw_az) ** 2
                + (df.select(DataHeader.ELEVATION_RAD).to_numpy() / bw_el) ** 2
            )
        )

        mag_db = to_db(mag_linear)

        return df.with_columns(
            [
                pl.Series(DataHeader.BEAM_GAIN_DB, mag_db.ravel()),
                pl.Series(DataHeader.BEAM_GAIN_LINEAR, mag_linear.ravel()),
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

        az = df.select(DataHeader.AZIMUTH_RAD).to_numpy()
        el = df.select(DataHeader.ELEVATION_RAD).to_numpy()

        # Constant for Sinc Half-Power Beamwidth (HPBW)
        # 1.3915 is the value where sinc^2(x) = 0.5
        k = 1.3915 * 2

        # np.sinc in numpy is sin(pi*x)/(pi*x)
        arg_az = (k * az / bw_az) / np.pi
        arg_el = (k * el / bw_el) / np.pi

        mag_linear = np.abs(np.sinc(arg_az) * np.sinc(arg_el))
        mag_db = to_db(mag_linear)

        return df.with_columns(
            [
                pl.Series(DataHeader.BEAM_GAIN_DB, mag_db.ravel()),
                pl.Series(DataHeader.BEAM_GAIN_LINEAR, mag_linear.ravel()),
            ]
        )
