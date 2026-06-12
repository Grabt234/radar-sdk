from abc import ABC, abstractmethod
from manim import ManimColor, VGroup
import numpy as np
import numpy.typing as npt
import polars as pl

from radar.utils.typing import DataHeader, ArrayOrientation, Distance, Length
from radar.utils import plotter, animate
from radar.utils.calculate import convert

from pydantic import PositiveInt


class Geometry(ABC):
    """Abstract base class establishing structural data and plotting for radar arrays.

    Provides core storage properties, enforces automatic default column configuration,
    and routes spatial positioning data to specialized visual plot renderers.
    """

    def __init__(self) -> None:
        """Initializes the base array geometry and attaches its plot interface."""
        super().__init__()

        self.plot = self.Plot(self)
        self.animate = self.Animate(self)

        # Automatically generate the dataframe structure via subclass specifications
        self.df = self._generate_positions_df()

        # Enforce default gains and phases seamlessly on initialization
        self._set_default_gain_phase()

    @abstractmethod
    def _generate_positions_df(self) -> pl.DataFrame:
        """Abstract method that subclasses must implement to yield their initial position dataframe."""
        pass

    def _set_default_gain_phase(self) -> None:
        self.df = self.df.with_columns(
            [
                pl.lit(0).alias(DataHeader.GEOM_AMP_GAIN_DB),
                pl.lit(1).alias(DataHeader.GEOM_AMP_GAIN_LIN),
                pl.lit(0).alias(DataHeader.GEOM_PHASE_SHIFTER_PHASE_DEG),
                pl.lit(0).alias(DataHeader.GEOM_PHASE_SHIFTER_PHASE_RAD),
            ]
        )

    @property
    def phases(self):
        raise AttributeError("To access phases call geometry")

    @phases.setter
    def phases(self, df: pl.DataFrame) -> None:
        """Sets the phase columns from a provided Polars DataFrame.

        Strictly requires EITHER Radian OR Degree columns to be present, but not both.
        """
        rad_col = DataHeader.GEOM_PHASE_SHIFTER_PHASE_RAD
        deg_col = DataHeader.GEOM_PHASE_SHIFTER_PHASE_DEG

        has_rad = rad_col in df.columns
        has_deg = deg_col in df.columns

        if not (has_rad ^ has_deg):
            raise ValueError(
                f"Input DataFrame must contain exactly one of '{rad_col}' or '{deg_col}', but not both."
            )

        target_col = rad_col if has_rad else deg_col
        self._validate_input_df(df, required_column=target_col)

        if has_rad:
            rad_series = df.get_column(rad_col)
            deg_series = pl.Series(np.rad2deg(rad_series))
        else:
            deg_series = df.get_column(deg_col)
            rad_series = pl.Series(np.deg2rad(deg_series))

        self.df = self.df.with_columns(
            [rad_series.alias(rad_col), deg_series.alias(deg_col)]
        )

    @property
    def gains(self):
        raise AttributeError("To access gains call geometry")

    @gains.setter
    def gains(self, df: pl.DataFrame) -> None:
        """Sets the gain columns from a provided Polars DataFrame.

        Strictly requires EITHER Decibel OR Linear columns to be present, but not both.
        """
        db_col = DataHeader.GEOM_AMP_GAIN_DB
        lin_col = DataHeader.GEOM_AMP_GAIN_LIN

        has_db = db_col in df.columns
        has_lin = lin_col in df.columns

        if not (has_db ^ has_lin):
            raise ValueError(
                f"Input DataFrame must contain exactly one of '{db_col}' or '{lin_col}', but not both."
            )

        target_col = db_col if has_db else lin_col
        self._validate_input_df(df, required_column=target_col)

        if has_db:
            db_series = df.get_column(db_col)
            lin_series = pl.Series(convert.from_db(db_series))
        else:
            lin_series = df.get_column(lin_col)
            db_series = pl.Series(convert.to_db(lin_series))

        self.df = self.df.with_columns(
            [db_series.alias(db_col), lin_series.alias(lin_col)]
        )

    def _validate_input_df(self, df: pl.DataFrame, required_column: str) -> None:
        """Helper method to validate that the input DataFrame matches the current geometry."""
        if not isinstance(df, pl.DataFrame):
            raise TypeError("Input must be a Polars DataFrame")

        if required_column not in df.columns:
            raise ValueError(
                f"Input DataFrame must contain a '{required_column}' column"
            )

        pos_cols = [DataHeader.X_POS_M, DataHeader.Y_POS_M]
        for col in pos_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Input DataFrame is missing required coordinate column: '{col}'"
                )

        current_coords = self.df.select(pos_cols)
        input_coords = df.select(pos_cols)

        if not current_coords.equals(input_coords):
            raise ValueError(
                "Element-wise geometric mismatch! The X and Y coordinates "
                "in the input DataFrame do not match this Geometry instance row-for-row."
            )

    @property
    def geometry(self) -> pl.DataFrame:
        """Returns the internal Polars DataFrame containing element physical positions."""
        return self.df

    class Plot(plotter.GeometryInterface):
        """Inner bridge class handling plotting commands for its parent Geometry."""

        def __init__(self, geometry: "Geometry") -> None:
            super().__init__()
            self._geometry = geometry
            self._plotter = plotter.Geometry

        def geometry(self) -> None:
            self._plotter._image(self._geometry.geometry)

    class Animate(animate.GeometryInterface):
        """Inner bridge class handling animate commands for its parent Geometry."""

        def __init__(self, geometry: "Geometry") -> None:
            super().__init__()
            self._geometry = geometry
            self._animator = animate.Geometry

        def geometry(self, position: npt.NDArray, colour: ManimColor) -> VGroup:
            return self._animator.dots_3d(self._geometry.geometry, position, colour)


class CustomGeometry(Geometry):
    """Defines an arbitrary custom geometry layout from predefined position arrays."""

    def __init__(self, x: npt.NDArray, y: npt.NDArray) -> None:
        if x.size != y.size:
            raise ValueError(
                f"Geometry size not the same x = {len(x)} and y = {len(y)}"
            )

        self._x = x
        self._y = y
        super().__init__()

    def _generate_positions_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                DataHeader.X_POS_M: self._x,
                DataHeader.Y_POS_M: self._y,
            }
        )


class Linear(Geometry):
    """Generates a center-aligned 1D linear sensor array along a targeted axis."""

    def __init__(
        self, elements: PositiveInt, orientation: ArrayOrientation, spacing: Distance
    ) -> None:
        self._elements = elements
        self._orientation = orientation
        self._spacing = spacing
        super().__init__()

    def _generate_positions_df(self) -> pl.DataFrame:
        raw_positions = np.arange(self._elements) * self._spacing.m
        positions = raw_positions - np.mean(raw_positions)

        if self._orientation == ArrayOrientation.AZIMUTH:
            data = {
                DataHeader.X_POS_M: positions,
                DataHeader.Y_POS_M: np.zeros(self._elements),
            }
        else:
            data = {
                DataHeader.X_POS_M: np.zeros(self._elements),
                DataHeader.Y_POS_M: positions,
            }
        return pl.DataFrame(data)


class Cross(Geometry):
    """Generates a center-aligned cross (cruciform) antenna array layout."""

    def __init__(
        self,
        azimuth_elements: PositiveInt,
        elevation_elements: PositiveInt,
        spacing: Distance,
    ) -> None:
        self._azimuth_elements = azimuth_elements
        self._elevation_elements = elevation_elements
        self._spacing = spacing
        super().__init__()

    def _generate_positions_df(self) -> pl.DataFrame:
        az_raw = np.arange(self._azimuth_elements) * self._spacing.m
        el_raw = np.arange(self._elevation_elements) * self._spacing.m

        azimuth_positions = az_raw - np.mean(az_raw)
        elevation_positions = el_raw - np.mean(el_raw)

        x_horizontal = azimuth_positions
        y_horizontal = np.zeros_like(azimuth_positions)

        x_vertical = np.zeros_like(elevation_positions)
        y_vertical = elevation_positions

        horizontal = set(zip(x_horizontal, y_horizontal))
        vertical = set(zip(x_vertical, y_vertical))
        positions = np.array(list(horizontal.union(vertical)))

        data = {
            DataHeader.X_POS_M: positions[:, 0],
            DataHeader.Y_POS_M: positions[:, 1],
        }
        return pl.DataFrame(data).unique()


class Circular(Geometry):
    """Generates a circular ring array layout with uniformly distributed elements."""

    def __init__(self, elements: PositiveInt, radius: Length) -> None:
        self._elements = elements
        self._radius = radius
        super().__init__()

    def _generate_positions_df(self) -> pl.DataFrame:
        angles = np.linspace(0, 2 * np.pi, self._elements, endpoint=False)
        x = self._radius.m * np.cos(angles)
        y = self._radius.m * np.sin(angles)

        return pl.DataFrame({DataHeader.X_POS_M: x, DataHeader.Y_POS_M: y})


class Grid(Geometry):
    """Generates a center-aligned 2D planar rectangular matrix/grid array layout."""

    def __init__(
        self,
        azimuth_elements: PositiveInt,
        elevation_elements: PositiveInt,
        spacing: Distance,
    ) -> None:
        self._azimuth_elements = azimuth_elements
        self._elevation_elements = elevation_elements
        self._spacing = spacing
        super().__init__()

    def _generate_positions_df(self) -> pl.DataFrame:
        az_raw = np.arange(self._azimuth_elements) * self._spacing.m
        el_raw = np.arange(self._elevation_elements) * self._spacing.m

        azimuth_positions = az_raw - np.mean(az_raw)
        elevation_positions = el_raw - np.mean(el_raw)

        xv, yv = np.meshgrid(azimuth_positions, elevation_positions)

        data = {
            DataHeader.X_POS_M: xv.flatten(),
            DataHeader.Y_POS_M: yv.flatten(),
        }
        return pl.DataFrame(data).unique()
