"""
Geometry Layout Management Module.

Provides abstractions and utility classes to parse spatial coordinates from
Polars DataFrames and map them into Manim 3D visualization objects (VGroups).
"""

from abc import ABC, abstractmethod
import numpy.typing as npt
import polars as pl
from manim import Dot, ManimColor, ThreeDAxes, VGroup
from radar.utils.typing.constants import DataHeader


class GeometryInterface(ABC):
    """
    Abstract Base Class acting as an interface for all radar geometry layouts.

    Enforces unified generation signatures across custom layout variations
    (e.g., Grid, Circular, Cross).
    """

    @abstractmethod
    def geometry(self, position: npt.NDArray, colour: ManimColor) -> VGroup:
        """
        Generate the Manim VGroup representation of the layout geometry.

        Args:
            position (npt.NDArray): The 3D translation vector [x, y, z] for the layout.
            colour (ManimColor): The structural color assigned to the rendered points.

        Returns:
            VGroup: A collection containing the spatial mobjects.
        """
        pass


class Geometry:
    """
    Utility handler for converting tabular position data into 3D Manim spaces.
    """

    @classmethod
    def dots_3d(
        cls, df: pl.DataFrame, position: npt.NDArray, colour: ManimColor
    ) -> VGroup:
        """
        Convert a DataFrame of spatial elements into a 3D space of positioned Dots.

        Args:
            df (pl.DataFrame): Tabular dataset containing spatial coordinates.
            position (npt.NDArray): Anchor position vector for the underlying 3D axes.
            colour (ManimColor): Color token applied to all generated dots.

        Returns:
            VGroup: A collection containing the translated 3D Dot mobjects.
        """
        return cls._df_to_dots(df, position, colour)

    @staticmethod
    def _df_to_dots(
        df: pl.DataFrame, position: npt.NDArray, color: ManimColor
    ) -> VGroup:
        """
        Internal implementation mapping DataFrame positions onto a localized 3D coordinate system.

        Changed from @classmethod to @staticmethod since it does not mutate or reference
        the class state directly.

        Args:
            df (pl.DataFrame): Tabular dataset containing element points.
            position (npt.NDArray): Origin anchor offset for the workspace axes.
            color (ManimColor): Color attribute for the resulting dots.

        Returns:
            VGroup: The populated group containing structural array dots.
        """
        # Global boundaries for standard mapping reference bounds
        axes_config = {
            "x_range": [-1, 1, 0.2],
            "y_range": [-1, 1, 0.2],
            "z_range": [-1, 1, 0.2],
            "x_length": 5,
            "y_length": 5,
            "z_length": 5,
        }

        # Initialize and translate reference axes anchor framework
        axes = ThreeDAxes(**axes_config).shift(position)
        dots = VGroup()

        # Parse and iterate rows out of target tracking dataframe
        for row in df.iter_rows(named=True):
            pos = axes.coords_to_point(
                row[DataHeader.X_POS_M], row[DataHeader.Y_POS_M], 0
            )
            dots.add(Dot(point=pos, color=color, radius=0.08))

        return dots
