import polars as pl
import plotly.express as px
from abc import ABC, abstractmethod
from typing import cast

from radar.utils.typing import (
    DataHeader,
)


class GeometryInterface(ABC):
    """Abstract base class defining the interface for geometry-related operations."""

    @abstractmethod
    def geometry(self) -> None:
        """Process or retrieve geometry data.

        This method must be implemented by subclasses to define specific
        geometry behaviors.
        """
        pass


class Geometry:
    """A utility class for handling and visualizing spatial geometry data.

    Attributes:
        FIGURE_WIDTH (int): Default width of the generated Plotly figure in pixels.
        FIGURE_HEIGHT (int): Default height of the generated Plotly figure in pixels.
        DISPLAY_GRID_PIXELS (int): Pixel spacing configuration for the grid layout.
    """

    FIGURE_WIDTH = 800
    FIGURE_HEIGHT = 800
    DISPLAY_GRID_PIXELS = 100

    @classmethod
    def _image(
        cls,
        df: pl.DataFrame,
    ) -> None:
        """Generates and displays a symmetric square scatter plot of X and Y positions.

        This method plots positional data from a Polars DataFrame, determines a
        symmetric bounding box based on the maximum data extent, and locks the
        aspect ratio to ensure accurate geometric scaling.

        Args:
            df (pl.DataFrame): The input DataFrame containing spatial coordinates.
                Must contain columns defined by `DataHeader.X_POS_M` and
                `DataHeader.Y_POS_M`.

        Raises:
            ValueError: If the DataFrame is empty or if the min/max values for
                the position coordinates cannot be calculated.
        """
        # Assuming you have a column name for the color data, e.g., DataHeader.COLOR_VAL
        fig = px.scatter(
            df,
            x=DataHeader.X_POS_M,
            y=DataHeader.Y_POS_M,
            color=df[DataHeader.GEOM_AMP_GAIN_DB],
            color_continuous_scale=px.colors.sequential.Viridis,
            width=cls.FIGURE_WIDTH,
            height=cls.FIGURE_HEIGHT,
        )

        values = [
            df[DataHeader.X_POS_M].min(),
            df[DataHeader.X_POS_M].max(),
            df[DataHeader.Y_POS_M].min(),
            df[DataHeader.Y_POS_M].max(),
        ]

        if any(v is None for v in values):
            raise ValueError("Empty dataframe")

        values = cast(tuple[float, float], values)

        # Pad the bounding box by 10% to ensure points near edges aren't cut off
        max_extent = max(abs(v) for v in values) * 1.1

        fig.update_xaxes(range=[-max_extent, max_extent])
        fig.update_yaxes(range=[-max_extent, max_extent])

        fig.update_layout(
            title="Position Distribution",
            yaxis_scaleanchor="x",  # keeps aspect ratio square
            coloraxis_colorbar=dict(
                title="Element Scaling [dB]"
            ),  # <-- Optional: Customize the color axis title
        )

        fig.show()
