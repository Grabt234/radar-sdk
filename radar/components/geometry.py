from abc import ABC
from manim import ManimColor, VGroup
import numpy as np
import numpy.typing as npt
import polars as pl

from radar.utils.typing import DataHeader, ArrayOrientation
from radar.utils import plotter, animate

from pydantic import PositiveInt

from radar.utils.typing import Distance, Length


class Geometry(ABC):
    """Abstract base class establishing structural data and plotting for radar arrays.

    Provides core storage properties and routes spatial positioning data to
    specialized visual plot renderers.
    """

    def __init__(self) -> None:
        """Initializes the base array geometry and attaches its plot interface."""
        super().__init__()
        self.plot = self.Plot(self)
        self.animate = self.Animate(self)

    df: pl.DataFrame

    @property
    def geometry(self) -> pl.DataFrame:
        """Returns the internal Polars DataFrame containing element physical positions."""
        return self.df

    class Plot(plotter.GeometryInterface):
        """Inner bridge class handling plotting commands for its parent Geometry."""

        def __init__(self, geometry: "Geometry") -> None:
            """Initializes the plotting handler bound to a Geometry context.

            Args:
                geometry (Geometry): The parent instance supplying positioning records.
            """
            super().__init__()
            self._geometry = geometry
            self._plotter = plotter.Geometry

        def geometry(self) -> None:
            """Dispatches coordinate snapshots to render an image of the antenna layout."""
            self._plotter._image(self._geometry.geometry)

    class Animate(animate.GeometryInterface):
        """Inner bridge class handling animate commands for its parent Geometry."""

        def __init__(self, geometry: "Geometry") -> None:
            """Initializes the animation handler bound to a Geometry context.

            Args:
                geometry (Geometry): The parent instance supplying positioning records.
            """
            super().__init__()
            self._geometry = geometry
            self._animator = animate.Geometry

        def geometry(self, position: npt.NDArray, colour: ManimColor) -> VGroup:
            """Dispatches coordinate snapshots to render an image of the antenna layout."""
            return self._animator.dots_3d(self._geometry.geometry, position, colour)


class CustomGeometry(Geometry):
    """Defines an arbitrary custom geometry layout from predefined position arrays."""

    def __init__(self, x: npt.NDArray, y: npt.NDArray) -> None:
        """Initializes a CustomGeometry layout map.

        Args:
            x (npt.NDArray): 1D array indicating the positional offset along the X-axis (meters).
            y (npt.NDArray): 1D array indicating the positional offset along the Y-axis (meters).

        Raises:
            ValueError: If the coordinate arrays have mismatched dimensions.
        """
        if x.size != y.size:
            raise ValueError(
                f"Geometry size not the same x = {len(x)} and y = {len(y)}"
            )

        data = {
            DataHeader.X_POS_M: x,
            DataHeader.Y_POS_M: y,
        }
        self.df = pl.DataFrame(data)


class Linear(Geometry):
    """Generates a center-aligned 1D linear sensor array along a targeted axis."""

    def __init__(
        self,
        elements: PositiveInt,
        orientation: ArrayOrientation,
        spacing: Distance,
    ) -> None:
        """Initializes a uniform center-aligned Linear sensor array.

        Args:
            elements (PositiveInt): The total number of elements in the array.
            orientation (ArrayOrientation): Spatial vector layout axis context
                (e.g., AZIMUTH or ELEVATION).
            spacing (Distance): The uniform structural gap separating adjacent elements.
        """
        super().__init__()

        # Generate zero-indexed raw positions and center by subtracting the mean
        raw_positions = np.arange(elements) * spacing.m
        positions = raw_positions - np.mean(raw_positions)

        if orientation == ArrayOrientation.AZIMUTH:
            data = {
                DataHeader.X_POS_M: positions,
                DataHeader.Y_POS_M: np.zeros(elements),
            }
        else:
            data = {
                DataHeader.X_POS_M: np.zeros(elements),
                DataHeader.Y_POS_M: positions,
            }

        self.df = pl.DataFrame(data)


class Cross(Geometry):
    """Generates a center-aligned cross (cruciform) antenna array layout."""

    def __init__(
        self,
        azimuth_elements: PositiveInt,
        elevation_elements: PositiveInt,
        spacing: Distance,
    ) -> None:
        """Initializes a Cross configuration array.

        Computes independent horizontal and vertical spans, merging them
        together seamlessly to eliminate duplicate origin elements.

        Args:
            azimuth_elements (PositiveInt): Number of sensors forming the horizontal arm.
            elevation_elements (PositiveInt): Number of sensors forming the vertical arm.
            spacing (Distance): The uniform spatial gap separating adjacent elements.
        """
        super().__init__()

        # Generate and center positions for both axes independently
        az_raw = np.arange(azimuth_elements) * spacing.m
        el_raw = np.arange(elevation_elements) * spacing.m

        azimuth_positions = az_raw - np.mean(az_raw)
        elevation_positions = el_raw - np.mean(el_raw)

        # Horizontal arm
        x_horizontal = azimuth_positions
        y_horizontal = np.zeros_like(azimuth_positions)

        # Vertical arm
        x_vertical = np.zeros_like(elevation_positions)
        y_vertical = elevation_positions

        # Combine positions using sets to seamlessly drop overlapping elements
        horizontal = set(zip(x_horizontal, y_horizontal))
        vertical = set(zip(x_vertical, y_vertical))
        positions = np.array(list(horizontal.union(vertical)))

        data = {
            DataHeader.X_POS_M: positions[:, 0],
            DataHeader.Y_POS_M: positions[:, 1],
        }

        self.df = pl.DataFrame(data).unique()


class Circular(Geometry):
    """Generates a circular ring array layout with uniformly distributed elements."""

    def __init__(self, elements: PositiveInt, radius: Length) -> None:
        """Initializes a Circular configuration ring array.

        Args:
            elements (PositiveInt): The total number of sensors distributed on the perimeter.
            radius (Length): The physical radius distance from the origin point to the elements.
        """
        super().__init__()
        angles = np.linspace(0, 2 * np.pi, elements, endpoint=False)
        x = radius.m * np.cos(angles)
        y = radius.m * np.sin(angles)

        self.df = pl.DataFrame({DataHeader.X_POS_M: x, DataHeader.Y_POS_M: y})


class Grid(Geometry):
    """Generates a center-aligned 2D planar rectangular matrix/grid array layout."""

    def __init__(
        self,
        azimuth_elements: PositiveInt,
        elevation_elements: PositiveInt,
        spacing: Distance,
    ) -> None:
        """Initializes a planar rectangular Grid configuration array.

        Args:
            azimuth_elements (PositiveInt): Total row count along the horizontal layout grid.
            elevation_elements (PositiveInt): Total column count along the vertical layout grid.
            spacing (Distance): Uniform structural spacing separating rows and columns.
        """
        super().__init__()

        # Generate and center positions for both grid dimensions
        az_raw = np.arange(azimuth_elements) * spacing.m
        el_raw = np.arange(elevation_elements) * spacing.m

        azimuth_positions = az_raw - np.mean(az_raw)
        elevation_positions = el_raw - np.mean(el_raw)

        # Create 2D coordinates grid
        xv, yv = np.meshgrid(azimuth_positions, elevation_positions)

        data = {
            DataHeader.X_POS_M: xv.flatten(),
            DataHeader.Y_POS_M: yv.flatten(),
        }

        self.df = pl.DataFrame(data).unique()
