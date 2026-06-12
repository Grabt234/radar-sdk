from abc import ABC, abstractmethod
import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from radar.utils.typing import (
    AmplitudeUnit,
    DataHeader,
    DirectionDomain,
    FigureType,
)
from radar.utils.typing.enums import PhaseUnit
from radar.utils.typing.units import Angle, Frequency


class MonopulseInterface(ABC):
    """Abstract base class defining the interface for geometry-related operations."""

    @abstractmethod
    def monopulse(
        self,
        direction_domain: DirectionDomain,
        phase_unit: PhaseUnit,
        amplitude_unit: AmplitudeUnit,
        figure_type: FigureType,
        frequency: Frequency,
        condition: pl.Expr,
        steer: tuple[Angle, Angle] | None = None,
    ) -> None:
        """Process or retrieve geometry data.

        This method must be implemented by subclasses to define specific
        geometry behaviors.
        """
        pass


class Monopulse:
    """A visualization utility class for rendering Monopulse radar beam patterns.

    Coordinates subplots for Sum, Difference, and Ratio patterns across 2D heatmaps,
    3D surfaces, and 2D linear slices.

    Attributes:
        FIGURE_WIDTH (int): Default width of the generated Plotly figure in pixels.
        FIGURE_HEIGHT (int): Default height of the generated Plotly figure in pixels.
        DISPLAY_GRID_PIXELS (int): Resolution of the grid (number of bins) used for
            heatmaps and surface downsampling.
        MIN_DB_CLAMP (float): Lower threshold limit used to clamp deep null values.
    """

    FIGURE_WIDTH = 1400
    FIGURE_HEIGHT = 1200
    DISPLAY_GRID_PIXELS = 100
    MIN_DB_CLAMP = -75.0

    @classmethod
    def _plot_monopulse(
        cls,
        df: pl.DataFrame,
        direction_domain: DirectionDomain,
        phase_unit: PhaseUnit,
        amplitude_unit: AmplitudeUnit,
        figure_type: FigureType,
    ) -> None:
        """Routes the visualization request to the appropriate plotting method.

        Args:
            df (pl.DataFrame): The source DataFrame containing the radar data.
            direction_domain (DirectionDomain): Spatial tracking domain context.
            phase_unit (PhaseUnit): Angular/phase coordinate unit tracking context.
            amplitude_domain (AmplitudeDomain): Amplitude classification context.
            amplitude_unit (AmplitudeUnit): Linear vs logarithmic scale tracking.
            figure_type (FigureType): Determines whether to dispatch to a
                2D line slice, 2D heatmap image, or a 3D surface plot.
        """
        if figure_type == FigureType.IMAGE:
            cls._monopulse_image(df, direction_domain, phase_unit, amplitude_unit)
        elif figure_type == FigureType.SURFACE:
            cls._monopulse_surface(df, direction_domain, phase_unit, amplitude_unit)
        elif figure_type == FigureType.SLICE:
            cls._monopulse_slice(df, direction_domain, phase_unit, amplitude_unit)

    @staticmethod
    def _xy_units(
        direction_domain: DirectionDomain, phase_unit: PhaseUnit
    ) -> tuple[str, str]:
        """Extracts text label units for display on visual axes."""
        az_unit = "u" if direction_domain is DirectionDomain.UV else phase_unit.value
        el_unit = "v" if direction_domain is DirectionDomain.UV else phase_unit.value
        return (az_unit, el_unit)

    @classmethod
    def _monopulse_slice(
        cls,
        df: pl.DataFrame,
        direction_domain: DirectionDomain,
        phase_unit: PhaseUnit,
        amplitude_unit: AmplitudeUnit,
    ) -> None:
        """Renders 2D line plot slices showing Monopulse subplots across Azimuth for distinct Elevations.

        Args:
            df (pl.DataFrame): Input dataset containing spatial and magnitude columns.
            direction_domain (DirectionDomain): Selected directional coordinate paradigm.
            phase_unit (PhaseUnit): Selected angle configuration unit.
            amplitude_domain (AmplitudeDomain): Selected amplitude tracking domain.
            amplitude_unit (AmplitudeUnit): Metric scale unit for the vertical axis.
        """
        az_header, el_header = DataHeader.direction_domain_headers(
            direction_domain, phase_unit
        )
        az_unit, el_unit = cls._xy_units(direction_domain, phase_unit)

        az_label = f"Azimuth [{az_unit}]"
        el_label = f"Elevation [{el_unit}]"
        mag_label = f"Magnitude [{amplitude_unit.value}]"

        # Initialize subplot structure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Monopulse Sum Pattern Slice",
                "Monopulse Difference Pattern Slice",
                "Monopulse Ratio Pattern Slice",
                "Monopulse Sensitivity Slice",
            ),
            horizontal_spacing=0.15,
            vertical_spacing=0.15,
        )

        headers = [
            DataHeader.MONOPULSE_SUMATION_DB,
            DataHeader.MONOPULSE_DIFFERENCE_DB,
            DataHeader.MONOPULSE_RAIO_DB,
            DataHeader.MONOPULSE_SENSITIVITY,
        ]
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        # Unique configurations for color groups to handle cross-subplot legend isolation cleanly
        for header, (row, col) in zip(headers, positions, strict=True):
            # Safe clamping directly inside the plot generation data sequence
            clamped_mag = np.maximum(df[header].to_numpy(), cls.MIN_DB_CLAMP)

            # Temporary express object utilized for fast lineage tracking
            temp_fig = px.line(
                x=df[az_header],
                y=clamped_mag,
                color=df[el_header],
                labels={"color": el_label},
            )

            # Move express traces directly over to our main subplot figure grid matrix
            for trace in temp_fig.data:
                # Group legends so toggling one toggles across all matching elevation subplots
                trace.legendgroup = str(trace.name)  # type: ignore
                if row != 1 or col != 1:
                    trace.showlegend = False  # type: ignore
                fig.add_trace(trace, row=row, col=col)

            fig.update_xaxes(title_text=az_label, row=row, col=col)
            fig.update_yaxes(title_text=mag_label, row=row, col=col)

        fig.update_layout(
            title="Monopulse Pattern Line Slices",
            width=cls.FIGURE_WIDTH,
            height=cls.FIGURE_HEIGHT,
        )
        fig.show()

    @classmethod
    def _monopulse_image(
        cls,
        df: pl.DataFrame,
        direction_domain: DirectionDomain,
        phase_unit: PhaseUnit,
        amplitude_unit: AmplitudeUnit,
    ) -> None:
        """Renders 2D binned density heatmaps for Sum, Difference, Ratio, and Sensitivity patterns."""
        az_header, el_header = DataHeader.direction_domain_headers(
            direction_domain, phase_unit
        )
        az_unit, el_unit = cls._xy_units(direction_domain, phase_unit)

        az_label = f"Azimuth [{az_unit}]"
        el_label = f"Elevation [{el_unit}]"
        mag_label = f"Magnitude [{amplitude_unit.value}]"

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Monopulse Sum Pattern",
                "Monopulse Difference Pattern",
                "Monopulse Ratio Pattern",
                "Monopulse Sensitivity",
            ),
            horizontal_spacing=0.25,
            vertical_spacing=0.12,
        )

        headers = [
            DataHeader.MONOPULSE_SUMATION_DB,
            DataHeader.MONOPULSE_DIFFERENCE_DB,
            DataHeader.MONOPULSE_RAIO_DB,
            DataHeader.MONOPULSE_SENSITIVITY,
        ]
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for header, (row, col) in zip(headers, positions, strict=True):
            x_bins, y_bins, z_vals = cls._bin_dataframe_2d(
                df, az_header, el_header, header
            )
            z_vals = np.maximum(z_vals, cls.MIN_DB_CLAMP)

            heatmap_trace = go.Heatmap(
                x=x_bins,
                y=y_bins,
                z=z_vals,
                colorscale="Viridis",
                zmin=cls.MIN_DB_CLAMP,
                colorbar=dict(
                    title=mag_label,
                    x=1.02 if col == 2 else 0.45,
                    len=0.4,
                    y=0.8 if row == 1 else 0.2,
                ),
            )
            fig.add_trace(heatmap_trace, row=row, col=col)
            fig.update_xaxes(title_text=az_label, row=row, col=col)
            fig.update_yaxes(title_text=el_label, row=row, col=col)

        fig.update_layout(
            title="Monopulse Beam Patterns (2D Density Heatmap)",
            width=cls.FIGURE_WIDTH,
            height=cls.FIGURE_HEIGHT,
        )
        fig.show()

    @classmethod
    def _monopulse_surface(
        cls,
        df: pl.DataFrame,
        direction_domain: DirectionDomain,
        phase_unit: PhaseUnit,
        amplitude_unit: AmplitudeUnit,
    ) -> None:
        """Generates and renders side-by-side 3D surfaces of monopulse components."""
        az_header, el_header = DataHeader.direction_domain_headers(
            direction_domain, phase_unit
        )
        az_unit, el_unit = cls._xy_units(direction_domain, phase_unit)
        mag_label = f"Magnitude [{amplitude_unit.value}]"

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scene"}, {"type": "scene"}],
                [{"type": "scene"}, {"type": "scene"}],
            ],
            subplot_titles=(
                "Monopulse Sum Pattern (3D)",
                "Monopulse Difference Pattern (3D)",
                "Monopulse Ratio Pattern (3D)",
                "Monopulse Sensitivity",
            ),
            horizontal_spacing=0.05,
            vertical_spacing=0.08,
        )

        headers = [
            DataHeader.MONOPULSE_SUMATION_DB,
            DataHeader.MONOPULSE_DIFFERENCE_DB,
            DataHeader.MONOPULSE_RAIO_DB,
            DataHeader.MONOPULSE_SENSITIVITY,
        ]
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for header, (row, col) in zip(headers, positions, strict=True):
            x_bins, y_bins, z_vals = cls._bin_dataframe_2d(
                df, az_header, el_header, header
            )
            z_vals = np.maximum(z_vals, cls.MIN_DB_CLAMP)

            surface_trace = go.Surface(
                x=x_bins,
                y=y_bins,
                z=z_vals,
                colorscale="Viridis",
                coloraxis="coloraxis",
            )
            fig.add_trace(surface_trace, row=row, col=col)

        scene_config = {
            "xaxis_title": f"Azimuth [{az_unit}]",
            "yaxis_title": f"Elevation [{el_unit}]",
            "zaxis_title": mag_label,
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 0.5},
        }

        fig.update_layout(
            title="Monopulse Beam Patterns (3D Surface)",
            coloraxis={
                "colorscale": "Viridis",
                "colorbar": {"title": mag_label},
                "cmin": cls.MIN_DB_CLAMP,
            },
            scene1=scene_config,
            scene2=scene_config,
            scene3=scene_config,
            width=cls.FIGURE_WIDTH,
            height=cls.FIGURE_HEIGHT,
        )
        fig.show()

    @classmethod
    def _bin_dataframe_2d(
        cls, df: pl.DataFrame, x_col: str, y_col: str, z_col: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Utility matrix method to downsample spatial data frames using 2D histogramming."""
        x_arr = df[x_col].to_numpy()
        y_arr = df[y_col].to_numpy()
        z_arr = df[z_col].to_numpy()

        x_bins = np.linspace(x_arr.min(), x_arr.max(), cls.DISPLAY_GRID_PIXELS)
        y_bins = np.linspace(y_arr.min(), y_arr.max(), cls.DISPLAY_GRID_PIXELS)

        counts, _, _ = np.histogram2d(x_arr, y_arr, bins=[x_bins, y_bins])
        sums, _, _ = np.histogram2d(x_arr, y_arr, bins=[x_bins, y_bins], weights=z_arr)

        with np.errstate(divide="ignore", invalid="ignore"):
            z_vals = sums / counts
            z_vals[counts == 0] = np.nan

        return x_bins, y_bins, z_vals.T
