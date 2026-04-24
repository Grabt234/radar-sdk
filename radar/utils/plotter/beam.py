import polars as pl
import numpy as np
import plotly.express as px
from abc import ABC, abstractmethod

import plotly.graph_objects as go
from radar.utils.typing import (
    AmplitudeUnit,
    DataHeader,
    DirectionDomain,
    FigureType,
    AmplitudeDomain,
)
from radar.utils.typing.enums import PhaseUnit
from radar.utils.typing.units import Angle, Frequency


class BeamInterface(ABC):
    """Abstract base class defining the interface for generating antenna beam patterns."""

    @abstractmethod
    def beam(
        self,
        direction_domain: DirectionDomain,
        phase_unit: PhaseUnit,
        amplitude_domain: AmplitudeDomain,
        amplitude_unit: AmplitudeUnit,
        figure_type: FigureType,
        frequency: Frequency,
        steer: tuple[Angle, Angle] | None = None,
    ) -> None:
        """Process and visualize the antenna beam pattern.

        This method must be implemented by subclasses to coordinate dataframe
        processing and layout rendering based on the domain parameters provided.

        Args:
            direction_domain (DirectionDomain): The spatial domain to utilize
                (e.g., Angle or UV space).
            phase_unit (PhaseUnit): The angular unit (e.g., Degrees or Radians).
            amplitude_domain (AmplitudeDomain): The amplitude measurement type
                (e.g., Gain or Antenna Factor).
            amplitude_unit (AmplitudeUnit): The representation unit for
                amplitude (e.g., Linear or Decibel).
            figure_type (FigureType): The targeted plot visualization type
                (e.g., Image, Surface, Slice).
            steer (tuple[Angle, Angle] | None): Optional azimuth/elevation steer
                angle used by array-based beam generation.
        """
        pass


class Beam:
    """A visualization utility class for generating various antenna beam pattern plots.

    Attributes:
        FIGURE_WIDTH (int): Default width of the generated Plotly figure in pixels.
        FIGURE_HEIGHT (int): Default height of the generated Plotly figure in pixels.
        DISPLAY_GRID_PIXELS (int): Resolution of the grid (number of bins) used for
            heatmaps and surface downsampling.
    """

    FIGURE_WIDTH = 800
    FIGURE_HEIGHT = 800
    DISPLAY_GRID_PIXELS = 100

    @classmethod
    def _plot_beam(
        cls,
        df: pl.DataFrame,
        direction_domain: DirectionDomain,
        phase_unit: PhaseUnit,
        amplitude_domain: AmplitudeDomain,
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
            cls._beam_image(
                df, direction_domain, phase_unit, amplitude_domain, amplitude_unit
            )
        elif figure_type == FigureType.SURFACE:
            cls._beam_surface(
                df, direction_domain, phase_unit, amplitude_domain, amplitude_unit
            )
        elif figure_type == FigureType.SLICE:
            cls._beam_slice(
                df, direction_domain, phase_unit, amplitude_domain, amplitude_unit
            )

    @staticmethod
    def _xy_units(
        direction_domain: DirectionDomain, phase_unit: PhaseUnit
    ) -> tuple[str, str]:
        """Extracts text label units for display on visual axes.

        Args:
            direction_domain (DirectionDomain): Targeted plotting spatial coordinate system.
            phase_unit (PhaseUnit): Configured orientation angle unit.

        Returns:
            tuple[str, str]: Axis string labels representing horizontal and vertical dimensions.
        """
        az_unit = "u" if direction_domain is DirectionDomain.UV else phase_unit.value
        el_unit = "v" if direction_domain is DirectionDomain.UV else phase_unit.value
        return (az_unit, el_unit)

    @classmethod
    def _beam_slice(
        cls,
        df: pl.DataFrame,
        direction_domain: DirectionDomain,
        phase_unit: PhaseUnit,
        amplitude_domain: AmplitudeDomain,
        amplitude_unit: AmplitudeUnit,
    ) -> None:
        """Renders a 2D line plot showing pattern slices across Azimuth for distinct Elevations.

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
        mag_header = DataHeader._amplitude_domain_headers(
            amplitude_domain, amplitude_unit
        )
        az_unit, el_unit = cls._xy_units(direction_domain, phase_unit)

        fig = px.line(
            x=df[az_header],
            y=df[mag_header],
            color=df[el_header],
            labels={
                "x": f"Azimuth [{az_unit}]",
                "y": f"gain [{amplitude_unit.value}]",
                "color": f"Elevation [{el_unit}]",
            },
            width=cls.FIGURE_WIDTH,
            height=cls.FIGURE_HEIGHT,
        )

        fig.update_layout(title="Element Antenna Beam Pattern Slices")
        fig.show()

    @classmethod
    def _beam_image(
        cls,
        df: pl.DataFrame,
        direction_domain: DirectionDomain,
        phase_unit: PhaseUnit,
        amplitude_domain: AmplitudeDomain,
        amplitude_unit: AmplitudeUnit,
    ) -> None:
        """Renders a 2D binned density heatmap representing the antenna beam pattern.

        Args:
            df (pl.DataFrame): Input dataset containing spatial and magnitude columns.
            direction_domain (DirectionDomain): Selected directional coordinate paradigm.
            phase_unit (PhaseUnit): Selected angle configuration unit.
            amplitude_domain (AmplitudeDomain): Selected amplitude tracking domain.
            amplitude_unit (AmplitudeUnit): Metric scale unit for color mapping values.
        """
        az_header, el_header = DataHeader.direction_domain_headers(
            direction_domain, phase_unit
        )
        mag_header, mag_type = (
            DataHeader._amplitude_domain_headers(amplitude_domain, amplitude_unit),
            amplitude_unit.value,
        )

        az_unit, el_unit = cls._xy_units(direction_domain, phase_unit)

        fig = px.density_heatmap(
            df,
            x=az_header,
            y=el_header,
            z=mag_header,
            histfunc="avg",
            nbinsx=cls.DISPLAY_GRID_PIXELS,
            nbinsy=cls.DISPLAY_GRID_PIXELS,
            color_continuous_scale="Viridis",
            labels={
                az_header: f"Azimuth [{az_unit}]",
                el_header: f"Elevation [{el_unit}]",
                mag_header: f"gain [{mag_type}]",
            },
            width=cls.FIGURE_WIDTH,
            height=cls.FIGURE_HEIGHT,
        )

        fig.update_xaxes(range=[min(df[az_header]), max(df[az_header])])
        fig.update_yaxes(
            range=[
                min(df[el_header]),
                max(df[el_header]),
            ]
        )

        fig.update_layout(title="Antenna Pattern Heatmap")
        fig.update_coloraxes(colorbar_title=mag_type)
        fig.show()

    @classmethod
    def _beam_surface(
        cls,
        df: pl.DataFrame,
        direction_domain: DirectionDomain,
        phase_unit: PhaseUnit,
        amplitude_domain: AmplitudeDomain,
        amplitude_unit: AmplitudeUnit,
    ) -> None:
        """Generates and renders a 3D surface mesh using 2D binned averaging downsampling.

        Args:
            df (pl.DataFrame): Input dataset containing spatial and magnitude columns.
            direction_domain (DirectionDomain): Selected directional coordinate paradigm.
            phase_unit (PhaseUnit): Selected angle configuration unit.
            amplitude_domain (AmplitudeDomain): Selected amplitude tracking domain.
            amplitude_unit (AmplitudeUnit): Metric scale unit for vertical 3D displacement.
        """
        az_header, el_header = DataHeader.direction_domain_headers(
            direction_domain, phase_unit
        )
        mag_header, mag_type = (
            DataHeader._amplitude_domain_headers(amplitude_domain, amplitude_unit),
            amplitude_unit.value,
        )

        x_min, x_max = min(df[az_header]), max(df[az_header])
        y_min, y_max = min(df[el_header]), max(df[el_header])

        u_bins = np.linspace(x_min, x_max, cls.DISPLAY_GRID_PIXELS)
        v_bins = np.linspace(y_min, y_max, cls.DISPLAY_GRID_PIXELS)
        counts, _, _ = np.histogram2d(
            df[az_header],
            df[el_header],
            bins=[u_bins, v_bins],
        )
        sums, _, _ = np.histogram2d(
            df[az_header],
            df[el_header],
            bins=[u_bins, v_bins],
            weights=df[mag_header],
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            z_vals = sums / counts
            z_vals[counts == 0] = np.nan

        fig = go.Figure(
            data=[
                go.Surface(
                    x=v_bins,
                    y=u_bins,
                    z=z_vals,
                    colorscale="Viridis",
                    colorbar={"title": mag_type},
                    connectgaps=True,
                )
            ]
        )

        az_unit, el_unit = cls._xy_units(direction_domain, phase_unit)

        fig.update_layout(
            title=f"Array Beam Pattern {direction_domain.value}",
            scene={
                "xaxis_title": f"Azimuth [{az_unit}]",
                "yaxis_title": f"Elevation [{el_unit}]",
                "zaxis_title": mag_type,
                "aspectmode": "manual",
                "aspectratio": {"x": 1, "y": 1, "z": 0.5},
            },
            width=cls.FIGURE_WIDTH,
            height=cls.FIGURE_HEIGHT,
        )
        fig.show()
