"""
Radar Beam Visualizer Module.

Provides abstractions and implementation classes to transform numerical beam
pattern datasets (gain, azimuth, elevation) into interpolatable 3D Manim
Surface objects.
"""

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs
from manim import BLUE, BLUE_E, RED, YELLOW, Surface, ThreeDAxes
from scipy.interpolate import RegularGridInterpolator, griddata

from radar.utils.typing.constants import DataHeader
from radar.utils.typing.enums import (
    AmplitudeDomain,
    AmplitudeUnit,
    DirectionDomain,
    PhaseUnit,
)
from radar.utils.typing.units import Frequency
from radar.utils.typing.units import Angle


class BeamInterface(ABC):
    """
    Abstract Base Class acting as an interface for all radar array beam generators.

    Ensures unified functional signatures across continuous or discrete array response forms.
    """

    @abstractmethod
    def beam(
        self,
        frequency: Frequency,
        position: npt.NDArray,
        direction_domain: DirectionDomain,
        phase_unit: PhaseUnit,
        amplitude_domain: AmplitudeDomain,
        amplitude_unit: AmplitudeUnit,
        steer: tuple[Angle, Angle] | None = None,
    ) -> Surface:
        """
        Generate a Manim Surface object representing the 3D radiation beam pattern.

        Args:
            frequency (Frequency): The operating tracking frequency instance.
            position (npt.NDArray): The 3D anchor coordinate [x, y, z] for the pattern center.
            direction_domain (DirectionDomain): Coordinate space domain (e.g., Angle, UV).
            phase_unit (PhaseUnit): Angular scale (e.g., Radian, Degree).
            amplitude_domain (AmplitudeDomain): Pattern scaling type (e.g., Gain, Directivity).
            amplitude_unit (AmplitudeUnit): Amplitude metrics type (e.g., Decibel, Linear).
            steer (tuple[Angle, Angle] | None): Optional azimuth/elevation steer
                angle used by array-based beam generation.

        Returns:
            Surface: A Manim 3D continuous surface wrapper.
        """
        pass


class Beam:
    """
    Utility handler for evaluating, interpolating, and mapping tabular beam logs
    into visual 3D Manim objects.
    """

    X_RESOLUTION: int = 100
    Y_RESOLUTION: int = 100

    @classmethod
    def surface_3d(
        cls,
        df: pl.DataFrame,
        position: npt.NDArray,
        direction_domain: DirectionDomain,
        phase_unit: PhaseUnit,
        amplitude_domain: AmplitudeDomain,
        amplitude_unit: AmplitudeUnit,
    ) -> Surface:
        """
        Construct a continuous, interpolated 3D Manim Surface based on spatial dataframe patterns.

        Args:
            df (pl.DataFrame): Tabular beam log dataset.
            position (npt.NDArray): 3D translation anchor coordinate.
            direction_domain (DirectionDomain): Spatial coordinate configuration mode.
            phase_unit (PhaseUnit): Angular metric framework units.
            amplitude_domain (AmplitudeDomain): Signal target scale format.
            amplitude_unit (AmplitudeUnit): Magnitude representation bounds.

        Returns:
            Surface: Renderable 3D Manim surface mesh.
        """
        return cls._df_to_surface(
            df, position, direction_domain, phase_unit, amplitude_domain, amplitude_unit
        )

    @classmethod
    def _df_to_surface(
        cls,
        df: pl.DataFrame,
        position: npt.NDArray,
        direction_domain: DirectionDomain,
        phase_unit: PhaseUnit,
        amplitude_domain: AmplitudeDomain,
        amplitude_unit: AmplitudeUnit,
    ) -> Surface:
        """
        Internal math processing kernel to handle scaffolding, sorting, grid interpolation,
        and surface color mapping.
        """
        # Resolve data matching properties using dynamic layout headers
        az_header, el_header = DataHeader.direction_domain_headers(
            direction_domain, phase_unit
        )
        mag_header, _ = (
            DataHeader._amplitude_domain_headers(amplitude_domain, amplitude_unit),
            amplitude_unit.value,
        )

        axes_config = {
            "x_range": [-1, 1, 0.2],
            "y_range": [-1, 1, 0.2],
            "z_range": [-1, 1, 0.2],
            "x_length": 5,
            "y_length": 5,
            "z_length": 5,
        }
        axes = ThreeDAxes(**axes_config).shift(position)

        # Normalize and arrange spatial logs cleanly
        df = cls.normalise(df)
        sorted_df = df.sort([az_header, el_header])

        az_vals = sorted_df[az_header].to_numpy()
        el_vals = sorted_df[el_header].to_numpy()
        gain_vals = sorted_df[mag_header].to_numpy()

        # Construct interpolation matrix grid
        num_u, num_v = cls.X_RESOLUTION, cls.Y_RESOLUTION
        u_vector = np.linspace(az_vals.min(), az_vals.max(), num_u)
        v_vector = np.linspace(el_vals.min(), el_vals.max(), num_v)
        X, Y = np.meshgrid(u_vector, v_vector)

        # Resolve edge NaN gaps via hybrid fallback calculation
        Z_linear = griddata((az_vals, el_vals), gain_vals, (X, Y), method="linear")
        Z_nearest = griddata((az_vals, el_vals), gain_vals, (X, Y), method="nearest")
        Z = np.where(np.isnan(Z_linear), Z_nearest, Z_linear)

        # Setup standard boundary axis mapping models
        interp_X = RegularGridInterpolator(
            (v_vector, u_vector), X, method="linear", bounds_error=False
        )
        interp_Y = RegularGridInterpolator(
            (v_vector, u_vector), Y, method="linear", bounds_error=False
        )
        interp_Z = RegularGridInterpolator(
            (v_vector, u_vector), Z, method="linear", bounds_error=False
        )

        def mesh_func(u: float, v: float) -> npt.NDArray:
            """Map parametric u,v coordinates to physical 3D scene space points."""
            x_val = interp_X((v, u))
            y_val = interp_Y((v, u))
            z_val = interp_Z((v, u))
            return axes.coords_to_point(x_val, y_val, z_val)

        # Generate structural mesh surface representation
        surface = Surface(
            mesh_func,
            u_range=(az_vals.min(), az_vals.max()),
            v_range=(el_vals.min(), el_vals.max()),
            resolution=(cls.X_RESOLUTION, cls.Y_RESOLUTION),
        )

        # Apply visual styling variables
        surface.set_style(
            fill_opacity=1, stroke_width=0.3, stroke_color=BLUE_E, stroke_opacity=0.4
        )
        surface.set_fill_by_value(
            axes=axes,
            colorscale=[(BLUE, 0), (RED, 0.5), (YELLOW, 1)],
            axis=2,  # Map variance dynamically along Z-axis variations
        )

        return surface

    @staticmethod
    def normalise(data: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize all numeric dataframe metric tracking series to a range between [-1, 1].

        Args:
            data (pl.DataFrame): Target source dataframe container.

        Returns:
            pl.DataFrame: Scaled and re-mapped dataset.

        Raises:
            TypeError: If input data is not a Polars DataFrame.
        """
        if isinstance(data, pl.DataFrame):
            return data.with_columns(
                (
                    2
                    * (pl.col(col) - pl.col(col).min())
                    / (pl.col(col).max() - pl.col(col).min())
                    - 1
                ).alias(col)
                for col in data.select(cs.numeric()).columns
            )

        raise TypeError(f"Unsupported input type profile: {type(data)}")
