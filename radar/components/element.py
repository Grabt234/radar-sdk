from manim import Surface

from radar.utils.calculate.pattern import Pattern

from radar.utils.typing import (
    PhaseUnit,
    DataHeader,
    AngleBound,
    DirectionDomain,
    FigureType,
    AmplitudeDomain,
    AmplitudeUnit,
)

import numpy as np
import polars as pl
import logging

import warnings
from radar.utils import plotter, animate
import numpy.typing as npt

from radar.utils.typing.units import Angle, Frequency

from .response import FrequencyResponse

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")


class Element:
    """Represents an individual radar antenna element with its radiation beam pattern.

    Handles spatial mesh grid configuration across explicit angular bounds and
    manages the evaluation of underlying analytical or data-driven pattern models.
    """

    def __init__(
        self,
        pattern: Pattern,
        az: AngleBound,
        el: AngleBound,
        frequency_response: FrequencyResponse | Frequency,
        over_sample_factor: int = 1,
    ) -> None:
        """Initializes the radar antenna element.

        Args:
            pattern (Pattern): An implementation of the Pattern interface used to
                compute spatial gain distribution.
            az (AngleBound): The validation bounds specifying minimum and maximum Azimuth.
            el (AngleBound): The validation bounds specifying minimum and maximum Elevation.
            over_sample_factor (int, optional): Multiplier to increase the resolution
                density of the calculated grid surface. Defaults to 1.
        """
        self._over_sample_factor = over_sample_factor

        self.plot = self.Plot(self)
        self.animate = self.Animate(self)

        self._az_bound = az
        self._el_bound = el

        self._pattern = pattern
        self._response = (
            FrequencyResponse(frequency_response)
            if isinstance(frequency_response, Frequency)
            else frequency_response
        )

        self._calculated_pattern = self._pattern.calculate_pattern(
            self._generate_beam_grid(az, el)
        )

    def beam_pattern(self, freq: Frequency) -> pl.DataFrame:
        """Returns the complete evaluated beam pattern dataset including coordinates and gains."""

        pattern = self._calculated_pattern
        resp_db = self._response.response(freq, AmplitudeUnit.DECIBEL)
        resp_lin = self._response.response(freq, AmplitudeUnit.LINEAR)

        pattern = pattern.with_columns(
            (pl.col(DataHeader.BEAM_GAIN_DB) + resp_db),
            (pl.col(DataHeader.BEAM_GAIN_LINEAR) * resp_lin),
        )

        return pattern

    @property
    def azimuth_bound(self) -> AngleBound:
        """Returns the configured spatial Azimuth boundaries for this element."""
        return self._az_bound

    @property
    def elevation_bound(self) -> AngleBound:
        """Returns the configured spatial Elevation boundaries for this element."""
        return self._el_bound

    def azimuth_domain(self, domain: PhaseUnit) -> pl.DataFrame:
        """Selects the Azimuth coordinates from the dataset in the specified unit.

        Args:
            domain (PhaseUnit): The desired angular format (e.g., Degree or Radian).

        Returns:
            pl.DataFrame: A single-column DataFrame containing Azimuth coordinate values.
        """
        return self._calculated_pattern.select(
            DataHeader.AZIMUTH_RAD
            if domain is PhaseUnit.RADIAN
            else DataHeader.AZIMUTH_DEG
        )

    def elevation_domain(self, domain: PhaseUnit) -> pl.DataFrame:
        """Selects the Elevation coordinates from the dataset in the specified unit.

        Args:
            domain (PhaseUnit): The desired angular format (e.g., Degree or Radian).

        Returns:
            pl.DataFrame: A single-column DataFrame containing Elevation coordinate values.
        """
        return self._calculated_pattern.select(
            DataHeader.ELEVATION_RAD
            if domain is PhaseUnit.RADIAN
            else DataHeader.ELEVATION_DEG
        )

    def _generate_beam_grid(
        self,
        az: AngleBound,
        el: AngleBound,
    ) -> pl.DataFrame:
        """Generates a 2D spatial mesh grid spanning across the angular limits.

        Transforms coordinate intervals into structured mesh domains, computes
        sine-space $u$ and $v$ projections, and applies a unit-circle safety
        filter mask to track physical boundaries.

        Args:
            az (AngleBound): Structural Azimuth constraint boundaries.
            el (AngleBound): Structural Elevation constraint boundaries.

        Returns:
            pl.DataFrame: Flattened coordinate map dataset containing angular,
            directional sine space columns ($u, v$), and the active region validation mask.
        """
        # ensure we include bounds and that we do not align exactly
        # on zero as this causes issues in uv projection
        num_az = (int(abs(az[1].deg - az[0].deg)) + 1) * self._over_sample_factor + 1
        num_el = (int(abs(el[1].deg - el[0].deg)) + 1) * self._over_sample_factor + 1

        logging.debug(
            f"az min {az[0].deg}, az maz {az[1].deg}, el min {el[0].deg}, el max {el[1].deg}"
        )

        az_values = np.linspace(az[0].rad, az[1].rad, num_az)
        el_values = np.linspace(el[0].rad, el[1].rad, num_el)

        az_grid, el_grid = np.meshgrid(az_values, el_values)

        # Calculate U and V across the entire 2D surface
        u_grid = np.sin(az_grid)
        v_grid = np.sin(el_grid)

        logging.debug(
            f"u min {np.min(u_grid)}, u maz {np.max(u_grid)}, v min {np.min(v_grid)}, v max {np.max(v_grid)}"
        )

        # 1. Calculate the radial distance squared for efficiency
        # (u^2 + v^2 <= 1^2)
        radial_dist_sq = u_grid**2 + v_grid**2
        mask = radial_dist_sq <= 1.0

        # 2. Apply the mask to all arrays and flatten them
        return pl.DataFrame(
            {
                DataHeader.AZIMUTH_RAD: az_grid.ravel(),
                DataHeader.ELEVATION_RAD: el_grid.ravel(),
                DataHeader.AZIMUTH_DEG: np.rad2deg(az_grid.ravel()),
                DataHeader.ELEVATION_DEG: np.rad2deg(el_grid.ravel()),
                DataHeader.U: u_grid.ravel(),
                DataHeader.V: v_grid.ravel(),
                DataHeader.UV_MASK: mask.ravel(),
            }
        )

    class Plot(plotter.BeamInterface):
        """Inner bridge class handling plotting commands for its parent Element."""

        def __init__(self, outer: "Element") -> None:
            """Initializes the plotting handler bound to an Element context.

            Args:
                outer (Element): Parent instance providing the underlying beam pattern records.
            """
            self._outer = outer
            self.plot = plotter.Beam

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
            """Prepares data parameters and dispatches requests to the plotter renderer.

            Automatically intercepts and falls back to a standard Gain representation
            if Antenna Factor processing is requested, as Antenna Factor parameters
            are undefined for independent standalone element models.

            Args:
                direction_domain (DirectionDomain): The spatial domain to utilize (e.g., Angle or UV).
                phase_unit (PhaseUnit): The angular display unit (e.g., Degrees or Radians).
                amplitude_domain (AmplitudeDomain): Measurement framework type to track.
                amplitude_unit (AmplitudeUnit): Linear vs logarithmic scale configuration context.
                figure_type (FigureType): Targeted plot layout style (e.g., Image, Surface, Slice).
                steer (tuple[Angle, Angle] | None): Ignored for single element beams.
            """
            if amplitude_domain is AmplitudeDomain.AntennaFactor:
                warnings.warn(
                    "Antenna factor is undefine for a single element, reverting to gain pattern"
                )
                amplitude_domain = AmplitudeDomain.Gain
            if steer is not None:
                warnings.warn(
                    "Steering is undefined for a single element, ignoring steer"
                )

            self.plot._plot_beam(
                self._outer.beam_pattern(frequency),
                direction_domain,
                phase_unit,
                amplitude_domain,
                amplitude_unit,
                figure_type,
            )

    class Animate(animate.BeamInterface):
        def __init__(self, outer: "Element") -> None:
            """Initializes the animate handler bound to an Element context.

            Args:
                outer (Element): Parent instance providing the underlying beam pattern records.
            """
            self._outer = outer
            self.animate = animate.Beam

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
            if steer is not None:
                warnings.warn(
                    "Steering is undefined for a single element, ignoring steer"
                )
            return self.animate.surface_3d(
                self._outer.beam_pattern(frequency),
                position,
                direction_domain,
                phase_unit,
                amplitude_domain,
                amplitude_unit,
            )
