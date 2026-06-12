from manim import ManimColor, Surface, VGroup

from radar.utils.calculate.monopulse import Monopulse

from .element import Element
from .geometry import Geometry
from radar.utils.typing import (
    PhaseUnit,
    DataHeader,
    RadarConstants,
    AmplitudeDomain,
    DirectionDomain,
    FigureType,
    Frequency,
    Angle,
    AmplitudeUnit,
)

from radar.utils.calculate.convert import to_db
import numpy as np
import numpy.typing as npt
import polars as pl
from radar.utils import plotter, animate


class Array:
    def __init__(
        self,
        element: Element,
        geometry: Geometry,
    ):
        self._element = element
        self._geometry = geometry
        self._monopulse = Monopulse()

        self.plot = self.Plot(self)
        self.animate = self.Animate(self)

    @property
    def element(self):
        return self._element

    def beam_pattern(
        self, frequency: Frequency, steer: tuple[Angle, Angle] | None = None
    ) -> pl.DataFrame:
        return self._beam_pattern(steer, frequency)

    def _beam_pattern(
        self, steer: tuple[Angle, Angle] | None, frequency: Frequency
    ) -> pl.DataFrame:
        return self._calculate_beam_pattern(self._element, frequency, steer)

    def _calculate_beam_pattern(
        self, element: Element, frequency: Frequency, steer: tuple[Angle, Angle] | None
    ):
        # 1. Compute the Array Factor DataFrame
        af_df = self.calculate_array_factor(frequency, steer)
        element_pattern = element.beam_pattern(frequency)

        # 1. Pull the element gain out as a literal or expression
        # If element_pattern is a DataFrame, use pl.lit() to broadcast it,
        # or join/append if they are aligned. Assuming it's a scalar or pre-aligned:
        element_gain_expr = pl.lit(element_pattern[DataHeader.BEAM_GAIN_LINEAR])

        # 2. Define the total gain as an Expression instead of a Series
        total_gain_expr = element_gain_expr * pl.col(DataHeader.ANTENNA_FACTOR_LINEAR)

        # 3. safely pass the Expression to to_db()
        return af_df.with_columns(
            [
                total_gain_expr.alias(DataHeader.BEAM_GAIN_LINEAR),
                to_db(total_gain_expr).alias(DataHeader.BEAM_GAIN_DB),
            ]
        )

    def _calculate_array_factor(
        self,
        frequency: Frequency,
        steer: tuple[Angle, Angle] | None,
    ) -> pl.DataFrame:
        k = (2 * np.pi) * frequency.Hz / RadarConstants.c

        pos_x = self._geometry.geometry[DataHeader.X_POS_M].to_numpy()
        pos_y = self._geometry.geometry[DataHeader.Y_POS_M].to_numpy()
        num_elements = pos_x.size

        # --- Extract Element Amplitudes and Phases ---
        # Fallback to uniform weights (1.0) and no phase shift if columns aren't present
        amp = self._geometry.geometry[DataHeader.GEOM_AMP_GAIN_LIN].to_numpy()
        elem_phase = self._geometry.geometry[
            DataHeader.GEOM_PHASE_SHIFTER_PHASE_RAD
        ].to_numpy()

        # Combine amplitude and element-specific phase into a complex weight vector
        # Shape: (N_elements,)
        element_weights = amp * np.exp(1j * elem_phase)

        el_dom_rad = self.element.elevation_domain(PhaseUnit.RADIAN)
        az_dom_rad = self.element.azimuth_domain(PhaseUnit.RADIAN)

        u = np.sin(az_dom_rad)
        v = np.sin(el_dom_rad)

        u_flat = u.ravel()
        v_flat = v.ravel()
        visible_mask = u_flat**2 + v_flat**2 <= 1

        steer = steer or (Angle(0.0, PhaseUnit.DEGREE), Angle(0.0, PhaseUnit.DEGREE))
        az_steer, el_steer = steer[0].rad, steer[1].rad
        u0 = np.sin(az_steer)
        v0 = np.sin(el_steer)

        # --- Accumulate array factor (Vectorized) ---
        delta_u = u_flat - u0
        delta_v = v_flat - v0

        # Spatial propagation phases
        # Shape: (N_elements, M_angles)
        spatial_phases = k * (
            pos_x[:, np.newaxis] * delta_u + pos_y[:, np.newaxis] * delta_v
        )

        # Total complex signal per element: element_weights * e^(j * spatial_phases)
        # Using broadcasting: (N_elements, 1) * (N_elements, M_angles)
        complex_signals = element_weights[:, np.newaxis] * np.exp(1j * spatial_phases)

        # Complex sum across the element axis (axis 0)
        # Normalized by the sum of amplitudes to keep peak linear gain at 1.0 (or divided by num_elements)
        norm_factor = np.sum(amp) if np.sum(amp) > 0 else num_elements
        af = np.sum(complex_signals, axis=0) / norm_factor

        af_mag = np.abs(af)
        af_mag = np.maximum(af_mag, 1e-15)

        # --- Output DataFrame ---
        result_data = {
            DataHeader.AZIMUTH_RAD: az_dom_rad.get_column(DataHeader.AZIMUTH_RAD),
            DataHeader.AZIMUTH_DEG: np.rad2deg(
                az_dom_rad.get_column(DataHeader.AZIMUTH_RAD).to_numpy()
            ),
            DataHeader.ELEVATION_RAD: el_dom_rad.get_column(DataHeader.ELEVATION_RAD),
            DataHeader.ELEVATION_DEG: np.rad2deg(
                el_dom_rad.get_column(DataHeader.ELEVATION_RAD).to_numpy()
            ),
            DataHeader.U: u_flat,
            DataHeader.V: v_flat,
            DataHeader.UV_MASK: visible_mask,
            DataHeader.ANTENNA_FACTOR_DB: 20 * np.log10(af_mag),
            DataHeader.ANTENNA_FACTOR_LINEAR: af_mag,
        }

        return pl.DataFrame(result_data)

    def calculate_array_factor(
        self,
        frequency: Frequency,
        steer: tuple[Angle, Angle] | None,
    ) -> pl.DataFrame:
        return self._calculate_array_factor(frequency, steer)

    def monopulse(
        self,
        frequency: Frequency,
        expr: pl.Expr,
        steer: tuple[Angle, Angle] | None = None,
    ):
        beam_pattern = self.beam_pattern(frequency, steer)
        geometry = self._geometry.geometry
        return self._monopulse.calculate_monopulse(
            frequency, beam_pattern, geometry, expr
        )

    class Plot(
        plotter.BeamInterface, plotter.GeometryInterface, plotter.MonopulseInterface
    ):
        def __init__(self, outer: "Array"):
            self._outer = outer

        def beam(
            self,
            direction_domain: DirectionDomain,
            phase_unit: PhaseUnit,
            amplitude_domain: AmplitudeDomain,
            amplitude_unit: AmplitudeUnit,
            figure_type: FigureType,
            frequency: Frequency,
            steer: tuple[Angle, Angle] | None = None,
        ):
            df = self._outer.beam_pattern(frequency, steer)
            plotter.Beam._plot_beam(
                df,
                direction_domain,
                phase_unit,
                amplitude_domain,
                amplitude_unit,
                figure_type,
            )

        def geometry(self):
            self._outer._geometry.plot.geometry()

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
            """Processes, calculates, and routes monopulse tracking data for visualization."""
            beam_pattern = self._outer.beam_pattern(frequency, steer)
            geometry_df = self._outer._geometry.geometry

            # Calculate the underlying monopulse channel data
            df = Monopulse.calculate_monopulse(
                frequency, beam_pattern, geometry_df, condition
            )

            # Route to the newly standardized Monopulse visual dispatcher
            plotter.Monopulse._plot_monopulse(
                df=df,
                direction_domain=direction_domain,
                phase_unit=phase_unit,
                amplitude_unit=amplitude_unit,
                figure_type=figure_type,
            )

    class Animate(animate.BeamInterface, animate.GeometryInterface):
        def __init__(self, outer: "Array") -> None:
            """Initializes the animate handler bound to an Element context.

            Args:
                outer (Element): Parent instance providing the underlying beam pattern records.
            """
            self._outer = outer
            self._animate_beam = animate.Beam
            self._animate_geometry = animate.Geometry

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

            return self._animate_beam.surface_3d(
                self._outer.beam_pattern(frequency, steer),
                position,
                direction_domain,
                phase_unit,
                amplitude_domain,
                amplitude_unit,
            )

        def geometry(self, position: npt.NDArray, colour: ManimColor) -> VGroup:
            """Dispatches coordinate snapshots to render an image of the antenna layout."""
            return self._animate_geometry.dots_3d(
                self._outer._geometry.geometry, position, colour
            )
