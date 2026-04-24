from manim import ManimColor, Surface, VGroup

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

        af_df = self.calculate_array_factor(frequency, steer)

        # 2. Get the element's individual pattern
        pattern = element.beam_pattern(frequency)

        # 3. Extract numpy arrays for math
        af_lin = af_df[DataHeader.ANTENNA_FACTOR_LINEAR].to_numpy().astype(np.float64)
        gain_lin = pattern[DataHeader.BEAM_GAIN_LINEAR].to_numpy().astype(np.float64)

        # Total Array Gain = Element Gain * Array Factor
        total_gain_lin = gain_lin * af_lin

        # 4. Return the af_df with the NEW gain columns added/updated
        return af_df.with_columns(
            [
                pl.Series(DataHeader.BEAM_GAIN_LINEAR, total_gain_lin),
                pl.Series(DataHeader.BEAM_GAIN_DB, to_db(total_gain_lin)),
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
        # (u - u0) and (v - v0) represent the shift in the UV plane
        delta_u = u_flat - u0
        delta_v = v_flat - v0

        # Broadcasting: (N_elements, 1) * (1, M_angles)
        phases = k * (pos_x[:, np.newaxis] * delta_u + pos_y[:, np.newaxis] * delta_v)

        # Complex sum across the element axis (axis 0)
        af = np.sum(np.exp(1j * phases), axis=0) / num_elements
        af_mag = np.abs(af)
        af_mag = np.maximum(af_mag, 1e-15)

        # --- Output DataFrame ---
        result_data = {
            # Extract as a Series and convert to numpy for the calculation
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

    class Plot(plotter.BeamInterface, plotter.GeometryInterface):
        def __init__(self, outer: "Array"):
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
        ):
            df = self._outer.beam_pattern(frequency, steer)
            self.plot._plot_beam(
                df,
                direction_domain,
                phase_unit,
                amplitude_domain,
                amplitude_unit,
                figure_type,
            )

        def geometry(self):
            self._outer._geometry.plot.geometry()

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
