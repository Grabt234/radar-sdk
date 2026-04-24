from manim import (
    BLUE,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    DEGREES,
    Text,
    Arrow,
    Line,
    ThreeDScene,
    Create,
    ReplacementTransform,
    Write,
)
from radar.components import geometry, Element
from radar.components.array import Array
from radar.utils.calculate import convert, pattern
from radar.utils.typing import Angle, PhaseUnit, FrequencyUnit
from radar.utils.typing.enums import AmplitudeDomain, AmplitudeUnit, DirectionDomain
from radar.utils.typing.units import Frequency

# --- Configuration Constants ---
FREQ = Frequency(1, FrequencyUnit.GIGAHERTZ)
DIST = convert.cf_to_min_dist(FREQ)

BW_TUPLE = (Angle(15, PhaseUnit.DEGREE), Angle(15, PhaseUnit.DEGREE))
BOUNDS = (Angle(-60, PhaseUnit.DEGREE), Angle(60, PhaseUnit.DEGREE))
ANT_ELEMENT = Element(pattern.Isotropic(), BOUNDS, BOUNDS, FREQ, 1)

# Steering path: (azimuth_deg, elevation_deg)
STEER_STEPS = [
    (0.0, 0.0),
    (0.0, 20.0),
    (20.0, 20.0),
    (20.0, 0.0),
    (0.0, 0.0),
]

STEER_ORIGIN = LEFT * 4.5 + DOWN * 2
STEER_MAX_DEG = 20.0
STEER_SCALE = 0.2
RIGHT_POS = RIGHT * 3.5

BEAM_KWARGS = {
    "position": RIGHT_POS,
    "direction_domain": DirectionDomain.ANGLE,
    "phase_unit": PhaseUnit.DEGREE,
    "amplitude_domain": AmplitudeDomain.Gain,
    "amplitude_unit": AmplitudeUnit.DECIBEL,
}


class Video(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=30 * DEGREES, theta=-90 * DEGREES)

        radar_array = Array(ANT_ELEMENT, geometry.Grid(20, 20, DIST))

        x_axis = Line(
            STEER_ORIGIN,
            STEER_ORIGIN + RIGHT * STEER_MAX_DEG * STEER_SCALE,
            color=WHITE,
            stroke_width=2,
        )
        y_axis = Line(
            STEER_ORIGIN,
            STEER_ORIGIN + UP * STEER_MAX_DEG * STEER_SCALE,
            color=WHITE,
            stroke_width=2,
        )

        def steer_arrow(az_deg: float, el_deg: float) -> Arrow:
            end = (
                STEER_ORIGIN
                + RIGHT * (az_deg * STEER_SCALE)
                + UP * (el_deg * STEER_SCALE)
            )
            if abs(az_deg) < 1e-9 and abs(el_deg) < 1e-9:
                arrow = Arrow(
                    STEER_ORIGIN,
                    STEER_ORIGIN + RIGHT * 1e-6,
                    buff=0,
                    color=YELLOW,
                    stroke_width=4,
                )
                arrow.set_opacity(0)
                return arrow
            return Arrow(STEER_ORIGIN, end, buff=0, color=YELLOW, stroke_width=4)

        def steer_tuple(az_deg: float, el_deg: float) -> tuple[Angle, Angle]:
            return (
                Angle(az_deg, PhaseUnit.DEGREE),
                Angle(el_deg, PhaseUnit.DEGREE),
            )

        # --- 1. Initial state ---
        init_az, init_el = STEER_STEPS[0]
        current_arrow = steer_arrow(init_az, init_el)
        current_surf = radar_array.animate.beam(
            FREQ,
            steer=steer_tuple(init_az, init_el),
            **BEAM_KWARGS,
        )

        self.play(
            Create(x_axis),
            Create(y_axis),
            Create(current_arrow),
            Create(current_surf),
            run_time=1.5,
        )
        self.wait(1)

        # --- 2. Iterate through remaining steer angles ---
        for az_deg, el_deg in STEER_STEPS[1:]:
            next_arrow = steer_arrow(az_deg, el_deg)
            next_surf = radar_array.animate.beam(
                FREQ,
                steer=steer_tuple(az_deg, el_deg),
                **BEAM_KWARGS,
            )

            self.play(
                ReplacementTransform(current_arrow, next_arrow),
                ReplacementTransform(current_surf, next_surf),
                run_time=1,
            )

            current_arrow = next_arrow
            current_surf = next_surf
            self.wait(1)

        self.wait(2)
