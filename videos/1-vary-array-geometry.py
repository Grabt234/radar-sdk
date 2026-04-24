from manim import BLUE, LEFT, RIGHT, DEGREES, ThreeDScene, Create, ReplacementTransform
from radar.components import geometry, Element
from radar.components.array import Array
from radar.utils.calculate import convert, pattern
from radar.utils.typing import PhaseUnit, Angle, FrequencyUnit
from radar.utils.typing.enums import (
    AmplitudeDomain,
    AmplitudeUnit,
    ArrayOrientation,
    DirectionDomain,
)
from radar.utils.typing.units import Frequency

# --- Configuration Constants ---
FREQ = Frequency(1, FrequencyUnit.GIGAHERTZ)
DIST = convert.cf_to_min_dist(FREQ)

# --- Element Pattern Setup ---
BW_TUPLE = (Angle(15, PhaseUnit.DEGREE), Angle(15, PhaseUnit.DEGREE))
BOUNDS = (Angle(-60, PhaseUnit.DEGREE), Angle(60, PhaseUnit.DEGREE))
ANT_ELEMENT = Element(pattern.Sinc(BW_TUPLE), BOUNDS, BOUNDS, FREQ, 1)

# --- Define Array Geometries ---
GEOMETRIES = [
    geometry.Linear(10, ArrayOrientation.AZIMUTH, DIST),
    geometry.Linear(10, ArrayOrientation.ELEVATION, DIST),
    geometry.Circular(10, DIST),
    geometry.Grid(10, 10, DIST),
    geometry.Grid(5, 5, DIST),
    geometry.Cross(5, 5, DIST),
    geometry.Cross(15, 15, DIST),
]

# Build the Array objects dynamically
ARRAYS = [Array(ANT_ELEMENT, geo) for geo in GEOMETRIES]


class Video(ThreeDScene):
    def construct(self):
        # Set isometric 3D camera viewpoint
        self.set_camera_orientation(phi=45 * DEGREES, theta=-90 * DEGREES)

        # Common configuration for beam generation
        beam_kwargs = {
            "freq": FREQ,
            "pos": RIGHT * 3.5,
            "domain": DirectionDomain.ANGLE,
            "unit": PhaseUnit.DEGREE,
            "amp_domain": AmplitudeDomain.Gain,
            "amp_unit": AmplitudeUnit.DECIBEL,
        }

        # --- 1. Initial State ---
        current_dots = ARRAYS[0].animate.geometry(LEFT * 3.5, BLUE)
        current_surf = ARRAYS[0].animate.beam(**beam_kwargs)

        self.play(Create(current_dots), Create(current_surf), run_time=1.5)
        self.wait(1)

        # --- 2. Loop Through Remaining Arrays ---
        for next_array in ARRAYS[1:]:
            next_dots = next_array.animate.geometry(LEFT * 3.5, BLUE)
            next_surf = next_array.animate.beam(**beam_kwargs)

            self.play(
                ReplacementTransform(current_dots, next_dots),
                ReplacementTransform(current_surf, next_surf),
                run_time=1.5,
            )
            self.wait(1)

            # Update tracking variables for the next iteration
            current_dots = next_dots
            current_surf = next_surf

        self.wait(0.5)  # Slight extra padding at the very end
