from manim import (
    DOWN,
    LEFT,
    RIGHT,
    DEGREES,
    Text,
    ThreeDScene,
    Create,
    ReplacementTransform,
    Write,
)
from radar.components import geometry, Element
from radar.components.array import Array
from radar.utils.calculate import convert, pattern
from radar.utils.typing import PhaseUnit, Angle, FrequencyUnit
from radar.utils.typing.enums import AmplitudeDomain, AmplitudeUnit, DirectionDomain
from radar.utils.typing.units import Frequency

# --- Configuration Constants ---
FREQ = Frequency(1, FrequencyUnit.GIGAHERTZ)
DIST = convert.cf_to_min_dist(FREQ)
BOUNDS = (Angle(-60, PhaseUnit.DEGREE), Angle(60, PhaseUnit.DEGREE))

# --- Define Geometry ---
LIN_GEOMETRY = geometry.Cross(8, 8, DIST)


# --- Define Antenna Elements ---
# Helper function to prevent repeating the setup lines for each Element
def create_element(pat):
    return Element(pat, BOUNDS, BOUNDS, FREQ, 1)


ELEMENTS = [
    create_element(
        pattern.Sinc((Angle(30, PhaseUnit.DEGREE), Angle(30, PhaseUnit.DEGREE)))
    ),
    create_element(
        pattern.Gaussian((Angle(40, PhaseUnit.DEGREE), Angle(40, PhaseUnit.DEGREE)))
    ),
    create_element(
        pattern.Gaussian((Angle(15, PhaseUnit.DEGREE), Angle(15, PhaseUnit.DEGREE)))
    ),
    create_element(pattern.Cosine(1)),
]


class Video(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=45 * DEGREES, theta=-90 * DEGREES, zoom=0.75)

        # Common configuration for element and array beam generation
        beam_kwargs = {
            "freq": FREQ,
            "domain": DirectionDomain.ANGLE,
            "unit": PhaseUnit.DEGREE,
            "amp_domain": AmplitudeDomain.Gain,
            "amp_unit": AmplitudeUnit.DECIBEL,
        }

        # --- Static UI Labels ---
        label_element = (
            Text("Element Pattern", font_size=24).next_to(LEFT * 4).shift(DOWN * 3)
        )
        label_array = (
            Text("Array Pattern", font_size=24).next_to(RIGHT * 1.65).shift(DOWN * 3.1)
        )
        self.add_fixed_in_frame_mobjects(label_element, label_array)

        # --- 1. Initial State (Sinc Element) ---
        first_elem = ELEMENTS[0]
        first_arr = Array(first_elem, LIN_GEOMETRY)

        current_elem_surf = first_elem.animate.beam(position=LEFT * 3.5, **beam_kwargs)
        current_beam_surf = first_arr.animate.beam(position=RIGHT * 3.5, **beam_kwargs)

        self.play(
            Create(current_elem_surf),
            Create(current_beam_surf),
            Write(label_element),
            Write(label_array),
            run_time=1.5,
        )
        self.wait(1)

        # --- 2. Loop Through Remaining Elements ---
        for idx, next_elem in enumerate(ELEMENTS[1:], start=2):
            next_arr = Array(next_elem, LIN_GEOMETRY)

            next_elem_surf = next_elem.animate.beam(position=LEFT * 3.5, **beam_kwargs)
            next_beam_surf = next_arr.animate.beam(position=RIGHT * 3.5, **beam_kwargs)

            # Special case condition carried over from your original script for the final Cosine loop
            if idx == 4:
                next_beam_surf.shift(DOWN * 2)

            self.play(
                ReplacementTransform(current_elem_surf, next_elem_surf),
                ReplacementTransform(current_beam_surf, next_beam_surf),
                run_time=1.5,
            )
            self.wait(1)

            # Update tracking references for the next sequence iteration
            current_elem_surf = next_elem_surf
            current_beam_surf = next_beam_surf
