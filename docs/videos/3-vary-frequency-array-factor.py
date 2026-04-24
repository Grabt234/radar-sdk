from manim import (
    BLUE,
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
from radar.components.response import FrequencyResponse
from radar.utils.calculate import convert, pattern
from radar.utils.typing import PhaseUnit, Angle, FrequencyUnit
from radar.utils.typing.constants import DataHeader
from radar.utils.typing.enums import AmplitudeDomain, AmplitudeUnit, DirectionDomain
from radar.utils.typing.units import Frequency

import polars as pl

# --- Configuration Constants ---
F1 = Frequency(0.1, FrequencyUnit.MEGAHERTZ)
F2 = Frequency(5, FrequencyUnit.GIGAHERTZ)

# Cleaned up Dataframe & Frequency response init
DF = pl.DataFrame(
    {DataHeader.FREQ_GAIN_DB: [0, 0], DataHeader.FREQ_FREQS: [F1.Hz, F2.Hz]}
)
F_RESP = FrequencyResponse(df=DF)

FREQ_1G = Frequency(1, FrequencyUnit.GIGAHERTZ)
DIST_1G = convert.cf_to_min_dist(FREQ_1G)

# --- Element Pattern ---
BW_TUPLE = (Angle(15, PhaseUnit.DEGREE), Angle(15, PhaseUnit.DEGREE))
BOUNDS = (Angle(-60, PhaseUnit.DEGREE), Angle(60, PhaseUnit.DEGREE))
ANT_ELEMENT = Element(pattern.Sinc(BW_TUPLE), BOUNDS, BOUNDS, F_RESP, 1)


class Video(ThreeDScene):
    def construct(self):
        # Set isometric 3D camera viewpoint
        self.set_camera_orientation(phi=30 * DEGREES, theta=-90 * DEGREES)

        left_pos = LEFT * 3.5
        right_pos = RIGHT * 3.2

        # Configuration steps: (Frequency_GHz, Distance/Geometry_Param)
        config_steps = [
            (1.00, DIST_1G),
            (1.05, DIST_1G),
            (1.10, DIST_1G),
            (1.15, DIST_1G),
            (1.10, DIST_1G),
            (1.05, DIST_1G),
            (0.95, DIST_1G),
            (0.90, DIST_1G),
            (0.85, DIST_1G),
            (0.10, DIST_1G),
        ]

        # Common configuration for beam generation
        beam_kwargs = {
            "pos": right_pos,
            "domain": DirectionDomain.ANGLE,
            "unit": PhaseUnit.DEGREE,
            "amp_domain": AmplitudeDomain.Gain,
            "amp_unit": AmplitudeUnit.DECIBEL,
        }

        # Helper to generate text labels uniformly
        def build_label(f_val):
            return (
                Text(f"Frequency: {f_val:.2f} GHz", font_size=24)
                .next_to(LEFT * 5.4)
                .shift(DOWN * 2)
            )

        # --- 1. Initial State Setup ---
        init_freq, init_dist = config_steps[0]

        radar_array = Array(ANT_ELEMENT, geometry.Grid(10, 10, init_dist))
        current_dots = radar_array.animate.geometry(left_pos, BLUE)
        current_surf = radar_array.animate.beam(
            Frequency(init_freq, FrequencyUnit.GIGAHERTZ), **beam_kwargs
        )

        label_text = build_label(init_freq)
        self.add_fixed_in_frame_mobjects(label_text)

        self.play(
            Create(current_dots), Create(current_surf), Write(label_text), run_time=1.5
        )
        self.wait(1)

        # --- 2. Iterate through remaining configurations ---
        for freq_val, dist_val in config_steps[1:]:
            # Core fix: dynamically passing dist_val now
            radar_array = Array(ANT_ELEMENT, geometry.Grid(10, 10, dist_val))

            next_dots = radar_array.animate.geometry(left_pos, BLUE)
            next_surf = radar_array.animate.beam(
                Frequency(freq_val, FrequencyUnit.GIGAHERTZ), **beam_kwargs
            )
            next_label = build_label(freq_val)

            self.add_fixed_in_frame_mobjects(next_label)

            self.play(
                ReplacementTransform(current_surf, next_surf),
                ReplacementTransform(label_text, next_label),
                run_time=1,
            )

            # Update loop tracking variables
            current_dots = next_dots
            current_surf = next_surf
            label_text = next_label
            self.wait(1)

        self.wait(2)
