from enum import Enum


class FigureType(Enum):
    SURFACE = "surface"
    IMAGE = "image"
    SLICE = "slice"


class DirectionDomain(Enum):
    UV = "uv"
    ANGLE = "angle"


class AmplitudeDomain(Enum):
    AntennaFactor = "af"
    Gain = "gain"


class AngleDirection(Enum):
    AZIMUTH = "Azimuth"
    ELEVATION = "Elevation"


ArrayOrientation = AngleDirection


class PhaseUnit(Enum):
    DEGREE = "Degree"
    RADIAN = "Radian"


class AmplitudeUnit(Enum):
    LINEAR = "Linear"
    DECIBEL = "Decibel"


class DistanceUnit(Enum):
    CENTIMETER = 0.01
    METER = 1
    KILOMETER = 1e3


class FrequencyUnit(Enum):
    HERTZ = 1
    KILOHERTZ = 1e3
    MEGAHERTZ = 1e6
    GIGAHERTZ = 1e9


class RotationDirection(Enum):
    CW = "clockwise"
    CCW = "counterclockwise"


RotationUnit = PhaseUnit
