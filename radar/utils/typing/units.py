from . import FrequencyUnit, PhaseUnit, DistanceUnit
import numpy as np


class Frequency:
    """A class representing a frequency value with automatic unit conversion support.

    Provides a clean interface for storing frequency data uniformly in Hertz
    and retrieving it in various scaled SI configurations.
    """

    def __init__(self, frequency: float, unit: FrequencyUnit):
        """Initializes a Frequency instance.

        Args:
            frequency (float): The numeric frequency value.
            unit (FrequencyUnit): The unit configuration associated with the value.
        """
        self._frequency_hz = frequency * unit.value

    @property
    def Hz(self) -> float:
        """Returns the frequency value in Hertz (Hz)."""
        return self._frequency_hz

    @property
    def KHz(self) -> float:
        """Returns the frequency value in Kilohertz (kHz)."""
        return self._frequency_hz / FrequencyUnit.KILOHERTZ.value

    @property
    def MHz(self) -> float:
        """Returns the frequency value in Megahertz (MHz)."""
        return self._frequency_hz / FrequencyUnit.MEGAHERTZ.value

    @property
    def GHz(self) -> float:
        """Returns the frequency value in Gigahertz (GHz)."""
        return self._frequency_hz / FrequencyUnit.GIGAHERTZ.value

    def __eq__(self, other: object) -> bool:
        """Evaluates equality based on the absolute value in Hertz."""
        if not isinstance(other, Frequency):
            return NotImplemented
        return self._frequency_hz == other._frequency_hz

    def __lt__(self, other: object) -> bool:
        """Evaluates a less-than comparison based on the absolute value in Hertz."""
        if not isinstance(other, Frequency):
            return NotImplemented
        return self._frequency_hz < other._frequency_hz


class Phase:
    """A class representing an angular phase measurement.

    Standardizes tracking internally using Radians while offering intuitive
    properties for working with Degrees or Radians interchangeably.
    """

    def __init__(self, phase: float, unit: PhaseUnit):
        """Initializes a Phase instance.

        Args:
            phase (float): The numeric magnitude of the phase.
            unit (PhaseUnit): The unit (Degree/Radian) defining the incoming parameter.
        """
        self._phase_rad = phase if unit is PhaseUnit.RADIAN else np.deg2rad(phase)

    @property
    def deg(self) -> float:
        """Returns the phase value scaled in Degrees."""
        return np.rad2deg(self._phase_rad)

    @property
    def rad(self) -> float:
        """Returns the phase value scaled in Radians."""
        return self._phase_rad

    def __eq__(self, other: object) -> bool:
        """Evaluates equality based on the absolute value in Radians."""
        if not isinstance(other, Phase):
            return NotImplemented
        return self._phase_rad == other._phase_rad

    def __lt__(self, other: object) -> bool:
        """Evaluates a less-than comparison based on the absolute value in Radians."""
        if not isinstance(other, Phase):
            return NotImplemented
        return self._phase_rad < other._phase_rad

    def __hash__(self) -> int:
        """Generates a unique hash index based on the internal Radian state value."""
        return hash(self._phase_rad)


# Alias for explicit angular readability contexts
Angle = Phase


class Distance:
    """A class representing a structural spatial distance or length measurement.

    Maintains standard baseline properties in meters while dynamically exposing
    conversions for Metric and Imperial tracking frames.
    """

    def __init__(self, value: float, unit: DistanceUnit):
        """Initializes a Distance instance.

        Args:
            value (float): The numeric spatial distance measurement magnitude.
            unit (DistanceUnit): The measurement dimension framework type.
        """
        self._meters = value * unit.value

    @property
    def m(self) -> float:
        """Returns the distance length measured in Meters (m)."""
        return self._meters

    @property
    def km(self) -> float:
        """Returns the distance length measured in Kilometers (km)."""
        return self._meters / 1000.0

    @property
    def miles(self) -> float:
        """Returns the distance length measured in Miles (mi)."""
        return self._meters / 1609.34

    @property
    def ft(self) -> float:
        """Returns the distance length measured in Feet (ft)."""
        return self._meters / 0.3048

    # --- Comparisons ---

    def __eq__(self, other: object) -> bool:
        """Evaluates distance equality based on absolute meters."""
        if not isinstance(other, Distance):
            return NotImplemented
        return self._meters == other._meters

    def __lt__(self, other: object) -> bool:
        """Evaluates a less-than comparison based on absolute meters."""
        if not isinstance(other, Distance):
            return NotImplemented
        return self._meters < other._meters

    # --- Representation ---
    def __repr__(self) -> str:
        """Generates a clean developer string interpretation showing distance in meters."""
        return f"Distance({self.m}m)"


# Alias for explicit physical dimensionality contexts
Length = Distance
