import numpy as np

from radar.utils.typing.enums import (
    AmplitudeDomain,
    AmplitudeUnit,
    DirectionDomain,
    PhaseUnit,
)


class RadarConstants:
    c = 299792458
    to_rad = np.pi / 180
    to_deg = 180 / np.pi


class DataHeader:
    AZIMUTH_RAD = "az_rad"
    AZIMUTH_DEG = "az_deg"
    ELEVATION_RAD = "el_rad"
    ELEVATION_DEG = "el_deg"
    BEAM_GAIN_DB = "beam_gain_db"
    BEAM_GAIN_LINEAR = "beam_gain_lin"
    X_POS_M = "x_m"
    Y_POS_M = "y_m"
    ANTENNA_FACTOR_DB = "af_db"
    ANTENNA_FACTOR_LINEAR = "af_lin"
    U = "u"
    V = "v"
    UV_MASK = "uv_mask"
    FREQ_GAIN_DB = "freq_gain_db"
    FREQ_GAIN_LINEAR = "freq_gain_lin"
    FREQ_FREQS = "freq_freqs"

    @staticmethod
    def direction_domain_headers(
        domain: DirectionDomain, phase_unit: PhaseUnit
    ) -> tuple[str, str]:
        """Maps domain configurations to their matching DataHeader string column keys.

        Args:
            domain (DirectionDomain): Target spatial domain (UV or Angular).
            phase_unit (PhaseUnit): Target configuration unit (Degree or Radian).

        Returns:
            tuple[str, str]: A pair of string header names corresponding to the
            horizontal (Azimuth/U) and vertical (Elevation/V) axes.
        """
        if domain is DirectionDomain.UV:
            return DataHeader.U, DataHeader.V

        if phase_unit is PhaseUnit.DEGREE:
            return DataHeader.AZIMUTH_DEG, DataHeader.ELEVATION_DEG

        return DataHeader.AZIMUTH_RAD, DataHeader.ELEVATION_RAD

    @staticmethod
    def _amplitude_domain_headers(
        domain: AmplitudeDomain, amplitude_unit: AmplitudeUnit
    ) -> str:
        """Maps amplitude configurations to their matching DataHeader string column keys.

        Args:
            domain (AmplitudeDomain): Domain context (AntennaFactor vs Gain).
            amplitude_unit (AmplitudeUnit): Mathematical scale context (Decibel vs Linear).

        Returns:
            str: The exact DataFrame column name corresponding to the configured parameters.
        """
        if amplitude_unit is AmplitudeUnit.DECIBEL:
            return (
                DataHeader.ANTENNA_FACTOR_DB
                if domain is AmplitudeDomain.AntennaFactor
                else DataHeader.BEAM_GAIN_DB
            )
        else:
            return (
                DataHeader.ANTENNA_FACTOR_LINEAR
                if domain is AmplitudeDomain.AntennaFactor
                else DataHeader.BEAM_GAIN_LINEAR
            )
