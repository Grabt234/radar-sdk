import numpy as np

from radar.utils.typing.constants import RadarConstants, DataHeader
from radar.utils.typing.enums import (
    DirectionDomain,
    PhaseUnit,
    AmplitudeDomain,
    AmplitudeUnit,
)


def test_radar_constants():
    assert RadarConstants.c == 299792458
    assert np.allclose(RadarConstants.to_rad, np.pi / 180)
    assert np.allclose(RadarConstants.to_deg, 180 / np.pi)


def test_direction_domain_headers():
    # UV domain
    assert DataHeader.direction_domain_headers(
        DirectionDomain.UV, PhaseUnit.DEGREE
    ) == (
        DataHeader.U,
        DataHeader.V,
    )
    assert DataHeader.direction_domain_headers(
        DirectionDomain.UV, PhaseUnit.RADIAN
    ) == (
        DataHeader.U,
        DataHeader.V,
    )

    # Angle domain, Degree unit
    assert DataHeader.direction_domain_headers(
        DirectionDomain.ANGLE, PhaseUnit.DEGREE
    ) == (
        DataHeader.AZIMUTH_DEG,
        DataHeader.ELEVATION_DEG,
    )

    # Angle domain, Radian unit
    assert DataHeader.direction_domain_headers(
        DirectionDomain.ANGLE, PhaseUnit.RADIAN
    ) == (
        DataHeader.AZIMUTH_RAD,
        DataHeader.ELEVATION_RAD,
    )


def test_amplitude_domain_headers():
    # AntennaFactor
    assert (
        DataHeader._amplitude_domain_headers(
            AmplitudeDomain.AntennaFactor, AmplitudeUnit.DECIBEL
        )
        == DataHeader.ANTENNA_FACTOR_DB
    )

    assert (
        DataHeader._amplitude_domain_headers(
            AmplitudeDomain.AntennaFactor, AmplitudeUnit.LINEAR
        )
        == DataHeader.ANTENNA_FACTOR_LINEAR
    )

    # Gain
    assert (
        DataHeader._amplitude_domain_headers(
            AmplitudeDomain.Gain, AmplitudeUnit.DECIBEL
        )
        == DataHeader.BEAM_GAIN_DB
    )

    assert (
        DataHeader._amplitude_domain_headers(AmplitudeDomain.Gain, AmplitudeUnit.LINEAR)
        == DataHeader.BEAM_GAIN_LINEAR
    )
