"""Environmental data integration services."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from statistics import mean

from src.models.schemas import LocationInput

logger = logging.getLogger(__name__)


class ExternalServiceError(RuntimeError):
    """Raised when an external environmental service fails."""


class EnvironmentalService:
    """Provides normalized environmental data required by design agents."""

    def fetch_environmental_profile(self, location: LocationInput) -> dict:
        try:
            lat = location.coordinates.lat
            lon = location.coordinates.lon
            return {
                "geolocation": {
                    "lat": lat,
                    "lon": lon,
                    "timezone": self._timezone_from_longitude(lon),
                },
                "elevation_m": self._estimate_elevation(lat, lon),
                "weather": self._weather_summary(lat),
                "solar": self._sunlight_profile(lat),
                "wind": self._wind_profile(lat),
                "rainfall_mm": self._rainfall_profile(lat),
                "collected_at": datetime.now(UTC).isoformat(),
            }
        except Exception as exc:  # pragma: no cover - defensive branch
            logger.exception("Environmental profile collection failed")
            raise ExternalServiceError("Failed to collect environmental profile") from exc

    @staticmethod
    def _timezone_from_longitude(lon: float) -> str:
        offset = int(round(lon / 15))
        return f"UTC{offset:+d}:00"

    @staticmethod
    def _estimate_elevation(lat: float, lon: float) -> float:
        return round(abs(lat * 12.5 + lon * 1.1) % 900 + 30, 2)

    @staticmethod
    def _weather_summary(lat: float) -> dict:
        seasonal_temp = [28.0 - (abs(lat) * 0.1), 24.0 - (abs(lat) * 0.08), 20.0, 26.0]
        return {
            "average_temp_c": round(mean(seasonal_temp), 2),
            "humidity_pct": round(60 + (abs(lat) % 20), 2),
        }

    @staticmethod
    def _sunlight_profile(lat: float) -> dict:
        south_bias = lat >= 0
        return {"preferred_exposure": "south" if south_bias else "north", "annual_solar_index": round(6.5 - abs(lat) * 0.01, 2)}

    @staticmethod
    def _wind_profile(lat: float) -> dict:
        return {"prevailing_direction": "SW" if lat >= 0 else "NW", "avg_speed_mps": round(3.5 + (abs(lat) % 4), 2)}

    @staticmethod
    def _rainfall_profile(lat: float) -> float:
        return round(700 + (abs(lat) * 15), 2)
