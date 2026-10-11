"""GPS metadata extraction for the image input module.

Real road images collected with a phone or a GPS-enabled camera carry their
location inside the EXIF "GPSInfo" block, as degrees/minutes/seconds (DMS)
rational numbers plus a hemisphere reference (N/S, E/W). This module reads
that block and converts it to plain decimal-degree latitude/longitude.

RDD2022 (and most redistributed research datasets) strip EXIF entirely, so
those images will return None here. That is expected, not a bug -- the
pipeline's geospatial step falls back to synthetic coordinates for images
with no embedded GPS. Real user-uploaded photos (e.g. data/pipeline_test/)
are expected to have it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import ExifTags, Image
from PIL.ExifTags import GPSTAGS, TAGS


@dataclass
class GpsCoordinate:
    """Decoded GPS location for one image."""

    latitude: float
    longitude: float
    altitude: float | None = None
    timestamp: str | None = None


def _dms_to_decimal(dms, ref: str) -> float:
    """Convert an EXIF (degrees, minutes, seconds) triple + hemisphere ref to decimal degrees."""
    degrees, minutes, seconds = dms
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def extract_gps(image_path: str | Path) -> GpsCoordinate | None:
    """Read EXIF GPS metadata from an image.

    Returns a GpsCoordinate if the image has embedded GPS tags, otherwise
    None. None is a normal, expected outcome for datasets with stripped
    EXIF (RDD2022) -- callers should treat it as "no real GPS available"
    and decide on a fallback (e.g. synthetic coordinates), not as an error.
    """
    image_path = Path(image_path)
    img = Image.open(image_path)
    exif = img.getexif()
    if not exif:
        return None

    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    if not gps_ifd:
        return None

    gps_info = {GPSTAGS.get(tag, tag): value for tag, value in gps_ifd.items()}

    if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
        return None

    latitude = _dms_to_decimal(gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N"))
    longitude = _dms_to_decimal(gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E"))

    altitude = None
    if "GPSAltitude" in gps_info:
        try:
            altitude = float(gps_info["GPSAltitude"])
            if gps_info.get("GPSAltitudeRef", 0) == 1:
                altitude = -altitude
        except (TypeError, ValueError):
            altitude = None

    timestamp = None
    if "GPSDateStamp" in gps_info and "GPSTimeStamp" in gps_info:
        try:
            h, m, s = gps_info["GPSTimeStamp"]
            timestamp = f"{gps_info['GPSDateStamp']} {int(h):02d}:{int(m):02d}:{int(s):02d} UTC"
        except (TypeError, ValueError):
            timestamp = None

    return GpsCoordinate(
        latitude=latitude, longitude=longitude, altitude=altitude, timestamp=timestamp
    )


if __name__ == "__main__":
    # Quick manual check against the real geotagged test images.
    test_dir = Path(__file__).resolve().parents[2] / "data" / "pipeline_test"
    for path in sorted(test_dir.glob("*.jpg")):
        coord = extract_gps(path)
        print(path.name, "->", coord)
