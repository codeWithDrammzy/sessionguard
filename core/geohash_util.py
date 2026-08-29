"""
SessionGuard geohash codec (encode + decode), stdlib only.
==========================================================

A standard-base32 geohash encodes a lat/lon into a short string
(10 digits of precision is ~4.8m; 6 is ~1.2km x 0.6km -- the precision
the dataset uses). No third-party geohash package is required, and none
should be introduced: the synthetic generators must be able to *encode*
real Nigerian city locations and the feature engine must be able to
*decode* any stored geohash to a coordinate pair, and both must agree
on the exact same cell boundaries -- so both live here in one module.

    geohash_encode(6.45, 3.39)      -> "s1hrn8..."   (Lagos, ~1km cell)
    geohash_decode("s1tst0")        -> (lat, lon) cell centroid
    haversine_km(a, b)              -> great-circle distance in km

The alphabet is the official geohash set: 0123456789bcdefghjkmnpqrstuvwxyz
(no 'a', 'i', 'l', 'o' -- those four letters are omitted to avoid
lookalike/confusion pairs).
"""

import math

# Official geohash base-32 alphabet (drops a/i/l/o for readability).
GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
_CHAR_TO_INDEX = {ch: i for i, ch in enumerate(GEOHASH_ALPHABET)}

_EARTH_RADIUS_KM = 6371.0

# Real Nigerian city centres, shared by BOTH dataset generators so the whole
# synthetic population lives in realistic, physically-coherent cities.
# (name, lat, lon). Precision-6 encoding gives ~1.2km x 0.6km cells, so
# "home" and "work" drawn a few km apart both sit inside the SAME city and
# well under the impossible-travel distance floor. IDs/towers/IPs remain
# opaque synthetic tokens -- only the geography is realistic; no real
# addresses are encoded.
NIGERIAN_CITIES = [
    ("Lagos", 6.45, 3.39),
    ("Abuja", 9.08, 7.48),
    ("Kaduna", 10.51, 7.42),
    ("Port Harcourt", 4.82, 7.03),
    ("Ibadan", 7.38, 3.93),
    ("Kano", 11.99, 8.52),
    ("Enugu", 6.46, 7.55),
    ("Benin City", 6.34, 5.62),
]

# International cities used by LOUD attack archetypes (credential theft,
# obvious SIM swap). Each is far enough from every Nigerian city (>3000 km)
# that a login there 10-40 minutes after the victim's last session implies
# a physically impossible speed (~>5000 km/h vs the 900 km/h threshold).
# (name, lat, lon).
FAR_ATTACK_CITIES = [
    ("Cairo", 30.04, 31.24),
    ("Johannesburg", -26.20, 28.05),
    ("London", 51.51, -0.13),
    ("Dubai", 25.20, 55.27),
]


def geohash_decode(geohash):
    """
    Decode a geohash string to its cell centre ``(lat, lon)``.

    Returns ``None`` for empty input or any character outside the base-32
    alphabet (defensive: a live request with garbage in the field must
    degrade to "cannot judge travel" rather than raise).
    """
    geohash = (geohash or "").strip().lower()
    if not geohash:
        return None

    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0
    even = True  # geohash interleaves longitude first, then latitude
    for ch in geohash:
        idx = _CHAR_TO_INDEX.get(ch)
        if idx is None:
            return None
        for bit in range(4, -1, -1):
            bit_val = (idx >> bit) & 1
            if even:  # longitude
                mid = (lon_min + lon_max) / 2.0
                if bit_val:
                    lon_min = mid
                else:
                    lon_max = mid
            else:  # latitude
                mid = (lat_min + lat_max) / 2.0
                if bit_val:
                    lat_min = mid
                else:
                    lat_max = mid
            even = not even
    return (lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0


def geohash_encode(lat, lon, precision=6):
    """
    Encode a lat/lon into a ``precision``-char geohash (standard algorithm,
    the exact inverse of :func:`geohash_decode`).
    """
    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0
    result, bit, ch = [], 0, 0
    even = True
    for _ in range(precision * 5):
        if even:  # longitude
            mid = (lon_min + lon_max) / 2.0
            if lon >= mid:
                lon_min = mid
                b = 1
            else:
                lon_max = mid
                b = 0
        else:  # latitude
            mid = (lat_min + lat_max) / 2.0
            if lat >= mid:
                lat_min = mid
                b = 1
            else:
                lat_max = mid
                b = 0
        even = not even
        ch = (ch << 1) | b
        bit += 1
        if bit == 5:
            result.append(GEOHASH_ALPHABET[ch])
            bit, ch = 0, 0
    return "".join(result)


def haversine_km(point_a, point_b):
    """
    Great-circle distance in kilometres between two (lat, lon) tuples.
    """
    lat1, lon1 = point_a
    lat2, lon2 = point_b
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    h = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


if __name__ == "__main__":
    # Self-test: encode->decode round-trip lands within the cell.
    for lat, lon in [(6.45, 3.39), (10.51, 7.42), (9.08, 7.48), (-26.2, 28.05)]:
        g = geohash_encode(lat, lon)
        back = geohash_decode(g)
        err = max(abs(back[0] - lat), abs(back[1] - lon))
        print(f"({lat},{lon}) -> '{g}' -> {tuple(round(v, 4) for v in back)}  (err={err:.3f})")