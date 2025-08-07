from pathlib import Path
from astropy.coordinates import Angle, EarthLocation

ROOT_DIR = Path(__file__).resolve().parents[1]
DEMO_DATA_DIR = ROOT_DIR / "demo_data"
DATA_DIR = ROOT_DIR / "data"
if not DATA_DIR.exists():
    DATA_DIR.mkdir()


# Location (SRAO)
# lat=37.4548 * u.deg, lon=126.9552 * u.deg, height=100 * u.m
obs_lat_s, obs_lon_s = "37d28m11s", "126d56m32s"
obs_lat, obs_lon = Angle(obs_lat_s), Angle(obs_lon_s)
observatory = EarthLocation.from_geodetic(lat=obs_lat, lon=obs_lon, height=100)

NFFT = 1024
