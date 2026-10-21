from pathlib import Path

import requests
import xarray as xr
import numpy as np
from shapely.geometry import Polygon
from shapely import contains_xy


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path.home() / "dupla_safra_view_web"

# ------------------------------------------------------------
# Historical period
# ------------------------------------------------------------

FIRST_YEAR = 1981
LAST_YEAR = 2025

# ------------------------------------------------------------
# CHIRPS v2
# ------------------------------------------------------------

BASE_URL = (
    "https://data.chc.ucsb.edu/products/"
    "CHIRPS-2.0/global_daily/netcdf/p05/"
)

# ------------------------------------------------------------
# Directories
# ------------------------------------------------------------

TEMP_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "chirps"
    / "tmp"
)

OUTPUT_BASE_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "chirps"
)


# ============================================================
# BR CENTRAL-2 POLYGON
# ============================================================

# Coordinates supplied by the user as:
#     (latitude, longitude)
#
# Shapely uses:
#     (longitude, latitude)
#
# Therefore the coordinates below are stored as (lon, lat).

POLYGON_COORDS = [
    (-44.5,  -3.5),
    (-54.0,  -3.5),
    (-54.0,  -6.5),
    (-57.0,  -6.5),
    (-57.0,  -7.5),
    (-65.0,  -7.5),
    (-65.0, -14.0),
    (-62.5, -14.0),
    (-62.5, -16.0),
    (-61.5, -16.0),
    (-61.5, -16.5),
    (-55.5, -16.5),
    (-55.5, -17.5),
    (-52.0, -17.5),
    (-52.0, -14.5),
    (-45.5, -14.5),
    (-45.5,  -9.5),
    (-44.5,  -9.5),
    (-44.5,  -3.5),
]

POLYGON = Polygon(
    POLYGON_COORDS
)

if not POLYGON.is_valid:
    raise ValueError(
        "The supplied BR Central-2 polygon is invalid."
    )

LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = POLYGON.bounds

# ============================================================
# DOWNLOAD
# ============================================================

def download_file(url, output_file):

    if output_file.exists():

        print(
            f"\nAlready downloaded:"
            f" {output_file.name}"
        )

        return

    print("\nDownloading:")
    print(url)

    response = requests.get(
        url,
        stream=True,
        timeout=300
    )

    response.raise_for_status()

    total_size = int(
        response.headers.get(
            "content-length",
            0
        )
    )

    downloaded = 0

    with open(output_file, "wb") as f:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:

                f.write(chunk)
                downloaded += len(chunk)

                if total_size:

                    percentage = (
                        downloaded
                        / total_size
                        * 100
                    )

                    print(
                        f"\rProgress: {percentage:6.2f}%",
                        end=""
                    )

    print("\nDownload complete.")


# ============================================================
# DOWNLOAD ONE CALENDAR YEAR
# ============================================================

def download_year(year):

    filename = (
        f"chirps-v2.0."
        f"{year}.days_p05.nc"
    )

    url = BASE_URL + filename

    output_file = TEMP_DIR / filename

    download_file(
        url,
        output_file
    )

    return output_file


# ============================================================
# CREATE POLYGON MASK
# ============================================================

def create_polygon_mask(ds):
    """
    Keep only CHIRPS 0.05° pixels whose CENTER is inside
    the BR Central-2 polygon.

    The polygon itself is the only spatial selection.
    No pixel-area intersection is used.

    Important:
        POLYGON.contains(...) is used conceptually here,
        so a pixel center exactly on the polygon boundary
        is NOT included.
    """

    lon = ds.longitude.values
    lat = ds.latitude.values

    lon2d, lat2d = np.meshgrid(
        lon,
        lat
    )

    # --------------------------------------------------------
    # Pixel-center criterion
    # --------------------------------------------------------
    #
    # Each point represents the CENTER of the CHIRPS pixel.
    #
    # True  -> center is strictly inside polygon
    # False -> center is outside or exactly on boundary
    # --------------------------------------------------------

    mask = contains_xy(
        POLYGON,
        lon2d,
        lat2d
    )

    return xr.DataArray(
        mask,
        coords={
            "latitude": lat,
            "longitude": lon
        },
        dims=[
            "latitude",
            "longitude"
        ],
        name="polygon_mask"
    )



# ============================================================
# PROCESS HYDROLOGICAL YEAR
# ============================================================

def process_hydrological_year(year):

    next_year = year + 1

    print("\n" + "=" * 70)
    print(
        f"PROCESSING {year}/{next_year}"
    )
    print("=" * 70)

    start_date = f"{year}-07-01"
    end_date = f"{next_year}-06-30"

    output_dir = (
        OUTPUT_BASE_DIR
        / f"{year}_{next_year}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / f"chirps_v2_{year}_{next_year}.nc"
    )

    # --------------------------------------------------------
    # Skip existing output
    # --------------------------------------------------------

    if output_file.exists():

        print(
            "\nOutput already exists:"
        )

        print(output_file)

        return

    # --------------------------------------------------------
    # Files already downloaded
    # --------------------------------------------------------

    file1 = TEMP_DIR / (
        f"chirps-v2.0."
        f"{year}.days_p05.nc"
    )

    file2 = TEMP_DIR / (
        f"chirps-v2.0."
        f"{next_year}.days_p05.nc"
    )

    # --------------------------------------------------------
    # Open yearly files
    # --------------------------------------------------------

    print(
        f"\nOpening {file1.name}"
    )

    ds1 = xr.open_dataset(
        file1
    )

    print(
        f"Opening {file2.name}"
    )

    ds2 = xr.open_dataset(
        file2
    )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Bounding box pre-selection
    #
    # This is only an efficiency step before the polygon mask.
    # It does NOT define the final spatial domain.
    # --------------------------------------------------------

    if ds1.latitude.values[0] < ds1.latitude.values[-1]:
        lat_slice = slice(
            LAT_MIN,
            LAT_MAX
        )
    else:
        lat_slice = slice(
            LAT_MAX,
            LAT_MIN
        )

    if ds1.longitude.values[0] < ds1.longitude.values[-1]:
        lon_slice = slice(
            LON_MIN,
            LON_MAX
        )
    else:
        lon_slice = slice(
            LON_MAX,
            LON_MIN
        )

    ds1 = ds1.sel(
        latitude=lat_slice,
        longitude=lon_slice
    )

    ds2 = ds2.sel(
        latitude=lat_slice,
        longitude=lon_slice
    )

    # --------------------------------------------------------
    # Polygon mask
    # --------------------------------------------------------

    mask = create_polygon_mask(
        ds1
    )

    ds1 = ds1.where(
        mask
    )

    # ds2 has the same spatial grid, so the same mask applies.
    ds2 = ds2.where(
        mask
    )

    n_inside = int(
        mask.sum().values
    )

    print(
        f"Pixels with center inside polygon: "
        f"{n_inside}"
    )

    # --------------------------------------------------------
    # Combine years
    # --------------------------------------------------------

    ds = xr.concat(
        [ds1, ds2],
        dim="time"
    )

    # --------------------------------------------------------
    # Select hydrological year
    # --------------------------------------------------------

    ds = ds.sel(
        time=slice(
            start_date,
            end_date
        )
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\nFinal dataset:")

    print(ds)

    print(
        "\nTime:"
        f" {ds.time.values[0]}"
        f" → "
        f"{ds.time.values[-1]}"
    )

    print(
        "\nDimensions:"
    )

    print(ds.sizes)

    print(
        "\nLatitude:"
        f" {float(ds.latitude.min())}"
        f" → "
        f"{float(ds.latitude.max())}"
    )

    print(
        "\nLongitude:"
        f" {float(ds.longitude.min())}"
        f" → "
        f"{float(ds.longitude.max())}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\nSaving:"
    )

    print(output_file)

    ds.to_netcdf(
        output_file
    )

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    ds1.close()
    ds2.close()
    ds.close()

    print(
        "\nCompleted:"
    )

    print(output_file)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_BASE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CHIRPS v2 HISTORICAL SERIES"
    )

    print(
        f"{FIRST_YEAR}/{FIRST_YEAR + 1}"
        f" → "
        f"{LAST_YEAR}/{LAST_YEAR + 1}"
    )

    print(
        "Resolution: 0.05°"
    )

    print(
        "Daily precipitation"
    )

    print(
        "\nPolygon bounds:"
    )

    print(
        f"Longitude:"
        f" {LON_MIN} → {LON_MAX}"
    )

    print(
        f"Latitude:"
        f" {LAT_MIN} → {LAT_MAX}"
    )

    print(
        "\n"
        + "=" * 70
    )

    # ========================================================
    # STEP 1
    # Download every calendar year only once
    # ========================================================

    print(
        "\nSTEP 1 — DOWNLOADING CHIRPS"
    )

    for year in range(
        FIRST_YEAR,
        LAST_YEAR + 2
    ):

        download_year(
            year
        )

    # ========================================================
    # STEP 2
    # Build hydrological years
    # ========================================================

    print(
        "\nSTEP 2 — BUILDING HYDROLOGICAL YEARS"
    )

    for year in range(
        FIRST_YEAR,
        LAST_YEAR + 1
    ):

        process_hydrological_year(
            year
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HISTORICAL CHIRPS SERIES COMPLETED"
    )

    print(
        f"{FIRST_YEAR}/{FIRST_YEAR + 1}"
        f" → "
        f"{LAST_YEAR}/{LAST_YEAR + 1}"
    )

    print("=" * 70)
