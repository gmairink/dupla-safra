import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path.home() / "dupla_safra_view_web"

RAW_BASE_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "era5"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "sst"
)

OUTPUT_ANOMALIES_FILE = (
    OUTPUT_DIR
    / "sst_anomalies_1982_2026.csv"
)

OUTPUT_ABSOLUTE_FILE = (
    OUTPUT_DIR
    / "sst_mean_regions_1982_2026.csv"
)

FIRST_YEAR = 1982
LAST_YEAR = 2026

# SST climatology
CLIM_START = 1982
CLIM_END = 2024


# ============================================================
# REGIONS
# ============================================================
#
# These are exactly the regions from the old mask:
#
# A4-2 -> Atlantic region used here as RA
# P4-2 -> Pacific region used here as RP1
# P5-2 -> Pacific region used here as RP2-2
#
# The masks are constructed as unions of rectangles, exactly
# as in the original processing.
# ============================================================

REGION_BOXES = {

    # --------------------------------------------------------
    # A4-2 = RA
    # --------------------------------------------------------

    "RA": [
        {
            "lat_north": 21.325,
            "lat_south": 10.875,
            "lon_west": -32.825,
            "lon_east": -27.325,
        },
        {
            "lat_north": 10.875,
            "lat_south": 4.875,
            "lon_west": -27.325,
            "lon_east": -21.375,
        },
    ],

    # --------------------------------------------------------
    # P4-2 = RP1
    # --------------------------------------------------------

    "RP1": [
        {
            "lat_north": -34.125,
            "lat_south": -39.635,
            "lon_west": -149.825,
            "lon_east": -137.825,
        },
        {
            "lat_north": -35.625,
            "lat_south": -42.125,
            "lon_west": -139.325,
            "lon_east": -126.325,
        },
    ],

    # --------------------------------------------------------
    # P5-2 = RP2-2
    # --------------------------------------------------------

    "RP2-2": [
        {
            "lat_north": -25.125,
            "lat_south": -29.125,
            "lon_west": -120.325,
            "lon_east": -110.325,
        },
    ],
}


# ============================================================
# CREATE MASK
# ============================================================

def create_region_mask(data, boxes):
    """
    Create a regional mask using rectangular boxes.

    This reproduces the old method:
        latitude <= north
        latitude >= south
        longitude >= west
        longitude <= east

    Multiple boxes are combined with OR.
    """

    mask = xr.zeros_like(
        data.isel(
            valid_time=0,
            drop=True
        ),
        dtype=bool
    )

    for box in boxes:

        box_mask = (
            (data.latitude <= box["lat_north"])
            & (data.latitude >= box["lat_south"])
            & (data.longitude >= box["lon_west"])
            & (data.longitude <= box["lon_east"])
        )

        mask = mask | box_mask

    return mask


# ============================================================
# AGGREGATE 0.25° → 0.5°
# ============================================================

def aggregate_to_05deg(ds):
    """
    Aggregate ERA5 SST from 0.25° to 0.5° by arithmetic mean
    of 2 x 2 neighboring cells.

    No area weighting is applied.
    """

    sst = ds["sst"]

    dlat = abs(
        float(
            sst.latitude.values[1]
            - sst.latitude.values[0]
        )
    )

    dlon = abs(
        float(
            sst.longitude.values[1]
            - sst.longitude.values[0]
        )
    )

    print(
        f"Original resolution: "
        f"{dlat:.4f}° × {dlon:.4f}°"
    )

    sst_05 = (
        sst
        .coarsen(
            latitude=2,
            longitude=2,
            boundary="trim"
        )
        .mean(
            skipna=True
        )
    )

    dlat_new = abs(
        float(
            sst_05.latitude.values[1]
            - sst_05.latitude.values[0]
        )
    )

    dlon_new = abs(
        float(
            sst_05.longitude.values[1]
            - sst_05.longitude.values[0]
        )
    )

    print(
        f"Aggregated resolution: "
        f"{dlat_new:.4f}° × {dlon_new:.4f}°"
    )

    return sst_05


# ============================================================
# SPATIAL MEAN
# ============================================================

def spatial_mean(data, mask):
    """
    Simple arithmetic mean over all valid grid cells inside
    the regional mask.

    No area weighting.
    """

    masked = data.where(
        mask
    )

    return masked.mean(
        dim=(
            "latitude",
            "longitude"
        ),
        skipna=True
    )


# ============================================================
# PROCESS ONE YEAR
# ============================================================

def process_year(year):

    file = (
        RAW_BASE_DIR
        / str(year)
        / f"era5_sst_{year}.nc"
    )

    if not file.exists():

        raise FileNotFoundError(
            f"ERA5 file not found:\n{file}"
        )

    print("\n" + "=" * 70)
    print(f"PROCESSING SST {year}")
    print("=" * 70)

    ds = xr.open_dataset(
        file
    )

    # --------------------------------------------------------
    # Convert longitude 0–360 → -180–180
    # --------------------------------------------------------

    ds = ds.assign_coords(
        longitude=(
            (ds.longitude + 180) % 360
        ) - 180
    )

    ds = ds.sortby(
        "longitude"
    )

    # --------------------------------------------------------
    # First aggregate 0.25° → 0.5°
    # --------------------------------------------------------

    sst_05 = aggregate_to_05deg(
        ds
    )

    # --------------------------------------------------------
    # Create masks on the 0.5° grid
    # --------------------------------------------------------

    masks = {}

    for region, boxes in REGION_BOXES.items():

        masks[region] = create_region_mask(
            sst_05,
            boxes
        )

        print(
            f"{region}: "
            f"{int(masks[region].sum())} grid cells"
        )

    # --------------------------------------------------------
    # January
    # --------------------------------------------------------

    january = sst_05.sel(
        valid_time=f"{year}-01-01"
    )

    # --------------------------------------------------------
    # February
    # --------------------------------------------------------

    february = sst_05.sel(
        valid_time=f"{year}-02-01"
    )

    # --------------------------------------------------------
    # Regional means
    # --------------------------------------------------------

    # P4-2 / RP1 -> January
    rp1 = spatial_mean(
        january,
        masks["RP1"]
    ).item()

    # A4-2 / RA -> February
    ra = spatial_mean(
        february,
        masks["RA"]
    ).item()

    # P5-2 / RP2-2 -> February
    rp2_2 = spatial_mean(
        february,
        masks["RP2-2"]
    ).item()

    ds.close()

    print(
        f"RP1 (P4-2), January = "
        f"{rp1:.4f} K"
    )

    print(
        f"RA (A4-2), February = "
        f"{ra:.4f} K"
    )

    print(
        f"RP2-2 (P5-2), February = "
        f"{rp2_2:.4f} K"
    )

    return {
        "year": year,
        "RP1": rp1,
        "RA": ra,
        "RP2-2": rp2_2,
    }


# ============================================================
# PROCESS ALL YEARS
# ============================================================

def build_sst_dataset():

    results = []

    for year in range(
        FIRST_YEAR,
        LAST_YEAR + 1
    ):

        try:

            result = process_year(
                year
            )

            results.append(
                result
            )

        except Exception as error:

            print("\nERROR:")
            print(
                f"Year: {year}"
            )
            print(error)

            print(
                "\nContinuing..."
            )

    if not results:

        raise RuntimeError(
            "No years were processed successfully."
        )

    df = pd.DataFrame(
        results
    )

    df = df.set_index(
        "year"
    )

    return df


# ============================================================
# CALCULATE CLIMATOLOGY
# ============================================================

def calculate_climatology(df):

    print("\n" + "=" * 70)
    print("CALCULATING SST CLIMATOLOGY")
    print("=" * 70)

    print(
        f"Reference period: "
        f"{CLIM_START}–{CLIM_END}"
    )

    expected_years = set(
        range(
            CLIM_START,
            CLIM_END + 1
        )
    )

    available_years = set(
        df.index
    )

    missing_years = sorted(
        expected_years - available_years
    )

    if missing_years:

        raise RuntimeError(
            "Missing years in climatology period: "
            + ", ".join(
                map(
                    str,
                    missing_years
                )
            )
        )

    climatology = (
        df
        .loc[
            CLIM_START:CLIM_END
        ]
        .mean()
    )

    print("\nClimatology:")

    for region in climatology.index:

        print(
            f"{region}: "
            f"{climatology[region]:.4f} K"
        )

    return climatology


# ============================================================
# CALCULATE ANOMALIES
# ============================================================

def calculate_anomalies(
    df,
    climatology
):

    print("\n" + "=" * 70)
    print("CALCULATING SST ANOMALIES")
    print("=" * 70)

    return (
        df
        - climatology
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    absolute_sst,
    anomalies
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    absolute_sst.to_csv(
        OUTPUT_ABSOLUTE_FILE
    )

    anomalies.to_csv(
        OUTPUT_ANOMALIES_FILE
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"\nAbsolute SST:\n"
        f"{OUTPUT_ABSOLUTE_FILE}"
    )

    print(
        f"\nSST anomalies:\n"
        f"{OUTPUT_ANOMALIES_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("ERA5 SST REGIONAL MEANS AND ANOMALIES")
    print("=" * 70)

    print(
        f"Years: "
        f"{FIRST_YEAR}–{LAST_YEAR}"
    )

    print(
        f"Climatology: "
        f"{CLIM_START}–{CLIM_END}"
    )

    print("\nRegions:")
    print("RP1   = P4-2   → January")
    print("RA    = A4-2   → February")
    print("RP2-2 = P5-2   → February")

    print("\nSpatial procedure:")
    print(
        "0.25° ERA5"
        " → 0.5° arithmetic mean"
        " → rectangular masks"
        " → simple regional mean"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 1. Absolute SST
    # --------------------------------------------------------

    absolute_sst = (
        build_sst_dataset()
    )

    # --------------------------------------------------------
    # 2. Climatology 1982–2024
    # --------------------------------------------------------

    climatology = (
        calculate_climatology(
            absolute_sst
        )
    )

    # --------------------------------------------------------
    # 3. Anomalies
    # --------------------------------------------------------

    anomalies = (
        calculate_anomalies(
            absolute_sst,
            climatology
        )
    )

    # --------------------------------------------------------
    # 4. Save
    # --------------------------------------------------------

    save_csv(
        absolute_sst,
        anomalies
    )

    # --------------------------------------------------------
    # 5. Display
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ABSOLUTE SST")
    print("=" * 70)

    print(
        absolute_sst
    )

    print("\n" + "=" * 70)
    print("SST ANOMALIES")
    print("=" * 70)

    print(
        anomalies
    )

    print("\nCompleted.")
