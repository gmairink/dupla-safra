from pathlib import Path

import xarray as xr
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = (
    Path.home()
    / "dupla_safra_view_web"
)

INPUT_BASE_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "chirps"
)

OUTPUT_BASE_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "chirps"
)

FIRST_YEAR = 1981
LAST_YEAR = 2025

# 0.05° -> 0.5°
# 10 pixels in latitude x 10 pixels in longitude
COARSEN_FACTOR = 10


# ============================================================
# CHECK RESOLUTION
# ============================================================

def check_resolution(ds):

    lat_res = abs(
        float(
            ds.latitude.diff("latitude").mean()
        )
    )

    lon_res = abs(
        float(
            ds.longitude.diff("longitude").mean()
        )
    )

    print("\nOriginal resolution:")
    print(f"Latitude : {lat_res:.5f}°")
    print(f"Longitude: {lon_res:.5f}°")

    if not np.isclose(
        lat_res,
        0.05,
        atol=1e-4
    ):
        raise ValueError(
            f"Unexpected latitude resolution: {lat_res}"
        )

    if not np.isclose(
        lon_res,
        0.05,
        atol=1e-4
    ):
        raise ValueError(
            f"Unexpected longitude resolution: {lon_res}"
        )


# ============================================================
# PROCESS ONE HYDROLOGICAL YEAR
# ============================================================

def resample_file(year):

    next_year = year + 1

    print("\n" + "=" * 70)
    print(
        f"PROCESSING {year}/{next_year}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    input_file = (
        INPUT_BASE_DIR
        / f"{year}_{next_year}"
        / f"chirps_v2_{year}_{next_year}.nc"
    )

    output_dir = (
        OUTPUT_BASE_DIR
        / f"{year}_{next_year}"
    )

    output_file = (
        output_dir
        / f"chirps_v2_{year}_{next_year}_05deg.nc"
    )

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not input_file.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n{input_file}"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Open dataset
    # --------------------------------------------------------

    print("\nOpening CHIRPS dataset...")

    ds = xr.open_dataset(
        input_file
    )

    print(ds)

    # --------------------------------------------------------
    # Check original resolution
    # --------------------------------------------------------

    check_resolution(ds)

    # --------------------------------------------------------
    # Sort coordinates
    # --------------------------------------------------------

    if not np.all(
        np.diff(ds.latitude.values) > 0
    ):

        print(
            "Sorting latitude..."
        )

        ds = ds.sortby(
            "latitude"
        )

    if not np.all(
        np.diff(ds.longitude.values) > 0
    ):

        print(
            "Sorting longitude..."
        )

        ds = ds.sortby(
            "longitude"
        )

    # --------------------------------------------------------
    # Select precipitation
    # --------------------------------------------------------

    if "precip" not in ds:

        raise ValueError(
            "Variable 'precip' not found."
        )

    precip = ds["precip"]

    # --------------------------------------------------------
    # Aggregate 0.05° -> 0.5°
    # --------------------------------------------------------

    print("\nAggregating:")
    print(
        "0.05° × 0.05°"
        " → "
        "0.5° × 0.5°"
    )

    print(
        f"Mean of "
        f"{COARSEN_FACTOR} × {COARSEN_FACTOR} "
        f"pixels"
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # The input files are ALREADY cropped by the download
    # script. No polygon, mask, or additional spatial crop
    # is applied here.
    #
    # coarsen(10, 10) simply groups each consecutive block
    # of 10 × 10 original 0.05° pixels and calculates their
    # arithmetic mean.
    # --------------------------------------------------------

    precip = precip.sortby(
        "latitude"
    )

    precip = precip.sortby(
        "longitude"
    )

    precip_05deg = (
        precip
        .coarsen(
            latitude=COARSEN_FACTOR,
            longitude=COARSEN_FACTOR,
            boundary="trim"
        )
        .mean(
            skipna=True
        )
    )

    # --------------------------------------------------------
    # Create output dataset
    # --------------------------------------------------------

    output = precip_05deg.to_dataset(
        name="precip"
    )

    # --------------------------------------------------------
    # Check resulting grid
    # --------------------------------------------------------

    lat_res = abs(
        float(
            output.latitude.diff(
                "latitude"
            ).mean()
        )
    )

    lon_res = abs(
        float(
            output.longitude.diff(
                "longitude"
            ).mean()
        )
    )

    print("\nAggregated grid:")
    print(
        f"Latitude resolution : {lat_res:.5f}°"
    )

    print(
        f"Longitude resolution: {lon_res:.5f}°"
    )

    print(
        f"Latitude range : "
        f"{float(output.latitude.min()):.3f} "
        f"→ "
        f"{float(output.latitude.max()):.3f}"
    )

    print(
        f"Longitude range: "
        f"{float(output.longitude.min()):.3f} "
        f"→ "
        f"{float(output.longitude.max()):.3f}"
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    n_lat_input = ds.sizes["latitude"]
    n_lon_input = ds.sizes["longitude"]

    n_lat_output = output.sizes["latitude"]
    n_lon_output = output.sizes["longitude"]

    print(
        "\nGrid dimensions:"
    )

    print(
        f"Input : "
        f"{n_lat_input} × {n_lon_input}"
    )

    print(
        f"Output: "
        f"{n_lat_output} × {n_lon_output}"
    )

    print(
        f"\nPixels averaged per output cell: "
        f"{COARSEN_FACTOR} × {COARSEN_FACTOR}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    output.attrs.update(
        {
            "processing": (
                "CHIRPS v2 daily precipitation "
                "aggregated from 0.05° to 0.5° "
                "using arithmetic mean."
            ),
            "input_resolution": "0.05°",
            "output_resolution": "0.5°",
            "aggregation": (
                "Arithmetic mean of consecutive "
                "10 x 10 CHIRPS pixels."
            ),
            "boundary": (
                "trim; incomplete edge blocks are removed."
            ),
            "spatial_crop": (
                "No additional crop or polygon mask "
                "applied; input file is assumed to "
                "already contain the desired spatial domain."
            ),
            "pixel_reference": (
                "Output coordinates are the mean coordinates "
                "of each 10 x 10 input block."
            ),
            "hydrological_year": (
                f"{year}/{next_year}"
            ),
            "precipitation_product": (
                "CHIRPS v2"
            ),
        }
    )

    output["precip"].attrs[
        "aggregation"
    ] = (
        "Arithmetic mean of consecutive "
        "10 x 10 CHIRPS pixels"
    )

    output["precip"].attrs[
        "spatial_crop"
    ] = (
        "No additional crop; input file already cropped."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\nSaving:"
    )

    print(
        output_file
    )

    output.to_netcdf(
        output_file
    )

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    ds.close()
    output.close()

    print(
        "\nCompleted successfully."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("CHIRPS RESAMPLING")
    print("0.05° → 0.5°")
    print("NO ADDITIONAL SPATIAL CROP")
    print(
        f"{FIRST_YEAR}/{FIRST_YEAR + 1}"
        f" → "
        f"{LAST_YEAR}/{LAST_YEAR + 1}"
    )
    print("=" * 70)

    for year in range(
        FIRST_YEAR,
        LAST_YEAR + 1
    ):

        try:

            resample_file(year)

        except Exception as error:

            print("\n" + "!" * 70)

            print(
                f"ERROR processing "
                f"{year}/{year + 1}"
            )

            print(
                error
            )

            print(
                "Continuing with next year..."
            )

            print("!" * 70)

    print("\n" + "=" * 70)
    print("ALL CHIRPS FILES PROCESSED.")
    print("=" * 70)
