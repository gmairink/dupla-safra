from pathlib import Path

import numpy as np
import xarray as xr


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
    / "processed"
    / "chirps"
)

OUTPUT_BASE_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "anomalous_accumulation"
)

FIRST_YEAR = 1981
LAST_YEAR = 2025

VAR_NAME = "precip"

# Reference precipitation
BASE_VALUE = 2.5


# ============================================================
# ANOMALOUS ACCUMULATION
# ============================================================

def calculate_anomalous_accumulation(
    precip,
    base_value=BASE_VALUE
):
    """
    Calculate wet-season onset, demise and length
    using the Anomalous Accumulation method.

    Parameters
    ----------
    precip : xarray.DataArray
        Daily precipitation with dimensions:
        time, latitude, longitude.

    base_value : float
        Reference precipitation in mm/day.

    Returns
    -------
    xarray.Dataset
        onset, demise and length.
    """

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    data = precip.values.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Valid pixels
    # --------------------------------------------------------

    valid_pixels = ~np.isnan(data).all(
        axis=0
    )

    # --------------------------------------------------------
    # Daily anomaly
    # --------------------------------------------------------

    daily_anomaly = (
        data - base_value
    )

    # --------------------------------------------------------
    # Anomalous accumulation
    # --------------------------------------------------------

    aa = np.cumsum(
        daily_anomaly,
        axis=0
    )

    # --------------------------------------------------------
    # Mean rainfall
    # --------------------------------------------------------

    mean_rainfall = np.mean(
        data,
        axis=0
    )

    # --------------------------------------------------------
    # Maximum daily anomalous accumulation
    # --------------------------------------------------------

    max_daily_anomaly = np.full(
        mean_rainfall.shape,
        np.nan,
        dtype=np.float32
    )

    max_daily_day = np.full(
        mean_rainfall.shape,
        -1,
        dtype=np.int32
    )

    max_daily_anomaly[valid_pixels] = np.max(
        daily_anomaly[:, valid_pixels],
        axis=0
    )

    max_daily_day[valid_pixels] = np.argmax(
        daily_anomaly[:, valid_pixels],
        axis=0
    )






    # --------------------------------------------------------
    # Output arrays
    # --------------------------------------------------------

    onset = np.full(
        mean_rainfall.shape,
        np.nan,
        dtype=np.float32
    )

    demise = np.full(
        mean_rainfall.shape,
        np.nan,
        dtype=np.float32
    )

    length = np.full(
        mean_rainfall.shape,
        np.nan,
        dtype=np.float32
    )

    # ========================================================
    # PIXELS WITH MEAN RAINFALL < BASE VALUE
    # ========================================================

    dry_mean = (
        valid_pixels
        & (mean_rainfall < base_value)
    )

    # --------------------------------------------------------
    # For these pixels:
    #
    # MDAA = maximum daily anomaly
    # MDAA day = day of maximum anomaly
    #
    # Then:
    #
    # onset = minimum AA before MDAA
    # demise = maximum AA after MDAA
    # --------------------------------------------------------

    rows, cols = np.where(
        dry_mean
    )

    for i, j in zip(rows, cols):

        peak_day = int(
            max_daily_day[i, j]
        )

        # AA before maximum daily anomaly
        aa_before = aa[
            :peak_day + 1,
            i,
            j
        ]

        # AA after maximum daily anomaly
        aa_after = aa[
            peak_day:,
            i,
            j
        ]

        if aa_before.size > 0:

            onset[i, j] = (
                np.nanargmin(
                    aa_before
                )
            )

        if aa_after.size > 0:

            demise[i, j] = (
                peak_day
                + np.nanargmax(
                    aa_after
                )
            )

    # ========================================================
    # PIXELS WITH MEAN RAINFALL >= BASE VALUE
    # ========================================================

    wet_mean = (
        valid_pixels
        & (mean_rainfall >= base_value)
    )

    rows, cols = np.where(
        wet_mean
    )

    for i, j in zip(rows, cols):

        aa_pixel = aa[
            :,
            i,
            j
        ]

        onset[i, j] = (
            np.nanargmin(
                aa_pixel
            )
        )

        demise[i, j] = (
            np.nanargmax(
                aa_pixel
            )
        )

    # --------------------------------------------------------
    # Wet-season length
    # --------------------------------------------------------

    length = (
        demise - onset
    )

    # --------------------------------------------------------
    # Convert to xarray
    # --------------------------------------------------------

    coords = {
        "latitude": precip.latitude,
        "longitude": precip.longitude
    }

    output = xr.Dataset(
        {
            "onset": (
                ["latitude", "longitude"],
                onset
            ),

            "demise": (
                ["latitude", "longitude"],
                demise
            ),

            "length": (
                ["latitude", "longitude"],
                length
            ),
        },
        coords=coords
    )

    return output


# ============================================================
# PROCESS ONE YEAR
# ============================================================

def process_year(year):

    next_year = year + 1

    hydro_year = (
        f"{year}_{next_year}"
    )

    print("\n" + "=" * 70)
    print(
        f"PROCESSING {year}/{next_year}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_file = (
        INPUT_BASE_DIR
        / hydro_year
        / f"chirps_v2_{hydro_year}_05deg.nc"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_BASE_DIR
        / hydro_year
    )

    output_file = (
        output_dir
        / f"anomalous_accumulation_{hydro_year}.nc"
    )

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not input_file.exists():

        raise FileNotFoundError(
            f"Input not found:\n{input_file}"
        )

    # --------------------------------------------------------
    # Skip existing
    # --------------------------------------------------------

    if output_file.exists():

        print(
            "Output already exists:"
        )

        print(output_file)

        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Open
    # --------------------------------------------------------

    print(
        "\nOpening CHIRPS..."
    )

    ds = xr.open_dataset(
        input_file
    )

    precip = ds[
        VAR_NAME
    ]

    print(
        precip
    )

    # --------------------------------------------------------
    # Calculate
    # --------------------------------------------------------

    print(
        "\nCalculating anomalous accumulation..."
    )

    output = calculate_anomalous_accumulation(
        precip
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    output["onset"].attrs = {
        "units": "day",
        "long_name": (
            "Wet-season onset"
        ),
        "description": (
            "Day after July 1 corresponding "
            "to minimum anomalous accumulation"
        ),
        "reference_value": (
            f"{BASE_VALUE} mm/day"
        ),
    }

    output["demise"].attrs = {
        "units": "day",
        "long_name": (
            "Wet-season demise"
        ),
        "description": (
            "Day after July 1 corresponding "
            "to maximum anomalous accumulation"
        ),
        "reference_value": (
            f"{BASE_VALUE} mm/day"
        ),
    }

    output["length"].attrs = {
        "units": "days",
        "long_name": (
            "Wet-season length"
        ),
        "description": (
            "Difference between demise and onset"
        ),
    }

    output.attrs = {
        "method": (
            "Anomalous Accumulation"
        ),
        "hydrological_year": (
            f"{year}/{next_year}"
        ),
        "reference_value": (
            f"{BASE_VALUE} mm/day"
        ),
        "input_resolution": "0.5°",
        "precipitation_product": (
            "CHIRPS v2"
        ),
    }

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

    ds.close()

    print(
        "\nCompleted."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ANOMALOUS ACCUMULATION"
    )

    print(
        f"{FIRST_YEAR}/{FIRST_YEAR + 1}"
        f" → "
        f"{LAST_YEAR}/{LAST_YEAR + 1}"
    )

    print(
        f"Reference value: "
        f"{BASE_VALUE} mm/day"
    )

    print(
        "=" * 70
    )

    for year in range(
        FIRST_YEAR,
        LAST_YEAR + 1
    ):

        try:

            process_year(
                year
            )

        except Exception as error:

            print(
                "\nERROR:"
            )

            print(
                f"{year}/{year + 1}"
            )

            print(
                error
            )

            print(
                "\nContinuing..."
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ALL YEARS PROCESSED."
    )

    print(
        "=" * 70
    )
    