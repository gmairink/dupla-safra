import os
import cdsapi


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = "/home/gustavo/dupla_safra_view_web"

RAW_BASE_DIR = (
    f"{PROJECT_DIR}/data/raw/era5"
)

FIRST_YEAR = 1982
LAST_YEAR = 2026


# ============================================================
# DOWNLOAD ERA5 SST
# ============================================================

def download_era5(year):

    year_dir = os.path.join(
        RAW_BASE_DIR,
        str(year)
    )

    os.makedirs(
        year_dir,
        exist_ok=True
    )

    raw_file = os.path.join(
        year_dir,
        f"era5_sst_{year}.nc"
    )

    # --------------------------------------------------------
    # Skip if already downloaded
    # --------------------------------------------------------

    if os.path.exists(raw_file):

        print("\nERA5 file already exists:")
        print(raw_file)

        return

    print("\n" + "=" * 70)
    print(f"DOWNLOADING ERA5 SST: {year}")
    print("=" * 70)

    print("Months: January and February")
    print("Variable: Sea surface temperature")
    print("Resolution: 0.25°")

    client = cdsapi.Client()

    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": [
            "sea_surface_temperature"
        ],
        "year": [
            str(year)
        ],
        "month": [
            "01",
            "02"
        ],
        "time": [
            "00:00"
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        request,
        raw_file,
    )

    print("\nDownload completed:")
    print(raw_file)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("ERA5 SST DOWNLOAD")
    print("=" * 70)

    print(
        f"Period: {FIRST_YEAR}–{LAST_YEAR}"
    )

    print(
        "Months: January + February"
    )

    print("=" * 70)

    for year in range(
        FIRST_YEAR,
        LAST_YEAR + 1
    ):

        try:

            download_era5(
                year
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

    print("\n" + "=" * 70)
    print("ALL YEARS PROCESSED")
    print("=" * 70)