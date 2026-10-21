import os
import numpy as np
import xarray as xr

base_path = "/home/gustavo/dupla_safra_view_web/data/processed/anomalous_accumulation"

first_year = 1981
last_year = 2025

output_file = os.path.join(
    base_path,
    f"demise_anomalous_accumulation_{first_year}-{last_year}.nc"
)

demise_list = []
years = []

for year in range(first_year, last_year + 1):

    file_path = os.path.join(
        base_path,
        f"{year}_{year + 1}",
        f"anomalous_accumulation_{year}_{year + 1}.nc"
    )

    print(f"Lendo: {file_path}")

    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        continue

    ds = xr.open_dataset(file_path)

    demise = ds["demise"].squeeze(drop=True)

    demise_list.append(demise)
    years.append(year)

    ds.close()

if len(demise_list) == 0:
    raise RuntimeError("Nenhum arquivo de demise foi encontrado.")

demise_all = xr.concat(
    demise_list,
    dim="time",
    join="exact",
    combine_attrs="override"
)

demise_all = demise_all.assign_coords(
    time=np.array(years)
)

demise_all = demise_all.sortby("time")

demise_all.name = "demise"

demise_all.attrs = {
    "long_name": "Wet-season demise",
    "units": "day",
    "description": "Day after July 1 corresponding to maximum anomalous accumulation",
    "reference_value": "2.5 mm/day"
}

ds_output = xr.Dataset(
    {
        "demise": demise_all
    }
)

ds_output.attrs = {
    "description": "Wet-season demise calculated using the Anomalous Accumulation method",
    "period": f"{first_year}-{last_year}",
    "source": "CHIRPS precipitation",
    "base_value": 2.5
}

ds_output.to_netcdf(output_file)

print()
print("Arquivo criado com sucesso:")
print(output_file)
print()
print(ds_output)