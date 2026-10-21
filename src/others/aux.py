#%%
import os
import xarray as xr


# ============================================================
# CAMINHOS
# ============================================================

base_dir = "/home/gustavo/dupla_safra_view_web/data/processed/forecast"

input_file = os.path.join(
    base_dir,
    "demise_forecast_1982-2025.nc"
)

output_file = os.path.join(
    base_dir,
    "demise_absolute_error_mean_selected_years.nc"
)


# ============================================================
# ANOS SELECIONADOS
# ============================================================

years_selected = [
    1983, 1986, 1987, 1988,
    1991, 1993, 1995, 1997,
    1998, 2000, 2007, 2008,
    2009, 2010, 2012, 2013,
    2015, 2016, 2018, 2020,
    2021, 2022
]


# ============================================================
# ABRIR DATASET
# ============================================================

ds = xr.open_dataset(input_file)

print("Dataset original:")
print(ds)


# ============================================================
# SELECIONAR OS ANOS
# ============================================================

ds_selected = ds.sel(
    time=years_selected
)

print("\nAnos selecionados:")
print(ds_selected.time.values)


# ============================================================
# MÉDIA TEMPORAL DO ERRO ABSOLUTO
# ============================================================
#
# A média é calculada SOMENTE ao longo de "time".
#
# latitude e longitude permanecem.
#
# Resultado:
#
#     latitude × longitude
#
# Cada pixel terá seu próprio MAE calculado
# considerando os 22 anos selecionados.
#

mae = ds_selected["absolute_error"].mean(
    dim="time",
    skipna=True
)

mae.name = "mean_absolute_error"

mae.attrs = {
    "long_name": "Mean absolute error of wet-season demise",
    "description": (
        "Mean absolute error calculated for each grid cell "
        "over the selected years."
    ),
    "units": "day",
    "formula": (
        "MAE = mean(abs(demise_predicted - demise_observed)) "
        "over selected years"
    ),
}


# ============================================================
# CRIAR DATASET FINAL
# ============================================================

ds_mae = xr.Dataset(
    {
        "mean_absolute_error": mae
    }
)


# ============================================================
# ATRIBUTOS
# ============================================================

ds_mae.attrs = {
    "description": (
        "Pixel-wise mean absolute error of regression-predicted "
        "wet-season demise."
    ),
    "selected_years": ", ".join(
        str(year) for year in years_selected
    ),
    "number_of_years": len(years_selected),
    "source": "demise_forecast_1982-2025.nc",
}


# ============================================================
# SALVAR
# ============================================================

ds_mae.to_netcdf(
    output_file,
    format="NETCDF4"
)


# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "=" * 60)
print("ARQUIVO SALVO")
print("=" * 60)

print(f"\n{output_file}")

print("\nDataset final:")
print(ds_mae)

print("\nDimensões:")
print(ds_mae.dims)

print("\nEstatísticas:")
print(
    f"MAE mínimo: "
    f"{float(mae.min(skipna=True)):.2f} dias"
)

print(
    f"MAE máximo: "
    f"{float(mae.max(skipna=True)):.2f} dias"
)

print(
    f"MAE médio espacial: "
    f"{float(mae.mean(skipna=True)):.2f} dias"
)


# ============================================================
# FECHAR
# ============================================================

ds.close()