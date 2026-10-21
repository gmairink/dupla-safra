#%%
#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# CAMINHOS
# ============================================================

base_dir = "/home/gustavo/dupla_safra_view_web/data/processed"

# ------------------------------------------------------------
# SST
# ------------------------------------------------------------

sst_file = os.path.join(
    base_dir,
    "sst",
    "sst_anomalies_1982_2026.csv"
)

# ------------------------------------------------------------
# DEMISE OBSERVADO
# ------------------------------------------------------------

demise_obs_file = os.path.join(
    base_dir,
    "anomalous_accumulation",
    "demise_anomalous_accumulation_1981-2025.nc"
)

# ------------------------------------------------------------
# ARQUIVO USADO PARA DEFINIR A MÁSCARA DE NaN
# ------------------------------------------------------------

mask_obs_file = os.path.join(
    base_dir,
    "anomalous_accumulation",
    "1981_1982",
    "anomalous_accumulation_1981_1982.nc"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

output_file = os.path.join(
    base_dir,
    "forecast",
    "demise_regression_1981-2025.nc"
)


# ============================================================
# COEFICIENTES DA REGRESSÃO
# ============================================================

CONSTANT = 277.697

COEF_LAT = 4.308
COEF_LON = -1.377

COEF_RA = -3.868
COEF_RP1 = -2.720
COEF_RP2_2 = 1.927


# ============================================================
# LEITURA DO ARQUIVO DE SST
# ============================================================

sst = pd.read_csv(sst_file)

print("============================================")
print("ARQUIVO SST")
print("============================================")

print("Colunas:")
print(sst.columns.tolist())

print("\nPrimeiras linhas:")
print(sst.head())

# Garantir que year seja inteiro
sst["year"] = sst["year"].astype(int)


# ============================================================
# LEITURA DA GRADE DO DEMISE OBSERVADO
# ============================================================

ds_obs = xr.open_dataset(demise_obs_file)

print("\n============================================")
print("DATASET OBSERVADO")
print("============================================")

print(ds_obs)

# Coordenadas
lat = ds_obs["latitude"]
lon = ds_obs["longitude"]


# ============================================================
# LEITURA DO ARQUIVO USADO PARA A MÁSCARA
# ============================================================

ds_mask = xr.open_dataset(mask_obs_file)

print("\n============================================")
print("DATASET DA MÁSCARA")
print("============================================")

print(ds_mask)

print("\nVariáveis:")
print(list(ds_mask.data_vars))


# ============================================================
# IDENTIFICAR VARIÁVEL DA MÁSCARA
# ============================================================

# Usa a primeira variável de dados do arquivo
mask_var = list(ds_mask.data_vars)[0]

mask_data = ds_mask[mask_var]

print("\nVariável utilizada para a máscara:")
print(mask_var)

print("\nDimensões originais da máscara:")
print(mask_data.dims)

print("\nShape original:")
print(mask_data.shape)


# ============================================================
# REMOVER DIMENSÃO TIME, SE EXISTIR
# ============================================================

if "time" in mask_data.dims:

    print("\nDimensão 'time' encontrada.")

    # Usar o primeiro timestep
    mask_data = mask_data.isel(time=0)

    print("Usando o primeiro timestep para construir a máscara.")

print("\nDimensões da máscara após tratamento:")
print(mask_data.dims)

print("Shape da máscara:")
print(mask_data.shape)


# ============================================================
# GARANTIR QUE A MÁSCARA TENHA AS MESMAS DIMENSÕES ESPACIAIS
# ============================================================

# Se necessário, selecionar somente latitude/longitude
expected_dims = {"latitude", "longitude"}

if not expected_dims.issubset(set(mask_data.dims)):

    raise ValueError(
        "\nA máscara não possui as dimensões esperadas "
        "'latitude' e 'longitude'.\n"
        f"Dimensões encontradas: {mask_data.dims}"
    )


# ============================================================
# CRIAR MÁSCARA DE VALIDADE
# ============================================================

# True  -> valor válido
# False -> NaN no arquivo de referência

valid_mask = mask_data.notnull()

print("\n============================================")
print("MÁSCARA")
print("============================================")

print(
    "Número de células válidas:",
    int(valid_mask.sum())
)

print(
    "Número de células NaN:",
    int((~valid_mask).sum())
)


# ============================================================
# MATRIZ ESPACIAL DE LATITUDE E LONGITUDE
# ============================================================

# latitude:  (latitude)
# longitude: (longitude)
#
# resultado:
# lat_2d: (latitude, longitude)
# lon_2d: (latitude, longitude)

lat_2d, lon_2d = xr.broadcast(lat, lon)


# ============================================================
# ANOS DISPONÍVEIS E ALINHAMENTO TEMPORAL
# ============================================================

# O arquivo de SST usa o ano de encerramento da estação.
# Exemplo: SST de 1982 -> estação 1981/1982, armazenada em time=1981.
# Assim, SST de 2026 -> estação 2025/2026, armazenada em time=2025.
years_obs = ds_obs["time"].values.astype(int)
years_sst = np.sort(sst["year"].astype(int).unique())

# Não fazer a interseção com os anos observados: a última previsão
# pode ser calculada mesmo quando a observação dessa estação ainda não existe.
season_years = years_sst - 1

print("\n============================================")
print("ANOS E ALINHAMENTO TEMPORAL")
print("============================================")

print("Anos disponíveis no arquivo observado:")
print(years_obs)

print("\nAnos disponíveis no arquivo SST:")
print(years_sst)

print("\nAnos da coordenada time para as previsões (ano inicial da estação):")
print(season_years)

print(f"\nNúmero de previsões: {len(years_sst)}")
print(
    f"Primeira estação: {years_sst[0] - 1}/{years_sst[0]} | "
    f"Última estação: {years_sst[-1] - 1}/{years_sst[-1]}"
)

# Conferir se cada ano de SST aparece uma única vez.
if sst["year"].duplicated().any():
    duplicated_years = sst.loc[sst["year"].duplicated(keep=False), "year"].tolist()
    raise ValueError(f"Há anos duplicados no arquivo SST: {duplicated_years}")

# Conferir que as variáveis usadas na regressão estão presentes.
required_sst_columns = {"year", "RP1", "RA", "RP2-2"}
missing_sst_columns = required_sst_columns.difference(sst.columns)
if missing_sst_columns:
    raise ValueError(
        f"Colunas ausentes no arquivo SST: {sorted(missing_sst_columns)}"
    )


# ============================================================
# PREPARAR LISTA PARA OS DEMISES PREVISTOS
# ============================================================

demise_pred_list = []


# ============================================================
# LOOP POR ANO
# ============================================================

for sst_year in years_sst:

    # A coordenada time representa o ano inicial da estação.
    # O ano da SST é o ano seguinte: 1982 -> time=1981, ..., 2026 -> time=2025.
    season_year = int(sst_year) - 1

    print("\n--------------------------------------------")
    print(
        f"Calculando a estação {season_year}/{sst_year} "
        f"com SST de {sst_year}..."
    )
    print("--------------------------------------------")

    # --------------------------------------------------------
    # SST do ano de encerramento da estação
    # --------------------------------------------------------

    row = sst.loc[
        sst["year"] == sst_year
    ]

    if len(row) != 1:

        raise ValueError(
            f"Esperava exatamente uma linha de SST para {sst_year}, "
            f"mas encontrei {len(row)}."
        )

    row = row.iloc[0]

    RP1 = row["RP1"]
    RA = row["RA"]
    RP2_2 = row["RP2-2"]

    print(f"RA     = {RA}")
    print(f"RP1    = {RP1}")
    print(f"RP2-2  = {RP2_2}")


    # --------------------------------------------------------
    # REGRESSÃO
    # --------------------------------------------------------

    demise_pred = (
        CONSTANT
        + COEF_LAT * lat_2d
        + COEF_LON * lon_2d
        + COEF_RA * RA
        + COEF_RP1 * RP1
        + COEF_RP2_2 * RP2_2
    )


    # --------------------------------------------------------
    # APLICAR MÁSCARA
    # --------------------------------------------------------

    # Onde o arquivo de referência é NaN,
    # o forecast também será NaN.

    demise_pred = demise_pred.where(
        valid_mask
    )


    # --------------------------------------------------------
    # TRANSFORMAR EM DATAARRAY COM TIME
    # --------------------------------------------------------

    demise_pred = demise_pred.expand_dims(
        time=[season_year]
    )

    # Guardar também o ano da SST usado, para tornar o deslocamento explícito.
    demise_pred = demise_pred.assign_coords(
        sst_year=("time", [int(sst_year)])
    )

    demise_pred.name = "demise"


    # --------------------------------------------------------
    # ATRIBUTOS
    # --------------------------------------------------------

    demise_pred.attrs = {

        "long_name":
            "Predicted wet-season demise",

        "description": (
            "Wet-season demise predicted using the linear "
            "regression based on latitude, longitude and "
            "SST anomalies in RA, RP1 and RP2-2. The time "
            "coordinate is the starting year of the season; "
            "SST predictors come from the following year. "
            "NaNs were imposed using the spatial mask from "
            "anomalous_accumulation_1981_1982.nc."
        ),

        "units":
            "day of year",

        "regression_constant":
            CONSTANT,

        "coefficient_latitude":
            COEF_LAT,

        "coefficient_longitude":
            COEF_LON,

        "coefficient_RA":
            COEF_RA,

        "coefficient_RP1":
            COEF_RP1,

        "coefficient_RP2-2":
            COEF_RP2_2,

        "mask_source":
            "anomalous_accumulation_1981_1982.nc",

        "mask_rule":
            "Forecast is NaN where the reference anomalous accumulation is NaN."
    }


    # --------------------------------------------------------
    # ADICIONAR À LISTA
    # --------------------------------------------------------

    demise_pred_list.append(
        demise_pred
    )


# ============================================================
# CONCATENAR TODOS OS ANOS
# ============================================================

demise_pred = xr.concat(
    demise_pred_list,
    dim="time"
)


# ============================================================
# ARREDONDAR DEMISE PREVISTO PARA DIAS INTEIROS
# ============================================================

demise_pred = demise_pred.round(0)

demise_pred.attrs["rounding"] = (
    "Rounded to the nearest whole day"
)

print("\nDemise previsto arredondado para dias inteiros.")


# ============================================================
# ORGANIZAR DATASET FINAL
# ============================================================

ds_pred = xr.Dataset(
    {
        "demise": demise_pred
    }
)


# ============================================================
# ATRIBUTOS DO DATASET
# ============================================================

ds_pred.attrs = {

    "description": (
        "Wet-season demise predicted from a linear regression "
        "using latitude, longitude and annual SST anomalies. "
        "The time coordinate gives the starting year of each season; "
        "SST predictors are taken from the following calendar year."
    ),

    "period":
        f"{season_years.min()}-{season_years.max()}",

    "season_period": (
        f"{season_years.min()}/{season_years.min() + 1}-"
        f"{season_years.max()}/{season_years.max() + 1}"
    ),

    "temporal_alignment": (
        "time = SST year - 1; e.g. SST 1982 predicts season 1981/1982 "
        "(time=1981), and SST 2026 predicts season 2025/2026 (time=2025)."
    ),

    "source_sst":
        "sst_anomalies_1982_2026.csv",

    "source_observed_demise":
        "demise_anomalous_accumulation_1981-2025.nc",

    "mask_source":
        "1981_1982/anomalous_accumulation_1981_1982.nc",

    "mask_rule":
        "Forecast set to NaN where the reference observed "
        "anomalous accumulation is NaN.",

    "regression":
        (
            "demise = 277.697 "
            "+ 4.308*latitude "
            "- 1.377*longitude "
            "- 3.868*RA "
            "- 2.720*RP1 "
            "+ 1.927*RP2-2"
        )
}


# ============================================================
# CRIAR DIRETÓRIO DE OUTPUT
# ============================================================

os.makedirs(
    os.path.dirname(output_file),
    exist_ok=True
)


# ============================================================
# SALVAR
# ============================================================

ds_pred.to_netcdf(
    output_file,
    format="NETCDF4"
)


# ============================================================
# RESULTADOS
# ============================================================

print("\n============================================")
print("ARQUIVO SALVO")
print("============================================")

print(output_file)


print("\n============================================")
print("DATASET FINAL")
print("============================================")

print(ds_pred)


print("\n============================================")
print("ESTATÍSTICAS")
print("============================================")

print(
    "Mínimo:",
    ds_pred["demise"].min().item()
)

print(
    "Máximo:",
    ds_pred["demise"].max().item()
)

print(
    "Número total de células:",
    ds_pred["demise"].size
)

print(
    "Número de NaNs:",
    int(ds_pred["demise"].isnull().sum())
)

print(
    "Número de valores válidos:",
    int(ds_pred["demise"].notnull().sum())
)

print("\n============================================")
print("FINALIZADO")
print("============================================")