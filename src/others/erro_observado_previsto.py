#!/usr/bin/env python3

import os
import numpy as np
import xarray as xr


# ============================================================
# CAMINHOS
# ============================================================

base_dir = "/home/gustavo/dupla_safra_view_web/data/processed"

observed_file = os.path.join(
    base_dir,
    "anomalous_accumulation",
    "demise_anomalous_accumulation_1981-2025.nc",
)

predicted_file = os.path.join(
    base_dir,
    "forecast",
    "demise_regression_1981-2025.nc",
)

forecast_dir = os.path.join(base_dir, "forecast")

# A coordenada time representa o ano de início da estação (1981–2025),
# mas o produto final será rotulado pelo ano de término (1982–2026).
output_file = os.path.join(
    forecast_dir,
    "demise_forecast_1982-2026.nc",
)


# ============================================================
# CRIAR DIRETÓRIO DE SAÍDA
# ============================================================

os.makedirs(forecast_dir, exist_ok=True)


# ============================================================
# ABRIR OS DATASETS
# ============================================================

print("Abrindo demise observado...")
ds_obs = xr.open_dataset(observed_file)

print("Abrindo demise previsto...")
ds_pred = xr.open_dataset(predicted_file)

try:
    # ========================================================
    # VERIFICAR A VARIÁVEL
    # ========================================================

    if "demise" not in ds_obs:
        raise KeyError(
            "'demise' não encontrado no arquivo observado. "
            f"Variáveis disponíveis: {list(ds_obs.data_vars)}"
        )

    if "demise" not in ds_pred:
        raise KeyError(
            "'demise' não encontrado no arquivo previsto. "
            f"Variáveis disponíveis: {list(ds_pred.data_vars)}"
        )

    # ========================================================
    # PADRONIZAR O TIPO DE TIME
    # ========================================================

    ds_obs = ds_obs.assign_coords(time=ds_obs.time.astype(int))
    ds_pred = ds_pred.assign_coords(time=ds_pred.time.astype(int))

    # ========================================================
    # SELECIONAR OS ANOS EM COMUM
    # ========================================================
    # Neste ponto, time ainda representa o ano de início da estação:
    # 1981 representa 1981/1982; 2025 representa 2025/2026.

    common_years = np.intersect1d(
        ds_obs.time.values,
        ds_pred.time.values,
    )

    if len(common_years) == 0:
        raise ValueError(
            "Não existem anos em comum entre os dois datasets."
        )

    print("\nAnos de início da estação usados para alinhar os dados:")
    print(common_years)

    # O ano exibido no produto final será o ano de término da estação.
    # Ex.: time original 1981 -> rótulo final 1982; 2025 -> 2026.
    display_years = common_years + 1

    print("\nAnos exibidos no arquivo final:")
    print(display_years)

    # ========================================================
    # SELECIONAR DEMISE OBSERVADO E PREVISTO
    # ========================================================

    obs = ds_obs["demise"].sel(time=common_years)
    pred = ds_pred["demise"].sel(time=common_years)

    # ========================================================
    # GARANTIR QUE AS GRADES ESPACIAIS COINCIDAM
    # ========================================================

    obs, pred = xr.align(obs, pred, join="exact")

    # ========================================================
    # CALCULAR O ERRO
    # ========================================================
    # error = previsto - observado
    # error > 0: previsto mais tarde que observado
    # error < 0: previsto mais cedo que observado

    error = pred - obs
    error.name = "error"
    error.attrs = {
        "long_name": "Forecast error in wet-season demise",
        "description": "Predicted demise minus observed demise",
        "units": "day",
        "formula": "error = demise_predicted - demise_observed",
    }

    # ========================================================
    # CALCULAR ERRO ABSOLUTO
    # ========================================================

    absolute_error = np.abs(error)
    absolute_error.name = "absolute_error"
    absolute_error.attrs = {
        "long_name": "Absolute forecast error in wet-season demise",
        "description": "Absolute difference between predicted and observed demise",
        "units": "day",
        "formula": "absolute_error = abs(demise_predicted - demise_observed)",
    }

    # ========================================================
    # RENOMEAR AS VARIÁVEIS
    # ========================================================

    obs.name = "demise_observed"
    pred.name = "demise_predicted"

    obs.attrs = {
        "long_name": "Observed wet-season demise",
        "description": "Demise calculated from Anomalous Accumulation",
        "source": "CHIRPS precipitation",
        "units": "day of year",
    }

    pred.attrs = {
        "long_name": "Predicted wet-season demise",
        "description": "Demise predicted by linear regression",
        "source": "Latitude, longitude and SST anomalies",
        "units": "day of year",
    }

    # ========================================================
    # ATUALIZAR O RÓTULO TEMPORAL PARA O ANO DE TÉRMINO
    # ========================================================
    # A seleção e os cálculos foram feitos usando os anos originais
    # para garantir correspondência entre observado e previsto.
    # Só agora o rótulo time é deslocado em +1 ano.

    obs = obs.assign_coords(time=display_years)
    pred = pred.assign_coords(time=display_years)
    error = error.assign_coords(time=display_years)
    absolute_error = absolute_error.assign_coords(time=display_years)

    # ========================================================
    # CRIAR DATASET FINAL
    # ========================================================

    ds_forecast = xr.Dataset(
        {
            "demise_observed": obs,
            "demise_predicted": pred,
            "error": error,
            "absolute_error": absolute_error,
        }
    )

    # ========================================================
    # ATRIBUTOS DO DATASET
    # ========================================================

    ds_forecast.attrs = {
        "description": (
            "Comparison between observed and regression-predicted "
            "wet-season demise for each wet-season ending year and grid cell."
        ),
        "period": f"{int(display_years.min())}-{int(display_years.max())}",
        "time_definition": (
            "time labels the calendar year in which the wet season ends; "
            "the original season-start year is time - 1."
        ),
        "error_definition": "predicted - observed",
        "observed_source": os.path.basename(observed_file),
        "predicted_source": os.path.basename(predicted_file),
    }

    # ========================================================
    # SALVAR
    # ========================================================

    print("\nSalvando arquivo...")
    ds_forecast.to_netcdf(output_file, format="NETCDF4")

    # ========================================================
    # RELATÓRIO
    # ========================================================

    print("\n" + "=" * 60)
    print("PROCESSAMENTO FINALIZADO")
    print("=" * 60)
    print("\nArquivo salvo em:")
    print(output_file)
    print("\nAnos exibidos:")
    print(ds_forecast.time.values)
    print("\nEstrutura:")
    print(ds_forecast)
    print("\nVariáveis:")
    print(list(ds_forecast.data_vars))
    print("\nEstatísticas do erro:")
    print(f"Erro mínimo: {float(error.min(skipna=True)):.2f} dias")
    print(f"Erro máximo: {float(error.max(skipna=True)):.2f} dias")
    print(f"Erro médio: {float(error.mean(skipna=True)):.2f} dias")
    print(
        "Erro absoluto médio: "
        f"{float(absolute_error.mean(skipna=True)):.2f} dias"
    )

finally:
    ds_obs.close()
    ds_pred.close()
