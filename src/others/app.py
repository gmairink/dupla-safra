import streamlit as st
import xarray as xr

# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Wet-Season Demise",
    layout="wide"
)

# ============================================================
# CAMINHO DOS DADOS
# ============================================================

forecast_file = (
    "/home/gustavo/dupla_safra_view_web/"
    "data/processed/forecast/"
    "demise_forecast_1982-2025.nc"
)

# ============================================================
# ABRIR DATASET
# ============================================================

ds = xr.open_dataset(forecast_file)

# ============================================================
# INTERFACE
# ============================================================

st.title("Wet-Season Demise")

st.write(
    "Operational visualization of observed, predicted "
    "and forecast error."
)

st.write("Dataset:")

st.write(ds)

# ============================================================
# SELEÇÃO DO ANO
# ============================================================

years = ds.time.values.astype(int)

year = st.selectbox(
    "Year",
    years
)

st.write(f"Selected year: {year}")