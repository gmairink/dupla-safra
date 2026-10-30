from pathlib import Path
from datetime import date, timedelta

import geopandas as gpd

import numpy as np

import pandas as pd

import plotly.graph_objects as go

import streamlit as st
from streamlit_js_eval import streamlit_js_eval

import xarray as xr

# ============================================================

# STREAMLIT CONFIGURATION

# ============================================================

st.set_page_config(

    page_title="Previsão do fim da estação chuvosa no Brasil Central",

    layout="wide",

)

st.markdown(
    """
    <style>
        [data-testid="stAppViewBlockContainer"],
        .block-container {
            padding-top: 0rem !important;
        }

        h1 {
            margin-top: -0.5rem !important;
            padding-top: 0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Estimate the available browser content width on the client. Browser zoom changes
# this CSS width, so the Plotly figure height can track the actual width of the
# three-column layout instead of staying fixed at 440 px. After changing browser
# zoom, reload the page so this value is measured again.
_browser_width = streamlit_js_eval(
    js_expressions="window.innerWidth",
    key="responsive_map_viewport_width",
)
try:
    _content_width = float(_browser_width) if _browser_width is not None else 0.0
except (TypeError, ValueError):
    _content_width = 0.0

if _content_width >= 700:
    _estimated_column_width = max(260.0, (_content_width - 48.0) / 3.0)
    # Map extent has a 22.5° longitude span and a 15° latitude span (ratio 1.5).
    # Add a small allowance for the title and Plotly margins; clamp the height so
    # very narrow/wide windows remain usable.
    MAP_HEIGHT = int(np.clip(_estimated_column_width / 1.5 + 24.0, 280, 440))
else:
    MAP_HEIGHT = 440

# ============================================================

# PROJECT PATHS

# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FORECAST_FILE = (

    BASE_DIR

    / "data"

    / "processed"

    / "forecast"

    / "demise_forecast_1982-2026.nc"

)

SHAPEFILE = BASE_DIR / "shapefiles" / "BR_UF_2022_rct.shp"
MUNICIPAL_SHAPEFILE = BASE_DIR / "shapefiles" / "cidades_rct.shp"  # /home/gustavo/dupla_safra_view_web/shapefiles/cidades_rct.shp

# ============================================================

# MAP SETTINGS

# ============================================================

# extent = [xmin, xmax, ymin, ymax]

EXTENT = [-65.5, -43, -18, -3]

XTICKS = [-45, -50, -55, -60, -65]

XTICKLABELS = ["45°W", "50°W", "55°W", "60°W", "65°W"]

YTICKS = [-5, -10, -15]

YTICKLABELS = ["5°S", "10°S", "15°S"]

# Each NetCDF latitude/longitude is the center of a 0.5° grid cell.

PIXEL_RESOLUTION = 0.5

PIXEL_HALF_SIZE = PIXEL_RESOLUTION / 2.0  # exactly 0.25° on each side

# The selection layer is visually transparent. Its small markers sit at the

# actual NetCDF cell centers and are used only as click/selection targets.

SELECTION_MARKER_SIZE = 10

SELECTION_CURVE_NUMBER = 4

# Reduce the complexity of the state boundaries only once, then cache them.

BOUNDARY_SIMPLIFY_TOLERANCE = 0.02

# Tolerância de simplificação dos limites municipais, em graus.
# Aplicada uma única vez e armazenada em cache para reduzir o custo
# de desenho do shapefile municipal.
MUNICIPAL_SIMPLIFY_TOLERANCE = 0.02

# Discrete classes for observed/predicted rainy-season demise.
DEFAULT_COLORS = [
    "#CC4C39", "#FD8C50", "#FBCF84", "#FFFFBD", "#8AC2E0",
    "#4769BA", "#31349C", "#8A2BE2", "#A8E6A2", "#4CAF50", "#7F7F7F",
]
DEMISE_THRESHOLDS = [243, 258, 274, 289, 304, 319, 335, 350]
# Classes: <243; [243,258); [258,274); [274,289); [289,304);
# [304,319); [319,335); [335,350); >=350.
# The grey color is reserved for missing values; valid classes follow the
# same red-to-green progression as the supplied reference legend.
DEMISE_CLASS_COLORS = [
    DEFAULT_COLORS[0],  # <243
    DEFAULT_COLORS[2],  # 243-258
    DEFAULT_COLORS[3],  # 258-274
    DEFAULT_COLORS[4],  # 274-289
    DEFAULT_COLORS[5],  # 289-304
    DEFAULT_COLORS[6],  # 304-319
    DEFAULT_COLORS[7],  # 319-335
    DEFAULT_COLORS[8],  # 335-350
    DEFAULT_COLORS[9],  # >=350
]
DEMISE_COLOR_MIN = 200
DEMISE_COLOR_MAX = 400

# Continuous error scale from -21 to +21 days.
# Negative errors: dark to light blue; positive errors: light to dark red.
ERROR_COLOR_MIN = -21
ERROR_COLOR_MAX = 21
ERROR_CONTINUOUS_COLORSCALE = [
    [0.00, "#08306B"],  # -21: dark blue
    [0.18, "#2171B5"],
    [0.38, "#9ECAE1"],
    [0.50, "#F7F7F7"],  # 0: neutral, no green
    [0.62, "#FCAE91"],
    [0.82, "#EF3B2C"],
    [1.00, "#67000D"],  # +21: dark red
]

# ============================================================

# LOAD NETCDF

# ============================================================

@st.cache_data(show_spinner="Carregando dados da previsão...")

def load_data(file_path: str, file_mtime: float):

    path = Path(file_path)

    if not path.exists():

        raise FileNotFoundError(f"NetCDF file not found: {path}")

    with xr.open_dataset(path) as source:

        dataset = source.load()

    required_variables = {

        "demise_observed",

        "demise_predicted",

        "error",

    }

    missing = required_variables.difference(dataset.data_vars)

    if missing:

        raise KeyError(

            f"Missing variables in NetCDF: {sorted(missing)}. "

            f"Available: {list(dataset.data_vars)}"

        )

    required_coords = {"time", "latitude", "longitude"}

    missing_coords = required_coords.difference(dataset.coords)

    if missing_coords:

        raise KeyError(

            f"Missing coordinates in NetCDF: {sorted(missing_coords)}. "

            f"Available: {list(dataset.coords)}"

        )

    return dataset

# ============================================================

# PREPARE STATE BOUNDARIES ONCE

# ============================================================

def iter_polygons(geometry):

    """Yield Polygon objects from Polygon, MultiPolygon or collections."""

    if geometry is None or geometry.is_empty:

        return

    if geometry.geom_type == "Polygon":

        yield geometry

    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):

        for part in geometry.geoms:

            yield from iter_polygons(part)

@st.cache_data(show_spinner="Carregando e simplificando divisas estaduais...")

def load_boundary_coordinates(

    shapefile_path: str,

    shapefile_mtime: float,

    simplify_tolerance: float,

):

    path = Path(shapefile_path)

    if not path.exists():

        raise FileNotFoundError(f"Shapefile not found: {path}")

    gdf = gpd.read_file(path)

    if gdf.crs is not None:

        gdf = gdf.to_crs("EPSG:4326")

    gdf = gdf.copy()

    gdf["geometry"] = gdf.geometry.simplify(

        tolerance=simplify_tolerance,

        preserve_topology=True,

    )

    xs = []

    ys = []

    for geometry in gdf.geometry:

        for polygon in iter_polygons(geometry):

            rings = [polygon.exterior, *polygon.interiors]

            for ring in rings:

                coordinates = list(ring.coords)

                if not coordinates:

                    continue

                xs.extend(coord[0] for coord in coordinates)

                ys.extend(coord[1] for coord in coordinates)

                # Separate rings so different boundaries are never connected.

                xs.append(None)

                ys.append(None)

    return xs, ys


# ============================================================
# PREPARE MUNICIPAL BOUNDARIES AND SEARCH OPTIONS ONCE
# ============================================================
@st.cache_data(show_spinner="Carregando e simplificando limites municipais...")
def load_municipality_layers(
    shapefile_path: str,
    shapefile_mtime: float,
    simplify_tolerance: float,
):
    """Load city polygons, simplify them, and prepare lightweight Plotly traces."""
    path = Path(shapefile_path)
    if not path.exists():
        raise FileNotFoundError(f"Municipal shapefile not found: {path}")

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(
            "O shapefile de municípios não informa o CRS. "
            "Verifique se o arquivo cidades_rct.prj está presente."
        )
    gdf = gdf.to_crs("EPSG:4326")

    required_columns = {"NM_MUN", "SIGLA_UF"}
    missing_columns = required_columns.difference(gdf.columns)
    if missing_columns:
        raise KeyError(
            f"Colunas ausentes no shapefile municipal: {sorted(missing_columns)}. "
            f"Colunas disponíveis: {list(gdf.columns)}"
        )

    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.simplify(
        tolerance=simplify_tolerance,
        preserve_topology=True,
    )

    # The supplied shapefile contains NM_MUN and SIGLA_UF. Include the state
    # abbreviation in each option so municipalities with repeated names are clear.
    names = gdf["NM_MUN"].fillna("Município sem nome").astype(str).str.strip()
    states = gdf["SIGLA_UF"].fillna("").astype(str).str.strip()
    labels = names + " - " + states
    duplicate_labels = labels.duplicated(keep=False)
    if duplicate_labels.any() and "CD_MUN" in gdf.columns:
        codes = gdf["CD_MUN"].fillna("").astype(str).str.strip()
        labels.loc[duplicate_labels] = (
            labels.loc[duplicate_labels] + " (" + codes.loc[duplicate_labels] + ")"
        )
    gdf["_municipality_label"] = labels
    gdf = gdf.sort_values("_municipality_label").reset_index(drop=True)

    # All municipal dividers are encoded into one trace. The index range for
    # each municipality lets us highlight the chosen one without rereading the SHP.
    all_x, all_y = [], []
    municipality_segments = {}
    municipality_fill_coordinates = {}
    municipality_labels = []

    for _, row in gdf.iterrows():
        label = str(row["_municipality_label"])
        geometry = row.geometry
        start = len(all_x)
        fill_x, fill_y = [], []

        for polygon in iter_polygons(geometry):
            # Exterior boundaries and interior rings form the full municipality line.
            rings = [polygon.exterior, *polygon.interiors]
            for ring in rings:
                coordinates = list(ring.coords)
                if not coordinates:
                    continue
                all_x.extend(coord[0] for coord in coordinates)
                all_y.extend(coord[1] for coord in coordinates)
                all_x.append(None)
                all_y.append(None)

            # Exterior rings provide a translucent fill for the selected area.
            exterior = list(polygon.exterior.coords)
            if exterior:
                fill_x.extend(coord[0] for coord in exterior)
                fill_y.extend(coord[1] for coord in exterior)
                fill_x.append(None)
                fill_y.append(None)

        municipality_segments[label] = (start, len(all_x))
        municipality_fill_coordinates[label] = (fill_x, fill_y)
        municipality_labels.append(label)

    return (
        all_x,
        all_y,
        municipality_labels,
        municipality_segments,
        municipality_fill_coordinates,
    )

if not FORECAST_FILE.exists():

    st.error(f"NetCDF file not found: {FORECAST_FILE}")

    st.stop()

if not SHAPEFILE.exists():

    st.error(

        f"Shapefile not found: {SHAPEFILE}\n\n"

        "Confira se o arquivo .shp e seus arquivos auxiliares estão na pasta shapefiles."

    )

    st.stop()

if not MUNICIPAL_SHAPEFILE.exists():

    st.error(

        f"Shapefile municipal não encontrado: {MUNICIPAL_SHAPEFILE}\n\n"

        "Coloque cidades_rct.shp, cidades_rct.shx, cidades_rct.dbf, "

        "cidades_rct.prj e cidades_rct.cpg na pasta shapefiles."

    )

    st.stop()

ds = load_data(str(FORECAST_FILE), FORECAST_FILE.stat().st_mtime)

boundary_x, boundary_y = load_boundary_coordinates(

    str(SHAPEFILE),

    SHAPEFILE.stat().st_mtime,

    BOUNDARY_SIMPLIFY_TOLERANCE,

)

municipal_boundary_x, municipal_boundary_y, municipality_labels, municipality_segments, municipality_fill_coordinates = load_municipality_layers(
    str(MUNICIPAL_SHAPEFILE),
    MUNICIPAL_SHAPEFILE.stat().st_mtime,
    MUNICIPAL_SIMPLIFY_TOLERANCE,
)

# ============================================================

# SESSION STATE

# ============================================================

if "selected_lat" not in st.session_state:

    st.session_state.selected_lat = None

if "selected_lon" not in st.session_state:

    st.session_state.selected_lon = None

if "selected_year" not in st.session_state:

    st.session_state.selected_year = None

if "selection_version" not in st.session_state:

    st.session_state.selection_version = 0

# ============================================================

# SELECTION HELPERS

# ============================================================

def get_field(obj, key, default=None):

    """Read a field from mappings or Streamlit's dict-like event objects."""

    if obj is None:

        return default

    getter = getattr(obj, "get", None)

    if callable(getter):

        try:

            value = getter(key, default)

            return default if value is None and default is not None else value

        except (AttributeError, KeyError, TypeError, ValueError):

            pass

    try:

        return getattr(obj, key)

    except (AttributeError, KeyError, TypeError, ValueError):

        return default

def get_selected_point(event):

    """Extract the point selected on the transparent cell-center layer."""

    if event is None:

        return None

    selection = get_field(event, "selection", {})

    points = get_field(selection, "points", []) or []

    if not points:

        return None

    candidates = []

    for point in points:

        curve_number = get_field(point, "curve_number", None)

        if curve_number is None:

            curve_number = get_field(point, "curveNumber", None)

        try:

            if int(curve_number) != SELECTION_CURVE_NUMBER:

                continue

        except (TypeError, ValueError):

            continue

        x = get_field(point, "x", None)

        y = get_field(point, "y", None)

        if x is None or y is None:

            continue

        try:

            clicked_lon = float(x)

            clicked_lat = float(y)

        except (TypeError, ValueError, OverflowError):

            continue

        if np.isfinite(clicked_lon) and np.isfinite(clicked_lat):

            candidates.append((clicked_lon, clicked_lat))

    if not candidates:

        return None

    # The interaction layer is deliberately small to avoid adjacent cells

    # competing for the same click. If more than one point is emitted, use

    # the last point reported by Plotly and snap it to the NetCDF grid below.

    return candidates[-1]

def nearest_grid_coordinate(value, coordinates):

    """Snap a selected coordinate to an exact coordinate from the NetCDF grid."""

    values = np.asarray(coordinates, dtype=float).ravel()

    values = values[np.isfinite(values)]

    if values.size == 0:

        raise ValueError("The NetCDF coordinate array contains no finite values.")

    return float(values[np.argmin(np.abs(values - float(value)))])

# ============================================================

# MAP DATA HELPERS

# ============================================================

def get_grid_values(data_array):

    """Return values arranged as [latitude, longitude] for go.Heatmap."""

    return np.asarray(

        data_array.transpose("latitude", "longitude").values,

        dtype=float,

    )

def make_selection_dataframe(data_array, lon_coordinates, lat_coordinates):

    """Create one selectable point at the center of every finite grid cell."""

    values = get_grid_values(data_array)

    lon_2d, lat_2d = np.meshgrid(lon_coordinates, lat_coordinates)

    valid = np.isfinite(values)

    return pd.DataFrame(

        {

            "longitude": lon_2d[valid],

            "latitude": lat_2d[valid],

        }

    )

def add_selection_layer(fig, dataframe):

    """Add a nearly invisible point target at each actual cell center.

    Heatmap renders the visible 0.5° cells. This separate Scatter layer is

    what Streamlit can select with selection_mode="points". Use regular SVG

    Scatter here (rather than Scattergl) because it provides reliable point

    hit-testing for very transparent markers in this small grid.

    """

    fig.add_trace(

        go.Scatter(

            x=dataframe["longitude"].to_numpy(),

            y=dataframe["latitude"].to_numpy(),

            mode="markers",

            marker=dict(

                symbol="square",

                size=SELECTION_MARKER_SIZE,

                # Almost invisible, but not zero-alpha: Plotly still needs a

                # real marker target for click/point-selection hit testing.

                color="rgba(0, 0, 0, 0.01)",

                line=dict(width=0),

            ),

            hoverinfo="none",

            showlegend=False,

            name="pixel_selection_layer",

        )

    )

def add_boundaries(fig):

    """Draw state boundaries with one trace."""

    fig.add_trace(

        go.Scattergl(

            x=boundary_x,

            y=boundary_y,

            mode="lines",

            line=dict(color="black", width=1),

            hoverinfo="skip",

            showlegend=False,

            connectgaps=False,

            name="state_boundaries",

        )

    )



def add_selected_municipality(fig, municipality_label):
    """Add two fixed traces: translucent selected area and its strong outline."""
    if municipality_label and municipality_label in municipality_fill_coordinates:
        fill_x, fill_y = municipality_fill_coordinates[municipality_label]
        start, end = municipality_segments[municipality_label]
        outline_x = municipal_boundary_x[start:end]
        outline_y = municipal_boundary_y[start:end]
    else:
        # Keep these traces even when no city is chosen, so the selection-layer
        # curve number remains constant and pixel clicks keep working.
        fill_x, fill_y = [None], [None]
        outline_x, outline_y = [None], [None]

    fig.add_trace(
        go.Scatter(
            x=fill_x,
            y=fill_y,
            mode="lines",
            fill="toself",
            fillcolor="rgba(115, 115, 115, 0)",
            line=dict(color="rgba(255, 0, 170, 0)", width=0),
            hoverinfo="skip",
            showlegend=False,
            connectgaps=False,
            name="selected_municipality_fill",
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=outline_x,
            y=outline_y,
            mode="lines",
            line=dict(color="#ff00aa", width=2.8),
            hoverinfo="skip",
            showlegend=False,
            connectgaps=False,
            name="selected_municipality_outline",
        )
    )

def add_selected_pixel_box(fig, selected_lon, selected_lat):

    """Outline the selected cell exactly 0.25° on each side of its center."""

    if selected_lon is None or selected_lat is None:

        return

    lon_min = float(selected_lon) - PIXEL_HALF_SIZE

    lon_max = float(selected_lon) + PIXEL_HALF_SIZE

    lat_min = float(selected_lat) - PIXEL_HALF_SIZE

    lat_max = float(selected_lat) + PIXEL_HALF_SIZE

    fig.add_trace(

        go.Scattergl(

            x=[lon_min, lon_max, lon_max, lon_min, lon_min],

            y=[lat_min, lat_min, lat_max, lat_max, lat_min],

            mode="lines",

            line=dict(color="black", width=2.5),

            hoverinfo="skip",

            showlegend=False,

            connectgaps=False,

            name="selected_pixel_outline",

        )

    )

def configure_map(fig, title):
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.99,
            yanchor="top",
            font=dict(color="#222222", size=16),
        ),
        font=dict(family="Arial, sans-serif", size=12, color="#222222"),
        # Keep enough top space for the title while reducing the vertical gap
        # between the title and the map by letting the map fill more of the figure.
        # Aumenta a área útil do mapa e reduz margens laterais/superiores.
        height=MAP_HEIGHT,
        autosize=True,
        margin=dict(l=4, r=4, t=12, b=0),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        clickmode="event+select",
        dragmode="zoom",
        xaxis=dict(
            title=None,
            range=EXTENT[:2],
            autorange=False,
            showticklabels=False,
            ticks="",
            showgrid=False,
            showline=False,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
            constrain="domain",
        ),
        yaxis=dict(
            title=None,
            range=EXTENT[2:],
            autorange=False,
            showticklabels=False,
            ticks="",
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
    )


def make_discrete_colorscale(thresholds, colors, zmin, zmax):
    """Create hard color transitions at numeric class thresholds."""
    bounds = [zmin, *thresholds, zmax]
    scale = []
    span = zmax - zmin
    for i, color in enumerate(colors):
        left = (bounds[i] - zmin) / span
        right = (bounds[i + 1] - zmin) / span
        scale.append([left, color])
        scale.append([max(left, right - 1e-10), color])
    scale[-1][0] = 1.0
    return scale


def render_demise_legend():
    """Render a horizontal categorical legend that follows Streamlit's active theme."""
    colors = DEMISE_CLASS_COLORS
    labels = ["1 mar", "15 mar", "1 abr", "15 abr", "1 mai", "15 mai", "1 jun", "15 jun"]

    # Use Streamlit's live CSS text-color token rather than selecting a fixed
    # color in Python. This lets the legend adapt immediately when the user
    # switches between light and dark mode, without requiring a script rerun.
    theme_text = "color:var(--text-color, inherit);"

    segments = []
    for i, color in enumerate(colors):
        shape = "clip-path:polygon(0 50%, 100% 0, 100% 100%)" if i == 0 else ""
        if i == len(colors) - 1:
            shape = "clip-path:polygon(0 0, 100% 50%, 0 100%)"
        segments.append(
            f'<div style="flex:1; min-width:0; height:14px; background:{color}; {shape}"></div>'
        )

    ticks = []
    for i, label in enumerate(labels, start=1):
        x = 100 * i / len(colors)
        ticks.append(
            f'<div style="position:absolute; left:{x}%; top:0; transform:translateX(-50%); '
            f'text-align:center; white-space:nowrap; font-size:9px; {theme_text}">'
            f'<div style="height:5px; border-left:1px solid currentColor; margin:auto; width:1px;"></div>'
            f'<div style="padding-top:2px;">{label}</div></div>'
        )

    html = (
        '<div style="width:100%; margin:-10px auto 0; padding:0 0 4px 0; box-sizing:border-box; '
        'background:transparent; color:var(--text-color, inherit);">'
        '<div style="display:flex; width:100%; height:14px; border:0; box-sizing:border-box;">'
        + "".join(segments)
        + '</div><div style="position:relative; height:25px;">'
        + "".join(ticks)
        + f'</div><div style="text-align:center; font-size:11px; margin-top:1px; {theme_text}">'
        'Fim da estação chuvosa</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_error_legend():
    """Render a horizontal continuous error legend with theme-aware labels."""
    theme_text = "color:var(--text-color, inherit);"
    gradient = (
        "linear-gradient(to right, "
        "#08306B 0%, #2171B5 18%, #9ECAE1 38%, "
        "#F7F7F7 50%, #FCAE91 62%, #EF3B2C 82%, #67000D 100%)"
    )
    tick_values = [-21, -14, -7, 0, 7, 14, 21]
    ticks = []
    for value in tick_values:
        x = (value - ERROR_COLOR_MIN) / (ERROR_COLOR_MAX - ERROR_COLOR_MIN) * 100
        label = f"{value}" if value > 0 else ("0" if value == 0 else f"−{abs(value)}")
        if value == ERROR_COLOR_MIN:
            transform = "translateX(0)"
            alignment = "text-align:left"
        elif value == ERROR_COLOR_MAX:
            transform = "translateX(-100%)"
            alignment = "text-align:right"
        else:
            transform = "translateX(-50%)"
            alignment = "text-align:center"
        ticks.append(
            f'<div style="position:absolute; left:{x:.4f}%; top:0; '
            f'transform:{transform}; {alignment}; white-space:nowrap; '
            f'font-size:9px; {theme_text}">'
            f'<div style="height:5px; border-left:1px solid currentColor; margin:auto; width:1px;"></div>'
            f'<div style="padding-top:2px;">{label}</div></div>'
        )

    html = (
        '<div style="width:100%; margin:-10px auto 0; padding:0 0 4px 0; box-sizing:border-box; '
        'background:transparent; color:var(--text-color, inherit);">'
        f'<div style="width:100%; height:14px; background:{gradient}; border:0; box-sizing:border-box;"></div>'
        '<div style="position:relative; width:100%; height:25px;">'
        + "".join(ticks)
        + f'</div><div style="text-align:center; font-size:11px; margin-top:1px; {theme_text}">'
        + 'Erro da previsão (dias)</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def build_map(

    data_array,

    title,

    color_scale,

    hover_label,

    selected_lon,

    selected_lat,

    lon_coordinates,

    lat_coordinates,

    selected_municipality_label=None,

    range_color=None,

    discrete_demise=False,

    discrete_error=False,

    continuous_error=False,

):

    """Build a map whose colored cells occupy their actual 0.5° grid cells."""

    grid_values = get_grid_values(data_array)

    selection_df = make_selection_dataframe(

        data_array,

        lon_coordinates,

        lat_coordinates,

    )

    heatmap_kwargs = dict(

        x=lon_coordinates,

        y=lat_coordinates,

        z=grid_values,

        colorscale=color_scale,

        xgap=0,

        ygap=0,

        zsmooth=False,

        hoverongaps=False,

        hovertemplate=(

            "Latitude: %{y:.2f}°<br>"

            "Longitude: %{x:.2f}°<br>"

            f"{hover_label}: %{{z:.1f}}"

            "<extra></extra>"

        ),

        name=hover_label,

    )

    if discrete_demise:
        heatmap_kwargs["colorscale"] = make_discrete_colorscale(
            DEMISE_THRESHOLDS,
            DEMISE_CLASS_COLORS,
            DEMISE_COLOR_MIN,
            DEMISE_COLOR_MAX,
        )
        heatmap_kwargs["zmin"] = DEMISE_COLOR_MIN
        heatmap_kwargs["zmax"] = DEMISE_COLOR_MAX
        heatmap_kwargs["showscale"] = False
    elif continuous_error:
        # Smooth diverging scale: blue for negative errors, neutral at zero,
        # and red for positive errors. The custom horizontal legend is rendered below.
        heatmap_kwargs["colorscale"] = ERROR_CONTINUOUS_COLORSCALE
        heatmap_kwargs["zmin"] = ERROR_COLOR_MIN
        heatmap_kwargs["zmax"] = ERROR_COLOR_MAX
        heatmap_kwargs["showscale"] = False
    elif discrete_error:
        # Retained for backwards compatibility if discrete error classes are re-enabled.
        heatmap_kwargs["showscale"] = False
    else:
        heatmap_kwargs["colorbar"] = dict(
            title=dict(text=hover_label),
            thickness=12,
            len=0.85,
        )

    if range_color is not None:

        heatmap_kwargs["zmin"] = range_color[0]

        heatmap_kwargs["zmax"] = range_color[1]

    fig = go.Figure()

    # Heatmap x/y values are grid-cell centers. With 0.5° regular spacing,

    # each cell extends 0.25° around the center in both coordinate directions.

    # Unlike Scatter square markers, this is a data-coordinate cell, not a

    # square whose size is measured in screen pixels.

    fig.add_trace(go.Heatmap(**heatmap_kwargs))

    # Trace order is deliberately fixed in every map:

    # 0 heatmap; 1 state boundaries; 2 selected-municipality fill;

    # 3 selected-municipality outline; 4 selectable pixel centers;

    # 5 selected-cell outline (if any).

    # Do not draw the full municipal boundary layer by default. The municipality

    # shapefile remains loaded for the selector and for highlighting one chosen

    # municipality. Keep the two highlight traces even with no selection so

    # SELECTION_CURVE_NUMBER stays fixed at 4.

    add_boundaries(fig)

    add_selected_municipality(fig, selected_municipality_label)

    add_selection_layer(fig, selection_df)

    add_selected_pixel_box(fig, selected_lon, selected_lat)

    configure_map(fig, title)

    return fig

# ============================================================

# TITLE AND YEAR SELECTOR

# ============================================================

st.title("Previsão do fim da estação chuvosa no Brasil Central")

# Description with two inline information icons and centered modal dialogs.
# Dialogs close through the X button, a click on the dark backdrop, or Escape.
st.html(
    """
    <div id="inline-information-widget">
        <p class="project-description">
            O modelo tem como objetivo prever o fim da estação chuvosa na região central 
            do Brasil com 1 a 3 meses de antecedência. As previsões são disponibilizadas 
            no início de março. No início de julho, quando os dados observados estão 
            integralmente disponíveis, a aplicação passa a apresentar também o mapa 
            com as datas observadas de término da estação chuvosa e os respectivos 
            erros de previsão. Maiores detalhes sobre o modelo desenvolvido 
            podem ser consultadas em
            <button
                type="button"
                class="inline-icon info-icon"
                data-modal-trigger="model-info-modal"
                aria-label="Mais informações sobre o modelo"
                title="Mais informações sobre o modelo"
            >i</button>.
            O uso dessas informações implica a aceitação da declaração de isenção
            de responsabilidade especificada em
            <button
                type="button"
                class="inline-icon warning-icon"
                data-modal-trigger="disclaimer-modal"
                aria-label="Consultar a declaração de isenção de responsabilidade"
                title="Consultar a declaração de isenção de responsabilidade"
            >!</button>.
        </p>

        <dialog id="model-info-modal" class="app-modal">
            <div class="modal-header">
                <strong>Informações sobre o modelo</strong>
                <button type="button" class="modal-close" aria-label="Fechar">×</button>
            </div>
            <div class="modal-body">
                <p>
                    A documentação e os arquivos relacionados ao desenvolvimento
                    desta aplicação estão disponíveis no repositório do projeto:
                    <a href="https://github.com/gmairink/dupla-safra" target="_blank"
                       rel="noopener noreferrer">gmairink/dupla-safra</a>.
                </p>
            </div>
        </dialog>

        <dialog id="disclaimer-modal" class="app-modal">
            <div class="modal-header">
                <strong>Declaração de isenção de responsabilidade</strong>
                <button type="button" class="modal-close" aria-label="Fechar">×</button>
            </div>
            <div class="modal-body">
                <p>
                    Este trabalho foi desenvolvido no âmbito da Universidade Federal de
                    Viçosa (UFV), como parte do Trabalho de Conclusão de Curso do autor,
                    com atividades realizadas em um laboratório da instituição.
                </p>

                <p>
                    As informações disponibilizadas nesta plataforma são apresentadas
                    na forma em que foram produzidas, sem garantias expressas ou
                    implícitas, incluindo, entre outras, garantias de adequação a uma
                    finalidade específica.
                </p>

                <p>
                    O usuário assume todos os riscos relacionados ao uso dessas
                    informações. Em nenhuma circunstância a UFV será responsável
                    perante os usuários ou terceiros por quaisquer danos diretos,
                    indiretos, incidentais, consequenciais ou especiais, nem por
                    lucros cessantes decorrentes do uso ou do uso indevido desses dados.
                </p>
                </p>
            </div>
        </dialog>
    </div>

    <style>
        #inline-information-widget {
            width: 100%;
            line-height: 1.6;
            color: var(--text-color, inherit);
        }

        #inline-information-widget .project-description {
            margin: 0;
        }

        #inline-information-widget .inline-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 17px;
            height: 17px;
            margin: 0 2px;
            padding: 0;
            border: 0;
            border-radius: 50%;
            color: #ffffff;
            font-family: Arial, sans-serif;
            font-size: 12px;
            font-weight: 700;
            line-height: 1;
            vertical-align: baseline;
            cursor: pointer;
        }

        #inline-information-widget .info-icon {
            background: #2878C8;
        }

        #inline-information-widget .info-icon:hover {
            background: #174b82;
        }

        #inline-information-widget .warning-icon {
            background: #D97706;
        }

        #inline-information-widget .warning-icon:hover {
            background: #A65305;
        }

        #inline-information-widget .inline-icon:focus-visible,
        #inline-information-widget .modal-close:focus-visible {
            outline: 2px solid #2878C8;
            outline-offset: 3px;
        }

        #inline-information-widget .app-modal {
            width: min(560px, calc(100vw - 32px));
            max-width: calc(100vw - 32px);
            max-height: min(78vh, 720px);
            margin: auto;
            padding: 0;
            border: 1px solid rgba(0, 0, 0, 0.12);
            border-radius: 4px;
            background: #ffffff;
            color: #555555;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.24);
            overflow: hidden;
        }

        #inline-information-widget .app-modal::backdrop {
            background: rgba(0, 0, 0, 0.48);
        }

        #inline-information-widget .modal-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            padding: 16px 20px;
            border-bottom: 1px solid #e5e5e5;
            color: #333333;
            font-size: 1rem;
            line-height: 1.4;
        }

        #inline-information-widget .modal-close {
            flex: 0 0 auto;
            margin: -5px -7px 0 0;
            padding: 0 4px;
            border: 0;
            background: transparent;
            color: #888888;
            font-size: 25px;
            font-weight: 300;
            line-height: 1;
            cursor: pointer;
        }

        #inline-information-widget .modal-close:hover {
            color: #222222;
        }

        #inline-information-widget .modal-body {
            max-height: calc(78vh - 65px);
            overflow-y: auto;
            padding: 18px 22px 22px;
            font-size: 0.95rem;
            line-height: 1.6;
            text-align: justify;
        }

        #inline-information-widget .modal-body p {
            margin: 0 0 0.8rem;
        }

        #inline-information-widget .modal-body p:last-child {
            margin-bottom: 0;
        }

        #inline-information-widget .modal-body a {
            color: #2878C8;
            text-decoration: underline;
        }
    </style>

    <script>
        (() => {
            const widgetId = "inline-information-widget";
            if (window.__inlineModalHandlersInstalled) return;

            document.addEventListener("click", (event) => {
                const widget = document.getElementById(widgetId);
                if (!widget) return;

                const target = event.target;
                if (!(target instanceof Element)) return;

                const trigger = target.closest("[data-modal-trigger]");
                if (trigger && widget.contains(trigger)) {
                    const modalId = trigger.getAttribute("data-modal-trigger");
                    const modal = widget.querySelector(`#${modalId}`);
                    if (modal && !modal.open) modal.showModal();
                    return;
                }

                const closeButton = target.closest(".modal-close");
                if (closeButton && widget.contains(closeButton)) {
                    const modal = closeButton.closest("dialog");
                    if (modal && modal.open) modal.close();
                }

                // Clicking the backdrop dispatches the click event on the dialog itself.
                if (target instanceof HTMLDialogElement && target.open &&
                    target.classList.contains("app-modal")) {
                    target.close();
                }
            });

            window.__inlineModalHandlersInstalled = true;
        })();
    </script>
    """,
    unsafe_allow_javascript=True,
    width="stretch",
)

selected_municipality_label = st.selectbox(
    "Localizar município",
    options=[""] + municipality_labels,
    index=0,
    key="municipality_selector",
    format_func=lambda value: "Digite ou selecione um município..." if value == "" else value,
)
selected_municipality_label = selected_municipality_label or None

if selected_municipality_label:
    st.caption(f"Município destacado: **{selected_municipality_label}**")

years = ds.time.values.astype(int)

year = st.selectbox(

    "Ano",

    years,

    index=len(years) - 1,

)

if st.session_state.selected_year != int(year):

    st.session_state.selected_lat = None

    st.session_state.selected_lon = None

    st.session_state.selected_year = int(year)

    st.session_state.selection_version += 1

# ============================================================

# SELECT THE YEAR'S DATA

# ============================================================

predicted = ds["demise_predicted"].sel(time=year)

observed = ds["demise_observed"].sel(time=year)

error = ds["error"].sel(time=year)

lon = np.asarray(ds["longitude"].values, dtype=float)

lat = np.asarray(ds["latitude"].values, dtype=float)

# Confirm that this app is working with the expected regular 0.5° grid.

if len(lon) > 1 and not np.allclose(np.abs(np.diff(lon)), PIXEL_RESOLUTION, atol=1e-6):

    st.warning(

        "A longitude do NetCDF não tem espaçamento uniforme de 0,5°. "

        "O contorno selecionado continuará usando ±0,25° conforme solicitado."

    )

if len(lat) > 1 and not np.allclose(np.abs(np.diff(lat)), PIXEL_RESOLUTION, atol=1e-6):

    st.warning(

        "A latitude do NetCDF não tem espaçamento uniforme de 0,5°. "

        "O contorno selecionado continuará usando ±0,25° conforme solicitado."

    )

# If a selection exists, snap it to the exact center coordinates stored in NetCDF.

if st.session_state.selected_lon is not None:

    st.session_state.selected_lon = nearest_grid_coordinate(

        st.session_state.selected_lon, lon

    )

if st.session_state.selected_lat is not None:

    st.session_state.selected_lat = nearest_grid_coordinate(

        st.session_state.selected_lat, lat

    )

# ============================================================

# BUILD THREE MAPS

# ============================================================

selected_lon = st.session_state.selected_lon

selected_lat = st.session_state.selected_lat

fig_pred = build_map(

    data_array=predicted,

    title="Previsto",

    color_scale="RdYlBu",

    hover_label="Fim previsto (dias)",

    selected_lon=selected_lon,

    selected_lat=selected_lat,

    lon_coordinates=lon,

    lat_coordinates=lat,

    selected_municipality_label=selected_municipality_label,

    discrete_demise=True,

)

fig_obs = build_map(

    data_array=observed,

    title="Observado",

    color_scale="RdYlBu",

    hover_label="Fim observado (dias)",

    selected_lon=selected_lon,

    selected_lat=selected_lat,

    lon_coordinates=lon,

    lat_coordinates=lat,

    selected_municipality_label=selected_municipality_label,

    discrete_demise=True,

)

fig_error = build_map(

    data_array=error,

    title="Erro",

    color_scale=ERROR_CONTINUOUS_COLORSCALE,

    hover_label="Erro (dias)",

    selected_lon=selected_lon,

    selected_lat=selected_lat,

    lon_coordinates=lon,

    lat_coordinates=lat,

    selected_municipality_label=selected_municipality_label,

    range_color=[ERROR_COLOR_MIN, ERROR_COLOR_MAX],

    continuous_error=True,

)

# ============================================================

# RENDER THREE MAPS

# ============================================================

# Streamlit's returned Plotly selection state is processed immediately below.

# This avoids relying on a callback to read the widget state before it is

# available, which can result in clicks being ignored in some Streamlit versions.

key_pred = f"predicted_map_{year}_{st.session_state.selection_version}"

key_obs = f"observed_map_{year}_{st.session_state.selection_version}"

key_error = f"error_map_{year}_{st.session_state.selection_version}"

col_pred, col_obs, col_error = st.columns(3, gap="small")

with col_pred:

    event_pred = st.plotly_chart(

        fig_pred,

        use_container_width=True,

        key=key_pred,

        config={"responsive": True},

        on_select="rerun",

        selection_mode="points",

    )
    render_demise_legend()

with col_obs:

    event_obs = st.plotly_chart(

        fig_obs,

        use_container_width=True,

        key=key_obs,

        config={"responsive": True},

        on_select="rerun",

        selection_mode="points",

    )
    render_demise_legend()

with col_error:

    event_error = st.plotly_chart(

        fig_error,

        use_container_width=True,

        key=key_error,

        config={"responsive": True},

        on_select="rerun",

        selection_mode="points",

    )
    render_error_legend()

# Process selection events from any of the three maps. Trace order in every

# figure: 0=heatmap, 1=municipal, 2=state, 3-4=selected municipality, 5=click target.

current_point = None

if st.session_state.selected_lon is not None and st.session_state.selected_lat is not None:

    current_point = (

        float(st.session_state.selected_lon),

        float(st.session_state.selected_lat),

    )

new_point = None

for chart_event in (event_pred, event_obs, event_error):

    clicked_point = get_selected_point(chart_event)

    if clicked_point is None:

        continue

    clicked_lon, clicked_lat = clicked_point

    snapped_point = (

        nearest_grid_coordinate(clicked_lon, lon),

        nearest_grid_coordinate(clicked_lat, lat),

    )

    if snapped_point != current_point:

        new_point = snapped_point

if new_point is not None:

    st.session_state.selected_lon = new_point[0]

    st.session_state.selected_lat = new_point[1]

    # Change all three chart keys to clear stale Plotly point selections.

    # This prevents an old click from being re-read on a later interaction.

    st.session_state.selection_version += 1

    st.rerun()

# ============================================================

# DATE FORMATTING FOR DEMISE VALUES

# ============================================================

def format_demise_date(day_offset, target_year):

    """Convert days after 1 July of the previous year into a calendar date.

    Using datetime arithmetic automatically accounts for leap years.
    Fractional day values are rounded to the nearest whole day.
    """

    if not np.isfinite(day_offset):
        return "—"

    start_date = date(int(target_year) - 1, 7, 1)
    event_date = start_date + timedelta(days=int(round(float(day_offset))))
    month_abbreviations = [
        "jan", "fev", "mar", "abr", "mai", "jun",
        "jul", "ago", "set", "out", "nov", "dez",
    ]
    return f"{event_date.day} {month_abbreviations[event_date.month - 1]} {event_date.year}"


# ============================================================

# SELECTED PIXEL VALUES

# ============================================================

if st.session_state.selected_lat is not None and st.session_state.selected_lon is not None:

    selected_lat = float(st.session_state.selected_lat)

    selected_lon = float(st.session_state.selected_lon)

    selected_observed = float(

        observed.sel(latitude=selected_lat, longitude=selected_lon).item()

    )

    selected_predicted = float(

        predicted.sel(latitude=selected_lat, longitude=selected_lon).item()

    )

    selected_error = float(

        error.sel(latitude=selected_lat, longitude=selected_lon).item()

    )

    selected_observed_date = format_demise_date(selected_observed, int(year))

    selected_predicted_date = format_demise_date(selected_predicted, int(year))

    st.divider()

    st.subheader("Pixel selecionado")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric("Previsto", selected_predicted_date)

    with col2:

        st.metric("Observado", selected_observed_date)

    with col3:
        if np.isfinite(selected_error):
            error_rounded = int(round(selected_error))
            unit = "dia" if abs(error_rounded) <= 1 else "dias"
            error_text = f"{error_rounded:+d} {unit}"
        else:
            error_text = "—"

        st.metric("Erro", error_text)
else:

    st.info("Clique em uma célula da grade em qualquer um dos mapas para ver maiores detalhes. " \
    "Caso a disposição dos mapas se desconfigure, atualize a página.")

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        width: 100%;
        box-sizing: border-box;
        margin-top: 2rem;
        padding: 0.75rem 0 0.5rem 0;
        border-top: 1px solid rgba(128, 128, 128, 0.25);
        text-align: right;
        font-size: 0.85rem;
        color: var(--text-color, #6b7280);
    ">
        Desenvolvido por <strong>Gustavo Mairink</strong>
    </div>
    """,
    unsafe_allow_html=True,
)
