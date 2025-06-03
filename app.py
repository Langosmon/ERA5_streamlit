# app.py ── ERA5 monthly-mean browser on Streamlit Cloud
import xarray as xr
import streamlit as st
import plotly.express as px
import numpy as np
import cartopy.feature as cfeature
import plotly.graph_objects as go

st.set_page_config(layout="wide")
YEARS = list(range(1980, 2023))                      # RDA stops at 2022
COMMON_PLEVELS = [1000, 975, 850, 700, 500,
                  300, 250, 200, 100, 50, 10]        # hPa

# ─────────────────────────────────────────────────────────────────────────────
#  variable catalogue  (common-name : (domain, code, varname, units, cmap) )
# ─────────────────────────────────────────────────────────────────────────────
SURFACE = {
    "Sea-surface temperature" : ("sfc", "034", "sstk", "°C", "thermal"),
    "CAPE"                    : ("sfc", "059", "cape", "J kg⁻¹", "viridis"),
    "Surface geopotential"    : ("sfc", "129", "z",   "m² s⁻²", "magma"),
    "Surface pressure"        : ("sfc", "134", "sp",  "hPa",    "icefire"),
    "Mean sea-level pressure" : ("sfc", "151", "msl", "hPa",    "icefire"),
    "10-m zonal wind"         : ("sfc", "165", "10u", "m s⁻¹",  "curl"),
    "10-m meridional wind"    : ("sfc", "166", "10v", "m s⁻¹",  "curl_r"),
    "2-m temperature"         : ("sfc", "167", "2t",  "°C",     "thermal"),
}

PRESSURE = {
    "Potential vorticity" : ("pl", "060", "pv",  "PVU",      "plasma"),
    "Geopotential"        : ("pl", "129", "z",   "m² s⁻²",   "magma"),
    "Temperature"         : ("pl", "130", "t",   "K",        "thermal"),
    "Zonal wind"          : ("pl", "131", "u",   "m s⁻¹",    "curl"),
    "Meridional wind"     : ("pl", "132", "v",   "m s⁻¹",    "curl_r"),
    "Specific humidity"   : ("pl", "133", "q",   "kg kg⁻¹",  "viridis"),
    "Vertical velocity"   : ("pl", "135", "w",   "Pa s⁻¹",   "icefire"),
    "Relative vorticity"  : ("pl", "138", "vo",  "s⁻¹",      "plasma"),
    "Divergence"          : ("pl", "155", "d",   "s⁻¹",      "plasma"),
    "Relative humidity"   : ("pl", "157", "r",   "%",        "viridis"),
    "Ozone"               : ("pl", "203", "o3",  "kg kg⁻¹",  "viridis"),
}

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Field")
field_type = st.sidebar.radio("Domain", ("Surface", "Pressure level"))

if field_type == "Surface":
    choice = st.sidebar.selectbox("Variable", list(SURFACE))
    domain, code, vname, units, cmap = SURFACE[choice]
    plevel = None
else:
    choice = st.sidebar.selectbox("Variable", list(PRESSURE))
    domain, code, vname, units, cmap = PRESSURE[choice]
    plevel = st.sidebar.selectbox("Pressure level (hPa)", COMMON_PLEVELS)

yr   = st.sidebar.selectbox("Year", YEARS)
mon  = st.sidebar.selectbox("Month", list(range(1,13)),
                            format_func=lambda m: ["Jan","Feb","Mar","Apr","May",
                                                   "Jun","Jul","Aug","Sep","Oct",
                                                   "Nov","Dec"][m-1])

show_coast = st.sidebar.checkbox("Show coastlines", value=True)

# ─────────────────────────────────────────────────────────────────────────────
#  URL BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def rda_url(year, domain, code, vname):
    head = ("https://thredds.rda.ucar.edu/thredds/dodsC/files/"
            "g/d633001_nc/")
    if domain == "sfc":
        tail = f"e5.moda.an.sfc.128_{code}_{vname}.ll025sc."
    else:                      # pressure levels
        uvflag = vname in {"u", "v"}
        tail = (f"e5.moda.an.pl.128_{code}_{vname}."
                f"ll025{'uv' if uvflag else 'sc'}.")
    return (f"{head}e5.moda.an.{domain}/{year}/"
            f"{tail}{year}010100_{year}120100.nc")

url = rda_url(yr, domain, code, vname)

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING (cached by year & URL)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def open_year(url):
    return xr.open_dataset(url, decode_times=False)

st.caption(f"Loading {choice} • {mon:02d}/{yr} …")
with st.spinner("Fetching NetCDF…"):
    ds = open_year(url)

# pick the requested month
# ── helper ──────────────────────────────────────────────────────────────
def find_var(ds, short):
    """Return the first matching variable name in the Dataset."""
    short_up = short.upper()
    for key in (short_up, f"VAR_{short_up}", short_up.replace("10", "10M")):
        if key in ds.variables:
            return key
    raise KeyError(f"No variable named {short} / VAR_{short} in file.")

# ── new extraction code ─────────────────────────────────────────────────
try:
    key = find_var(ds, vname)
except KeyError as e:
    st.error(str(e))
    st.stop()

da = ds[key].isel(time=mon-1)

# pressure-level slicing
if plevel is not None:
    # RDA uses 'level' in hPa
    da = da.sel(level=plevel)

# basic unit tweaks
if vname in {"sstk", "2t", "t"}:             # K → °C
    da = da - 273.15
    units = "°C"
if vname in {"sp", "msl"}:                   # Pa → hPa
    da = da / 100.0
    units = "hPa"

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────────────────────────────────────
##### Cached helper for coastlines
@st.cache_resource(show_spinner=False)
def coastlines_trace(res="110m"):
    """Return a Plotly Scatter trace with Natural-Earth coastlines."""
    feat = cfeature.NaturalEarthFeature(
        "physical", "coastline", res, edgecolor="black", facecolor="none"
    )

    xs, ys = [], []
    for geom in feat.geometries():
        # Each geometry can be MultiLineString or LineString → iterate over coords
        for line in getattr(geom, "geoms", [geom]):
            lon, lat = line.coords.xy
            # Shift −180…180 → 0…360 so it matches ERA5 grid
            lon = np.mod(lon, 360)
            xs.extend(lon.tolist() + [np.nan])
            ys.extend(lat.tolist() + [np.nan])

    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color="black", width=0.8),
        hoverinfo="skip",
        showlegend=False,
    )



title = f"{choice} • {mon:02d}/{yr}" + (f" • {plevel} hPa" if plevel else "")
fig = px.imshow(
    da,
    origin="lower", aspect="auto",
    color_continuous_scale=cmap,
    labels=dict(color=units),
    title=title
)

if show_coast:
  fig.add_trace(coastlines_trace())
st.plotly_chart(fig, use_container_width=True)

