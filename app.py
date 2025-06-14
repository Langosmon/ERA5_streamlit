# app.py  – ERA5 interactive maps with colour-bar sliders
import xarray as xr
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import cartopy.feature as cfeature
from pathlib import Path

st.set_page_config(layout="wide")

# ─────────── constants ────────────────────────────────────────────────────
YEARS          = list(range(1980, 2023))
COMMON_PLEVELS = [975, 850, 700, 500, 250, 100, 50, 10]
CLIM_DIR       = Path(__file__).with_name("climatology")

# (domain, code, vname, units, cmap_abs, cmap_anom)
SURFACE = {
    "Sea-surface temperature": ("sfc","034","sstk","°C",   "thermal","RdBu_r"),
    "CAPE"                   : ("sfc","059","cape","J kg⁻¹","viridis","PuOr"),
    "Surface geopotential"   : ("sfc","129","z","m² s⁻²",  "magma","RdBu_r"),
    "Surface pressure"       : ("sfc","134","sp","hPa",    "icefire","RdBu_r"),
    "Mean sea-level press."  : ("sfc","151","msl","hPa",   "icefire","RdBu_r"),
    "10-m zonal wind"        : ("sfc","165","10u","m s⁻¹", "curl","RdBu_r"),
    "10-m meridional wind"   : ("sfc","166","10v","m s⁻¹", "curl_r","RdBu_r"),
    "2-m temperature"        : ("sfc","167","2t","°C",     "thermal","RdBu_r"),
}

PRESSURE = {
    "Potential vorticity"  : ("pl","060","pv","PVU",      "plasma","RdBu_r"),
    "Geopotential"         : ("pl","129","z","m² s⁻²",    "magma","RdBu_r"),
    "Temperature"          : ("pl","130","t","K",         "thermal","RdBu_r"),
    "Zonal wind"           : ("pl","131","u","m s⁻¹",     "curl","RdBu_r"),
    "Meridional wind"      : ("pl","132","v","m s⁻¹",     "curl_r","RdBu_r"),
    "Specific humidity"    : ("pl","133","q","kg kg⁻¹",   "viridis","BrBG"),
    "Vertical velocity"    : ("pl","135","w","Pa s⁻¹",    "icefire","RdBu"),
    "Relative vorticity"   : ("pl","138","vo","s⁻¹",      "plasma","RdBu_r"),
    "Divergence"           : ("pl","155","d","s⁻¹",       "plasma","RdBu_r"),
    "Relative humidity"    : ("pl","157","r","%",         "viridis","BrBG"),
    "Ozone"                : ("pl","203","o3","kg kg⁻¹",  "viridis","RdBu_r"),
}

# ─────────── sidebar ─────────────────────────────────────────────────────
st.sidebar.header("Field")
field_type = st.sidebar.radio("Domain", ("Surface", "Pressure level"))

if field_type == "Surface":
    choice = st.sidebar.selectbox("Variable", list(SURFACE))
    domain, code, vname, units, cmap_abs, cmap_anom = SURFACE[choice]
    plevel = None
else:
    choice = st.sidebar.selectbox("Variable", list(PRESSURE))
    domain, code, vname, units, cmap_abs, cmap_anom = PRESSURE[choice]
    plevel = st.sidebar.selectbox("Pressure level (hPa)", COMMON_PLEVELS)

yr  = st.sidebar.selectbox("Year", YEARS)
mon = st.sidebar.selectbox(
    "Month", range(1, 13),
    format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])

show_anom  = st.sidebar.checkbox("Show anomaly (selected – climatology)")
show_coast = st.sidebar.checkbox("Show coastlines", value=True)

# ─────────── helpers ────────────────────────────────────────────────────
def rda_url(y, dom, code, var):
    base = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633001_nc/"
    if dom == "sfc":
        tail = f"e5.moda.an.sfc.128_{code}_{var}.ll025sc."
    else:
        extra = "uv" if var in {"u","v"} else "sc"
        tail  = f"e5.moda.an.pl.128_{code}_{var}.ll025{extra}."
    return f"{base}e5.moda.an.{dom}/{y}/{tail}{y}010100_{y}120100.nc"

@st.cache_resource
def open_year(url):
    return xr.open_dataset(url, decode_times=False)

def find_var(ds, short):
    up = short.upper()
    for k in (up, f"VAR_{up}", up.replace("10","10M")):
        if k in ds: return k
    raise KeyError(short)

@st.cache_resource
def load_clim(dom, var, lvl):
    path = (CLIM_DIR / ("sfc" if dom=="sfc" else "pl") /
            (f"{var}.nc" if dom=="sfc" else f"{var}_{lvl}.nc"))
    ds = xr.open_dataset(path)
    return ds[find_var(ds, var)]

@st.cache_resource
def coastlines_trace(res="110m", gap=10):
    xs, ys = [], []
    feat = cfeature.NaturalEarthFeature("physical","coastline", res,
                                        edgecolor="black", facecolor="none")
    for geom in feat.geometries():
        for line in getattr(geom,"geoms",[geom]):
            lon, lat = line.coords.xy
            lon = np.mod(lon,360)
            xs.append(np.nan); ys.append(np.nan)
            for i in range(len(lon)):
                xs.append(lon[i]); ys.append(lat[i])
                if i<len(lon)-1 and abs(lon[i+1]-lon[i])>gap:
                    xs.append(np.nan); ys.append(np.nan)
    return go.Scatter(x=xs,y=ys,mode="lines",
                      line=dict(color="black",width=0.8),
                      hoverinfo="skip",showlegend=False)

# ─────────── load monthly field ─────────────────────────────────────────
ds = open_year(rda_url(yr, domain, code, vname))
da = ds[find_var(ds, vname)].isel(time=mon-1)
if plevel is not None:
    da = da.sel(level=plevel)

# unit conversions
if vname in {"sstk","2t","t"}:   da, units = da - 273.15, "°C"
if vname in {"sp","msl"}:        da, units = da / 100.0, "hPa"

# anomaly
cmap = cmap_abs
if show_anom:
    clim = load_clim(domain, vname, plevel)
    da   = da - clim.sel(month=mon)
    cmap = cmap_anom
    units += " anomaly"

# ─────────── colour-bar controls ──────────────────────────────────────────
data_min = float(np.nanmin(da))
data_max = float(np.nanmax(da))

if show_anom:
    # symmetric default for anomalies
    default_max = float(np.nanmax(np.abs(da)))
    default_min = -default_max
else:
    default_min, default_max = data_min, data_max

# Scientific-notation helper for µ-scale variables
def sci(v):
    if abs(v) < 1e-3:
        return f"{v*1e6:,.0f} µ"
    if abs(v) < 1:
        return f"{v*1e3:,.0f} m"
    return f"{v:,.2f}"

st.sidebar.markdown("### Colour-bar limits")

step = (default_max - default_min) / 50 or 1e-6   # avoid step=0

cmin = st.sidebar.slider(
    "Min", data_min, data_max, value=default_min,
    step=step, format="%.4g"
)
cmax = st.sidebar.slider(
    "Max", data_min, data_max, value=default_max,
    step=step, format="%.4g"
)

# Auto-scale button (98 % central quantile)
if st.sidebar.button("Auto-scale (98 % of data)"):
    qmin, qmax = np.nanquantile(da, [0.01, 0.99])
    cmin, cmax = float(qmin), float(qmax)

if cmin >= cmax:
    st.sidebar.error("Min must be less than Max")
    st.stop()

# ─────────── plot ──────────────────────────────────────────────────────
title = f"{choice} • {mon:02d}/{yr}" + (f" • {plevel} hPa" if plevel else "")
if show_anom: title += " • anomaly"

fig = px.imshow(
    da,
    origin="lower", aspect="auto",
    color_continuous_scale=cmap,
    labels=dict(color=units),
    title=title
)
fig.update_coloraxes(cmin=cmin, cmax=cmax)
fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), uirevision="keep")

if show_coast:
    fig.add_trace(coastlines_trace())

st.plotly_chart(fig, use_container_width=True)




