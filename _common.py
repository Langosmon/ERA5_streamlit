"""Shared helpers for the ERA5 Streamlit apps.

This module is duplicated between ERA5_streamlit and ERA5_hourly_streamlit so
each deploys independently to Streamlit Cloud. Keep them in sync.

Provides:
  - SURFACE, PRESSURE, COMMON_PLEVELS variable catalogues
  - coastlines_trace(): fast, no-cartopy coastline overlay
  - find_var(): tolerant ERA5 variable name lookup
  - open_dataset_cached(): cached remote xarray dataset opener
  - load_climatology(): fetches climatology .nc from GitHub Releases on demand
  - apply_unit_conversions(): K→°C, Pa→hPa
  - build_figure(): site-themed Plotly figure
  - significance_stipple(): hatch-mask for not-significant regions
  - configure_page() / render_footer(): site-branded UI shell
"""

from __future__ import annotations
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import streamlit as st
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go


# ─────────── catalogues ───────────────────────────────────────────────────────
# (domain, code, vname, units, cmap_abs, cmap_anom)
SURFACE: dict[str, tuple] = {
    "Sea-surface temperature": ("sfc", "034", "sstk", "°C",     "thermal", "RdBu_r"),
    "CAPE":                    ("sfc", "059", "cape", "J kg⁻¹", "viridis", "PuOr"),
    "Surface geopotential":    ("sfc", "129", "z",    "m² s⁻²", "magma",   "RdBu_r"),
    "Surface pressure":        ("sfc", "134", "sp",   "hPa",    "icefire", "RdBu_r"),
    "Mean sea-level press.":   ("sfc", "151", "msl",  "hPa",    "icefire", "RdBu_r"),
    "10-m zonal wind":         ("sfc", "165", "10u",  "m s⁻¹",  "curl",    "RdBu_r"),
    "10-m meridional wind":    ("sfc", "166", "10v",  "m s⁻¹",  "curl_r",  "RdBu_r"),
    "2-m temperature":         ("sfc", "167", "2t",   "°C",     "thermal", "RdBu_r"),
}

PRESSURE: dict[str, tuple] = {
    "Potential vorticity":   ("pl", "060", "pv", "PVU",     "plasma",  "RdBu_r"),
    "Geopotential":          ("pl", "129", "z",  "m² s⁻²",  "magma",   "RdBu_r"),
    "Temperature":           ("pl", "130", "t",  "K",       "thermal", "RdBu_r"),
    "Zonal wind":            ("pl", "131", "u",  "m s⁻¹",   "curl",    "RdBu_r"),
    "Meridional wind":       ("pl", "132", "v",  "m s⁻¹",   "curl_r",  "RdBu_r"),
    "Specific humidity":     ("pl", "133", "q",  "kg kg⁻¹", "viridis", "BrBG"),
    "Vertical velocity":     ("pl", "135", "w",  "Pa s⁻¹",  "icefire", "RdBu"),
    "Relative vorticity":    ("pl", "138", "vo", "s⁻¹",     "plasma",  "RdBu_r"),
    "Divergence":            ("pl", "155", "d",  "s⁻¹",     "plasma",  "RdBu_r"),
    "Relative humidity":     ("pl", "157", "r",  "%",       "viridis", "BrBG"),
    "Ozone":                 ("pl", "203", "o3", "kg kg⁻¹", "viridis", "RdBu_r"),
}

COMMON_PLEVELS: list[int] = [975, 850, 700, 500, 250, 100, 50, 10]


# ─────────── REMOTE CLIMATOLOGY ──────────────────────────────────────────────
# Climatology .nc files live on a GitHub Release attached to the ERA5_streamlit
# repo (so both apps can pull from the same place). One release tag holds all
# 96 files. URLs are of the form:
#     {CLIM_BASE}/{tag}/{sfc|pl}__{var}[_<lvl>].nc
# (double-underscore separator because GitHub Releases flatten paths).
CLIM_BASE = "https://github.com/Langosmon/ERA5_streamlit/releases/download"
CLIM_TAG  = "climatology-v1"   # bump if you reupload regenerated files


def _clim_remote_url(domain: str, var: str, lvl: Optional[int]) -> str:
    """URL of a single climatology file in the GitHub Release."""
    if domain == "sfc":
        name = f"sfc__{var}.nc"
    else:
        name = f"pl__{var}_{lvl}.nc"
    return f"{CLIM_BASE}/{CLIM_TAG}/{name}"


def _clim_local_path(domain: str, var: str, lvl: Optional[int]) -> Path:
    """Local cache path for a downloaded climatology file."""
    cache = Path("/tmp/era5-climatology")
    cache.mkdir(parents=True, exist_ok=True)
    return cache / (f"sfc__{var}.nc" if domain == "sfc" else f"pl__{var}_{lvl}.nc")


@st.cache_resource(show_spinner="Fetching climatology…")
def load_climatology(domain: str, var: str, lvl: Optional[int]):
    """Lazily fetch a climatology file from GitHub Releases on first use,
    cache to /tmp, then memoize the loaded DataArray (in memory) for the
    rest of the session. Returns a DataArray with `.load()` already called
    so all subsequent slicing is in-memory and instant."""
    local = _clim_local_path(domain, var, lvl)

    if not local.exists():
        url = _clim_remote_url(domain, var, lvl)
        try:
            r = requests.get(url, stream=True, timeout=30)
            if r.status_code != 200:
                raise FileNotFoundError(
                    f"Climatology not yet uploaded to release '{CLIM_TAG}'.\n"
                    f"URL: {url}\nStatus: {r.status_code}"
                )
            with open(local, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            raise FileNotFoundError(f"Could not reach climatology host: {e}") from e

    ds = xr.open_dataset(local)
    da = ds[find_var(ds, var)]
    # Load eagerly so .sel(month=...) below is a free in-memory operation.
    return da.load()


def climatology_has_std(domain: str, var: str, lvl: Optional[int]) -> bool:
    """Check if the climatology file includes an 'std' variable (needed for
    statistical-significance tests). Returns False if file missing OR std
    variable not present."""
    try:
        local = _clim_local_path(domain, var, lvl)
        if not local.exists():
            return False
        ds = xr.open_dataset(local)
        for k in ds.variables:
            if "std" in k.lower() or "stddev" in k.lower() or "sigma" in k.lower():
                return True
        return False
    except Exception:
        return False


def load_climatology_std(domain: str, var: str, lvl: Optional[int]):
    """Return the std-dev DataArray from the climatology file, if present."""
    local = _clim_local_path(domain, var, lvl)
    ds = xr.open_dataset(local)
    for k in ds.variables:
        if "std" in k.lower() or "stddev" in k.lower() or "sigma" in k.lower():
            return ds[k]
    raise KeyError("std variable not found in climatology file")


# ─────────── data access ─────────────────────────────────────────────────────
def find_var(ds: xr.Dataset, short: str) -> str:
    """Return the actual variable name in a dataset for an ERA5 short code.
    Tolerant of VAR_ prefixes and 10 ↔ 10M substitutions."""
    up = short.upper()
    for k in (up, f"VAR_{up}", up.replace("10", "10M")):
        if k in ds:
            return k
    raise KeyError(short)


@st.cache_resource(show_spinner="Opening remote dataset…")
def open_dataset_cached(url: str, decode_times: bool = True) -> xr.Dataset:
    return xr.open_dataset(url, decode_times=decode_times)


@st.cache_data(show_spinner="Loading from RDA…", max_entries=24, ttl=3600)
def load_field_cached(url: str, vname: str, plevel: Optional[int],
                      decode_times: bool = True) -> xr.DataArray:
    """Open a remote dataset, slice the requested variable (and pressure
    level), and EAGERLY LOAD the data into memory.  Cached by Streamlit so
    subsequent calls with the same args return instantly (no RDA round-trip).

    This is the biggest perceived-speed win: changing the *month* (monthly
    app) or *hour* (hourly app) on the same (year, var, plevel) becomes
    in-memory slicing instead of a fresh OPeNDAP request.

    max_entries=24 caps memory at roughly 24 × 50 MB ≈ 1.2 GB which is
    Streamlit Cloud's free-tier ceiling.  TTL=1h trims stale entries."""
    ds = xr.open_dataset(url, decode_times=decode_times)
    da = ds[find_var(ds, vname)]
    if plevel is not None:
        # Server-side slice — OPeNDAP only transfers the one level.
        da = da.sel(level=plevel)
    # Force the actual download now; subsequent operations are in-memory.
    return da.load()


# ─────────── ERA5 LAND-SEA MASK ──────────────────────────────────────────────
# Static field, ~16 MB, fetched once from RDA invariants. Values: 1=land,
# 0=sea, fractional at coasts.
LSM_URL = ("https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/"
           "e5.oper.invariant/197901/e5.oper.invariant.128_172_lsm.ll025sc."
           "1979010100_1979010100.nc")


@st.cache_resource(show_spinner="Fetching land-sea mask…")
def load_lsm() -> xr.DataArray:
    """Load ERA5 land-sea mask (0=sea, 1=land, fractional coasts).
    On the same 0.25° grid as the data, so .where() works directly."""
    ds = xr.open_dataset(LSM_URL)
    da = ds[find_var(ds, "lsm")].squeeze()
    return da.load()


def apply_lsm_mask(da: xr.DataArray, mode: str) -> xr.DataArray:
    """Apply a land-sea mask to a DataArray.  mode is one of:
       "All" / "Land" / "Ocean".  Returns the masked DataArray."""
    if mode == "All":
        return da
    lsm = load_lsm()
    # Make sure dims match — both should be (latitude, longitude) at 0.25°.
    # If lat orderings differ, xarray's .where() with broadcasted comparison
    # would fail; reindex defensively.
    try:
        lsm_aligned = lsm.reindex_like(da, method="nearest", tolerance=0.01)
    except Exception:
        lsm_aligned = lsm  # fall through; .where() will broadcast best-effort
    if mode == "Land":
        return da.where(lsm_aligned >= 0.5)
    if mode == "Ocean":
        return da.where(lsm_aligned < 0.5)
    return da


# ─────────── REGION-AWARE COLORBAR AUTOSCALE ─────────────────────────────────
# When the user box-selects a region, recompute colorbar from the 98% quantile
# of data INSIDE that box. Solves the "Andes/Himalayas spikes dominate the
# scale, ITCZ looks washed out" problem.
def rescale_to_region(da: xr.DataArray,
                      lat_min: float, lat_max: float,
                      lon_min: float, lon_max: float,
                      quantile_lo: float = 0.01,
                      quantile_hi: float = 0.99,
                      symmetric: bool = False) -> tuple[float, float]:
    """Return (cmin, cmax) covering quantile_lo..quantile_hi of the data
    inside the lat/lon box. Symmetric=True returns ±max(|q01|,|q99|), useful
    for anomalies. Falls back to global quantiles if the box is empty."""
    sub = da.where(
        (da.latitude >= lat_min) & (da.latitude <= lat_max) &
        (da.longitude >= lon_min) & (da.longitude <= lon_max)
    )
    vals = sub.values
    if np.all(np.isnan(vals)):
        vals = da.values
    qlo, qhi = np.nanquantile(vals, [quantile_lo, quantile_hi])
    if symmetric:
        m = max(abs(float(qlo)), abs(float(qhi)))
        return -m, m
    return float(qlo), float(qhi)


# Quick-pick region presets — common atmospheric/ocean basins.
# Stored as (lat_min, lat_max, lon_min, lon_max) in -180..180 longitude.
# (When ERA5 uses 0..360, longitudes here are converted at use-time.)
REGION_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "Global":         (-90, 90, -180, 180),
    "Tropics":        (-30, 30, -180, 180),
    "ITCZ band":      (-15, 15, -180, 180),
    "E Pacific (ENP)": (5, 30, -130, -85),
    "Atlantic":        (5, 35, -85, -10),
    "W Pacific":      (5, 35, 110, 180),
    "Indian Ocean":  (-30, 30, 40, 105),
    "N America":     (15, 70, -170, -50),
    "Europe":         (35, 72, -15, 50),
}


# ─────────── coastlines (no cartopy) ─────────────────────────────────────────
@st.cache_resource
def coastlines_trace() -> go.Scatter:
    path = Path(__file__).with_name("coastlines.json")
    data = json.loads(path.read_text())
    return go.Scatter(
        x=data["xs"], y=data["ys"],
        mode="lines",
        line=dict(color="rgba(20,20,20,0.65)", width=0.7),
        hoverinfo="skip",
        showlegend=False,
        name="coast",
    )


# ─────────── unit conversions ────────────────────────────────────────────────
def apply_unit_conversions(da: xr.DataArray, vname: str, units: str) -> tuple[xr.DataArray, str]:
    if vname in {"sstk", "2t", "t"}:
        da = da - 273.15
        units = "°C"
    if vname in {"sp", "msl"}:
        da = da / 100.0
        units = "hPa"
    return da, units


# ─────────── colour-bar controls ─────────────────────────────────────────────
def colourbar_controls(da: xr.DataArray, show_anom: bool,
                      override_default: Optional[tuple[float, float]] = None,
                      override_label: Optional[str] = None):
    """Sidebar sliders + auto-scale button for the colour-bar.

    `override_default`: if provided, these (cmin, cmax) become the slider
    defaults — used by the region-preset picker and the box-select rescale.
    The slider RANGE still spans the full data so the user can pan beyond
    the preset.
    `override_label`: optional small caption shown under the controls."""
    data_min = float(np.nanmin(da))
    data_max = float(np.nanmax(da))

    if override_default is not None:
        default_min, default_max = override_default
    elif show_anom:
        m = float(np.nanmax(np.abs(da)))
        default_min, default_max = -m, m
    else:
        default_min, default_max = data_min, data_max

    # Slider widget range covers the entire data PLUS the override so the
    # default is always inside the slider range.
    slider_min = min(data_min, default_min)
    slider_max = max(data_max, default_max)
    step = (slider_max - slider_min) / 50 or 1e-6

    with st.sidebar.expander("Colour-bar limits", expanded=False):
        if override_label:
            st.caption(override_label)
        cmin = st.slider("Min", slider_min, slider_max, value=default_min,
                         step=step, format="%.4g", key="cmin")
        cmax = st.slider("Max", slider_min, slider_max, value=default_max,
                         step=step, format="%.4g", key="cmax")
        if st.button("Auto-scale (98 % of data)", use_container_width=True):
            qmin, qmax = np.nanquantile(da, [0.01, 0.99])
            st.session_state["cmin"] = float(qmin)
            st.session_state["cmax"] = float(qmax)
            st.rerun()

    if cmin >= cmax:
        st.sidebar.error("Min must be less than Max")
        st.stop()
    return cmin, cmax


# ─────────── region picker (sidebar) ─────────────────────────────────────────
def region_picker():
    """Render the region preset selector. Returns the chosen region tuple
    (lat_min, lat_max, lon_min, lon_max) or None for Global / no rescale."""
    st.sidebar.header("Region focus")
    name = st.sidebar.selectbox(
        "Rescale colour-bar to region",
        list(REGION_PRESETS.keys()),
        index=0,
        help="Tunes the colour-bar to the 98% quantile of data inside the "
             "chosen region. The map still shows the whole field — values "
             "outside this range are clipped to the bar ends. "
             "Hint: use Plotly's box-select tool to define a custom region.",
        key="region_name",
    )
    if name == "Global":
        return None, name
    return REGION_PRESETS[name], name


def box_selection_to_bounds(event: Optional[dict]) -> Optional[tuple[float, float, float, float]]:
    """Extract (lat_min, lat_max, lon_min, lon_max) from a Streamlit
    plotly_chart on_select event. Returns None if no box."""
    if not event or "selection" not in event:
        return None
    boxes = event.get("selection", {}).get("box") or []
    if not boxes:
        return None
    b = boxes[0]
    xs = b.get("x") or []
    ys = b.get("y") or []
    if len(xs) < 2 or len(ys) < 2:
        return None
    return (float(min(ys)), float(max(ys)), float(min(xs)), float(max(xs)))


# ─────────── statistical significance ────────────────────────────────────────
def significance_mask(anom: xr.DataArray, std: xr.DataArray, z: float = 1.96) -> np.ndarray:
    """Return a boolean array True where |anom| >= z·std (i.e. significant).
    Default z=1.96 ≈ 95% confidence assuming year-to-year normality of the
    climatology."""
    return (np.abs(anom.values) >= z * np.abs(std.values))


def add_significance_stipple(fig: go.Figure, anom: xr.DataArray,
                             std: xr.DataArray, z: float = 1.96,
                             stride: int = 8) -> None:
    """Overlay sparse dots where the anomaly IS significant. Stride controls
    density (every Nth grid point), so we don't bury the map in markers."""
    sig = significance_mask(anom, std, z=z)
    lats = anom.latitude.values
    lons = anom.longitude.values
    ys, xs = np.where(sig)
    # subsample
    keep = (ys % stride == 0) & (xs % stride == 0)
    ys, xs = ys[keep], xs[keep]
    fig.add_trace(go.Scatter(
        x=lons[xs], y=lats[ys],
        mode="markers",
        marker=dict(size=2, color="rgba(0,0,0,0.55)", symbol="circle"),
        hoverinfo="skip", showlegend=False, name=f"sig (|z|≥{z})",
    ))


# ─────────── figure builder ──────────────────────────────────────────────────
def build_figure(da: xr.DataArray, title: str, units: str, cmap: str,
                 cmin: float, cmax: float, show_coast: bool,
                 height: int = 560) -> go.Figure:
    fig = px.imshow(
        da,
        origin="lower",
        aspect="auto",
        color_continuous_scale=cmap,
        labels=dict(color=units),
        title=title,
    )
    fig.update_coloraxes(cmin=cmin, cmax=cmax,
                         colorbar=dict(thickness=12, len=0.85, x=1.01,
                                       outlinewidth=0))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=46, b=0),
        uirevision="keep",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=title,
            x=0.0, xanchor="left",
            font=dict(size=18, color="#d97757",
                      family="Source Serif 4, Georgia, serif"),
        ),
        xaxis=dict(title="", showgrid=False, zeroline=False, ticks="outside"),
        yaxis=dict(title="", showgrid=False, zeroline=False, ticks="outside"),
    )
    if show_coast:
        fig.add_trace(coastlines_trace())
    return fig


# ─────────── presentation ────────────────────────────────────────────────────
SITE_URL = "https://langosmon.github.io"


def configure_page(title: str, subtitle: str | None = None,
                   icon: str = "🌀") -> None:
    st.set_page_config(
        page_title=f"{title} · Ocegueda Sanchez",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get help": SITE_URL,
            "About": f"{title} — Jose A. Ocegueda Sanchez ({SITE_URL}). "
                     f"ERA5 data via NCAR's RDA. Source on GitHub.",
        },
    )
    st.markdown(
        f"""
        <div style='display:flex; justify-content:space-between; align-items:baseline;
                    border-bottom:1px solid rgba(128,128,128,0.18); padding-bottom:6px;
                    margin-bottom:8px;'>
          <div>
            <div style='font-family:"JetBrains Mono",monospace; font-size:11px;
                        letter-spacing:0.16em; text-transform:uppercase;
                        color:#d97757;'>ERA5 · NCAR RDA</div>
            <div style='font-family:"Source Serif 4",Georgia,serif; font-size:24px;
                        line-height:1.1; margin-top:4px;'>{title}</div>
            {('<div style="font-size:13px; color:#888; margin-top:4px;">'
              f'{subtitle}</div>') if subtitle else ''}
          </div>
          <div style='font-family:"JetBrains Mono",monospace; font-size:10px;
                      letter-spacing:0.08em; color:#888;'>
            <a href='{SITE_URL}' target='_blank' style='color:#d97757;
                text-decoration:none;'>langosmon.github.io ↗</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer(repo_url: str) -> None:
    st.markdown(
        f"""
        <div style='margin-top:32px; padding-top:14px;
                    border-top:1px solid rgba(128,128,128,0.18);
                    font-size:12px; color:#888; display:flex; gap:16px;
                    flex-wrap:wrap; justify-content:space-between;'>
          <span>Plot built from ERA5 via NCAR's
            <a href='https://rda.ucar.edu/' target='_blank' style='color:#d97757;'>RDA</a>.
          </span>
          <span>
            <a href='{repo_url}' target='_blank' style='color:#d97757; text-decoration:none;'>source ↗</a>
            ·
            <a href='{SITE_URL}' target='_blank' style='color:#d97757; text-decoration:none;'>site ↗</a>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────── reusable sidebar widget ─────────────────────────────────────────
def variable_picker():
    st.sidebar.header("Field")
    field_type = st.sidebar.radio("Domain", ("Surface", "Pressure level"),
                                  horizontal=True, key="field_type")
    if field_type == "Surface":
        choice = st.sidebar.selectbox("Variable", list(SURFACE), key="surface_var")
        domain, code, vname, units, cmap_abs, cmap_anom = SURFACE[choice]
        plevel = None
    else:
        choice = st.sidebar.selectbox("Variable", list(PRESSURE), key="pressure_var")
        domain, code, vname, units, cmap_abs, cmap_anom = PRESSURE[choice]
        plevel = st.sidebar.selectbox("Pressure level (hPa)", COMMON_PLEVELS,
                                      index=COMMON_PLEVELS.index(500),
                                      key="plevel")
    return choice, domain, code, vname, units, cmap_abs, cmap_anom, plevel
