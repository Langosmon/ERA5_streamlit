"""ERA5 monthly means + anomalies, interactive maps.

Climatology files live on a GitHub Release attached to this repo and are
fetched lazily on first anomaly request.  See _common.CLIM_TAG.
"""

import numpy as np
import xarray as xr
import streamlit as st

import _common as C


# ─── page setup ───────────────────────────────────────────────────────────────
C.configure_page(
    title="ERA5 · Monthly Maps",
    subtitle="Monthly means + anomalies, 1980–2022. Surface fields and pressure levels.",
    icon="🌀",
)

REPO_URL = "https://github.com/Langosmon/ERA5_streamlit"
YEARS = list(range(1980, 2023))


# ─── sidebar controls ────────────────────────────────────────────────────────
choice, domain, code, vname, units, cmap_abs, cmap_anom, plevel = C.variable_picker()

st.sidebar.header("Date")
col_y, col_m = st.sidebar.columns(2)
yr  = col_y.selectbox("Year", YEARS, index=len(YEARS) - 1)
mon = col_m.selectbox(
    "Month", range(1, 13),
    format_func=lambda m: ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1],
    index=0,
)

st.sidebar.header("Display")
show_anom  = st.sidebar.toggle("Anomaly (value − climatology)", value=False)
show_sig   = st.sidebar.toggle("Mark statistically significant", value=False,
                               disabled=not show_anom,
                               help="Stipples points where |anomaly| ≥ 1.96·σ. "
                                    "Requires std-dev climatology — see repo README.")
show_coast = st.sidebar.toggle("Coastlines", value=True)

# Land-sea mask
mask_mode = st.sidebar.radio(
    "Show data on", ("All", "Land", "Ocean"),
    horizontal=True,
    help="Masks the field by ERA5's land-sea mask. Land/Ocean show only "
         "the side you pick.",
)


# ─── data loading ────────────────────────────────────────────────────────────
def rda_url(year: int, dom: str, code: str, var: str) -> str:
    base = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633001_nc/"
    if dom == "sfc":
        tail = f"e5.moda.an.sfc.128_{code}_{var}.ll025sc."
    else:
        extra = "uv" if var in {"u", "v"} else "sc"
        tail  = f"e5.moda.an.pl.128_{code}_{var}.ll025{extra}."
    return f"{base}e5.moda.an.{dom}/{year}/{tail}{year}010100_{year}120100.nc"


try:
    full_year = C.load_field_cached(
        rda_url(yr, domain, code, vname), vname, plevel, decode_times=False,
    )
    da = full_year.isel(time=mon - 1)
except Exception as e:
    st.error(
        "**Failed to load remote ERA5 data.**  "
        "The RDA server may be temporarily unreachable, or this "
        "year/variable/level combination may not exist."
    )
    st.exception(e)
    st.stop()

da, units = C.apply_unit_conversions(da, vname, units)


# ─── anomaly + significance ──────────────────────────────────────────────────
cmap = cmap_abs
clim_std = None

if show_anom:
    try:
        clim = C.load_climatology(domain, vname, plevel)
        clim_month = clim.sel(month=mon)
        da = da - clim_month
        cmap = cmap_anom
        units = (units or "") + " anomaly"

        if show_sig:
            if C.climatology_has_std(domain, vname, plevel):
                std_full = C.load_climatology_std(domain, vname, plevel)
                clim_std = std_full.sel(month=mon)
            else:
                st.info(
                    "**Significance:** the current climatology file only "
                    "contains the mean, not the year-to-year standard "
                    "deviation. Regenerate using "
                    "`tools/build_climatology.py` and re-upload to the "
                    "GitHub Release to enable this overlay."
                )
                show_sig = False
    except FileNotFoundError as e:
        st.warning(f"Climatology unavailable for this field — showing absolute value instead.\n\n{e}")
        show_anom = False


# ─── land / sea mask ─────────────────────────────────────────────────────────
if mask_mode != "All":
    try:
        da = C.apply_lsm_mask(da, mask_mode)
        if clim_std is not None:
            clim_std = C.apply_lsm_mask(clim_std, mask_mode)
    except Exception as e:
        st.warning(f"Couldn't load land-sea mask — showing unmasked field.\n\n{e}")


# ─── region picker + box-select autoscale ────────────────────────────────────
region_bbox, region_name = C.region_picker()

# Decide colour-bar defaults: precedence is BOX-SELECT > REGION PRESET > DEFAULT.
override_default = None
override_label = None

# Persist the last applied box across reruns so colour-bar stays tuned.
last_box = st.session_state.get("_last_box")

if last_box is not None:
    lat_min, lat_max, lon_min, lon_max = last_box
    # Convert lon if ERA5 stores 0..360 (which it does for these files)
    if da.longitude.max() > 180 and lon_min < 0:
        lon_min, lon_max = (lon_min + 360) % 360, (lon_max + 360) % 360
    qlo, qhi = C.rescale_to_region(da, lat_min, lat_max, lon_min, lon_max,
                                   symmetric=show_anom)
    override_default = (qlo, qhi)
    override_label = (f"Tuned to box: {lat_min:.1f}–{lat_max:.1f}°N, "
                      f"{lon_min:.1f}–{lon_max:.1f}°E")
elif region_bbox is not None:
    lat_min, lat_max, lon_min, lon_max = region_bbox
    if da.longitude.max() > 180:
        lon_min, lon_max = (lon_min + 360) % 360, (lon_max + 360) % 360
    qlo, qhi = C.rescale_to_region(da, lat_min, lat_max, lon_min, lon_max,
                                   symmetric=show_anom)
    override_default = (qlo, qhi)
    override_label = f"Tuned to 98% of data in: {region_name}"

cmin, cmax = C.colourbar_controls(da, show_anom,
                                  override_default=override_default,
                                  override_label=override_label)


# ─── figure ──────────────────────────────────────────────────────────────────
title = f"{choice} · {mon:02d}/{yr}"
if plevel is not None: title += f" · {plevel} hPa"
if show_anom: title += " · anomaly"
if mask_mode != "All": title += f" · {mask_mode.lower()} only"

fig = C.build_figure(da, title, units, cmap, cmin, cmax, show_coast, height=580)

if show_anom and show_sig and clim_std is not None:
    C.add_significance_stipple(fig, da, clim_std, z=1.96, stride=8)

# Wire box-select on the plot for "rescale colour-bar to this region".
event = st.plotly_chart(
    fig, use_container_width=True,
    on_select="rerun", selection_mode=("box",),
    key="main_plot",
    config={"displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d"]},
)

# Did the user just box-select? Update session state and rerun once.
new_box = C.box_selection_to_bounds(event)
if new_box is not None and new_box != last_box:
    st.session_state["_last_box"] = new_box
    st.rerun()

# Tip + reset
col_t, col_r = st.columns([4, 1])
with col_t:
    st.caption(
        "💡 **Box-select tool** in the Plotly toolbar (dashed-rectangle "
        "icon, top-right of the plot) → draw a region → colour-bar "
        "rescales to the 98% quantile of that box. Useful for ITCZ-style "
        "work when topography dominates the global range."
    )
with col_r:
    if last_box is not None and st.button("Reset region", use_container_width=True):
        st.session_state["_last_box"] = None
        st.rerun()


with st.expander("Quick picks — notable months", expanded=False):
    st.caption("Common cases worth looking at:")
    st.markdown(
        "- **Oct 2023** — Hurricane Otis intensification month (try SST anomaly)\n"
        "- **Aug 1992** — Hurricane Andrew, peak Atlantic season\n"
        "- **Jan 1998** — strong El Niño peak (try SST anomaly)\n"
        "- **Jun 1991** — Pinatubo aftermath (try 2-m temp anomaly)"
    )

C.render_footer(REPO_URL)
