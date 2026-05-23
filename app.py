"""ERA5 monthly means + anomalies, interactive maps.

Climatology files live on a GitHub Release attached to this repo and are
fetched lazily on first anomaly request.  See _common.CLIM_TAG.
"""

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
                               help="Stipples points where |anomaly| ≥ 1.96·σ "
                                    "(year-to-year std dev of climatology). "
                                    "Requires std-dev climatology — see repo README.")
show_coast = st.sidebar.toggle("Coastlines", value=True)


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
    ds = C.open_dataset_cached(rda_url(yr, domain, code, vname), decode_times=False)
    da = ds[C.find_var(ds, vname)].isel(time=mon - 1)
    if plevel is not None:
        da = da.sel(level=plevel)
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
anom_da = None
clim_std = None

if show_anom:
    try:
        clim = C.load_climatology(domain, vname, plevel)
        clim_month = clim.sel(month=mon)
        anom_da = da - clim_month
        da = anom_da
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


# ─── colour-bar + figure ─────────────────────────────────────────────────────
cmin, cmax = C.colourbar_controls(da, show_anom)

title = f"{choice} · {mon:02d}/{yr}"
if plevel is not None: title += f" · {plevel} hPa"
if show_anom: title += " · anomaly"

fig = C.build_figure(da, title, units, cmap, cmin, cmax, show_coast, height=580)

if show_anom and show_sig and clim_std is not None:
    C.add_significance_stipple(fig, da, clim_std, z=1.96, stride=8)

st.plotly_chart(fig, use_container_width=True,
                config={"displaylogo": False,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

# Quick-pick example dates relevant to tropical dynamics
with st.expander("Quick picks — notable months", expanded=False):
    st.caption("Common cases worth looking at:")
    st.markdown(
        "- **Oct 2023** — Hurricane Otis intensification month (try SST anomaly)\n"
        "- **Aug 1992** — Hurricane Andrew, peak Atlantic season\n"
        "- **Jan 1998** — strong El Niño peak (try SST anomaly)\n"
        "- **Jun 1991** — Pinatubo aftermath (try 2-m temp anomaly)"
    )

C.render_footer(REPO_URL)
