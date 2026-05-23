# ERA5_streamlit

Interactive Streamlit app for ERA5 monthly maps + anomalies, reading data from
NCAR's RDA via OPeNDAP. Companion to
[ERA5_hourly_streamlit](https://github.com/Langosmon/ERA5_hourly_streamlit).

Live: https://era5app-alfredocegueda.streamlit.app/

## Features

- 8 surface fields + 11 pressure-level fields, 8 standard pressure levels
- Monthly means **1980–2022**
- **Anomaly** toggle: departure from the 1980–2010 monthly climatology
- **Statistical significance** overlay: stipples points where |anomaly| ≥ 1.96·σ
  (requires std-dev climatology — see *Regenerate climatology* below)
- Coastline overlay (no cartopy dependency)
- Plotly toolbar with one-click PNG download

## Architecture

```
app.py            Streamlit app
_common.py        Shared helpers (also copied into ERA5_hourly_streamlit)
coastlines.json   Precomputed Natural Earth 110m coastlines (no cartopy)
tools/
  build_climatology.py    Regenerate mean+std climatology files from RDA
.streamlit/
  config.toml     Theme tokens matching langosmon.github.io
```

**Climatology files are NOT in the repo.** They live on a GitHub Release
attached to this repo (tag: `climatology-v1`). The apps lazily fetch them on
first anomaly request and cache them in `/tmp` for the lifetime of the
Streamlit Cloud container.

If you ever need to re-host or refresh them, see *Regenerate climatology*.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Anomaly + significance work the same locally as on Streamlit Cloud — files are
fetched from the GitHub Release on first use.

## Regenerate climatology (with std dev for significance)

The climatology files distributed in release `climatology-v1` contain MEAN
ONLY. To enable the statistical-significance overlay, regenerate them with
year-to-year standard deviation included:

```bash
# This takes hours — reads ERA5 monthly data from RDA via OPeNDAP for the
# 1980–2010 base period and writes new netCDF files locally.
python tools/build_climatology.py --out climatology --years 1980-2010 --skip-existing
```

Then upload the new files to a NEW release tag and bump `CLIM_TAG` in
`_common.py`:

```bash
gh release create climatology-v2 climatology/*.nc \
   --title "Climatology v2 (mean + std)" \
   --notes "1980-2010 monthly mean + year-to-year std dev. Replaces v1 (mean only)."
```

Streamlit Cloud will pick up the bumped tag on the next deploy.

## License & Citation

Apache License 2.0. If you use this in academic work, please cite:

> Jose A. Ocegueda Sanchez. *ERA5 Streamlit.* https://github.com/Langosmon/ERA5_streamlit

See the `NOTICE` file for additional attribution guidelines.

## Contact

- jocegue@purdue.edu
- [LinkedIn](https://www.linkedin.com/in/josé-alfredo-ocegueda-sanchez-a3598b122/)
- Personal site: https://langosmon.github.io
