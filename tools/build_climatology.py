"""Regenerate climatology files with MEAN + STANDARD DEVIATION across years.

Reads ERA5 monthly data from NCAR's RDA via OPeNDAP, computes per-month
mean and year-to-year std-dev across a user-chosen base period (default
1980–2010), and writes new netCDF files matching the structure expected
by the apps.

Usage:
    python tools/build_climatology.py --out climatology --years 1980-2010

The resulting files have the same shape as before (month, lat, lon) plus
a new {VAR}_STD variable with the year-to-year standard deviation.

Run this ONCE on a machine with the climatology files mounted or with a
fast connection to RDA. Expect this to take hours; results are deterministic
so safe to resume by --skip-existing.

After regeneration, upload the new climatology to the GitHub Release tag
configured in `_common.py` (CLIM_TAG). Bump the tag to invalidate caches.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr


# Variable catalogues (kept in sync with _common.SURFACE / _common.PRESSURE)
SURFACE = {  # short → (code, ll-suffix)
    "sstk": ("034", "sc"),
    "cape": ("059", "sc"),
    "z":    ("129", "sc"),
    "sp":   ("134", "sc"),
    "msl":  ("151", "sc"),
    "10u":  ("165", "sc"),
    "10v":  ("166", "sc"),
    "2t":   ("167", "sc"),
}

PRESSURE = {
    "pv": ("060", "sc"),
    "z":  ("129", "sc"),
    "t":  ("130", "sc"),
    "u":  ("131", "uv"),
    "v":  ("132", "uv"),
    "q":  ("133", "sc"),
    "w":  ("135", "sc"),
    "vo": ("138", "sc"),
    "d":  ("155", "sc"),
    "r":  ("157", "sc"),
    "o3": ("203", "sc"),
}

PLEVELS = [975, 850, 700, 500, 250, 100, 50, 10]


def rda_monthly_url(year: int, dom: str, code: str, var: str, suffix: str) -> str:
    base = "https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633001_nc/"
    fname = f"e5.moda.an.{dom}.128_{code}_{var}.ll025{suffix}.{year}010100_{year}120100.nc"
    return f"{base}e5.moda.an.{dom}/{year}/{fname}"


def find_var(ds: xr.Dataset, short: str) -> str:
    up = short.upper()
    for k in (up, f"VAR_{up}", up.replace("10", "10M")):
        if k in ds:
            return k
    raise KeyError(short)


def build_for(dom: str, var: str, code: str, suffix: str, lvl: int | None,
              years: Iterable[int], out_dir: Path, skip_existing: bool) -> None:
    """Compute mean+std climatology for one variable (+ optional level)."""
    fname = (f"sfc__{var}.nc" if dom == "sfc" else f"pl__{var}_{lvl}.nc")
    out = out_dir / fname
    if skip_existing and out.exists():
        print(f"  skip (exists): {out}")
        return

    print(f"  building {out} from years {min(years)}..{max(years)}…")
    arrs = []
    for y in years:
        url = rda_monthly_url(y, dom, code, var, suffix)
        ds = xr.open_dataset(url, decode_times=False)
        da = ds[find_var(ds, var)]
        if lvl is not None:
            da = da.sel(level=lvl)
        # Each yearly file has 12 time steps; build (year, month, lat, lon)
        da = da.rename({da.dims[0]: "month"})
        da = da.assign_coords(month=np.arange(1, 13))
        arrs.append(da.expand_dims(year=[y]))
    stack = xr.concat(arrs, dim="year")    # (year, month, lat, lon)

    mean = stack.mean("year", keep_attrs=True)
    std  = stack.std("year",  ddof=1, keep_attrs=True)  # sample std (n-1)

    vname_up = var.upper()
    out_ds = xr.Dataset({
        vname_up:               mean,
        f"{vname_up}_STD":      std,
    })
    out_ds.attrs["base_period"] = f"{min(years)}-{max(years)}"
    out_ds.attrs["history"] = "Built by tools/build_climatology.py (mean + year-to-year std)."

    out.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(out)
    print(f"    wrote {out}  ({out.stat().st_size/1024/1024:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="climatology", help="output dir")
    ap.add_argument("--years", default="1980-2010",
                    help="inclusive year range, e.g. 1980-2010")
    ap.add_argument("--skip-existing", action="store_true",
                    help="don't overwrite files already in --out")
    ap.add_argument("--only", default="", help="comma-separated list of "
                    "var keys to build (e.g. '2t,sstk,t_500'). Defaults to "
                    "ALL of SURFACE + PRESSURE × PLEVELS.")
    args = ap.parse_args()

    y0, y1 = (int(x) for x in args.years.split("-"))
    years = list(range(y0, y1 + 1))
    out_dir = Path(args.out)

    only = set(s.strip() for s in args.only.split(",")) if args.only else None

    print(f"Climatology base period: {y0}–{y1}")
    print(f"Output: {out_dir.resolve()}")

    print("\n== Surface ==")
    for var, (code, suffix) in SURFACE.items():
        if only and var not in only: continue
        try:
            build_for("sfc", var, code, suffix, None, years, out_dir,
                      args.skip_existing)
        except Exception as e:
            print(f"  ! failed for sfc/{var}: {e}")

    print("\n== Pressure levels ==")
    for var, (code, suffix) in PRESSURE.items():
        for lvl in PLEVELS:
            key = f"{var}_{lvl}"
            if only and key not in only: continue
            try:
                build_for("pl", var, code, suffix, lvl, years, out_dir,
                          args.skip_existing)
            except Exception as e:
                print(f"  ! failed for pl/{var}_{lvl}: {e}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
