import xarray as xr
import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide")

YEARS  = list(range(1980, 2023))
MONTHS = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
          7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

yr  = st.sidebar.selectbox("Year", YEARS)
mon = st.sidebar.selectbox("Month", list(MONTHS), format_func=lambda m: MONTHS[m])

code, var = "034", "sstk"
url = (f"https://thredds.rda.ucar.edu/thredds/dodsC/files/"
       f"g/d633001_nc/e5.moda.an.sfc/{yr}/"
       f"e5.moda.an.sfc.128_{code}_{var}.ll025sc.{yr}010100_{yr}120100.nc")
st.caption(f"Fetching ERA5 SST for {MONTHS[mon]} {yr}")
with st.spinner("Opening NetCDF…"):
    da = xr.open_dataset(url, decode_times=False)["SSTK"][0] - 273.15

fig = px.imshow(
    da,
    origin="lower",
    aspect="auto",
    color_continuous_scale="thermal",
    labels=dict(color="°C"),
    title=f"ERA5 Sea-Surface Temperature • {MONTHS[mon]} {yr}"
)
st.plotly_chart(fig, use_container_width=True)
