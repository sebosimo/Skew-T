import xarray as xr
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import datetime

# 1. Define Coordinates (e.g., Bern)
LAT, LON = 46.948, 7.447 

def fetch_and_plot():
    # In a real script, you'd use the STAC API to find the latest 'icon-ch1' collection
    # For now, let's assume we load the latest available run via xarray
    # Example: ds = xr.open_dataset("https://stac.meteoswiss.ch/...") 
    
    # Extract vertical profile for your location
    # profile = ds.sel(lat=LAT, lon=LON, method='nearest').isel(time=0)
    
    # Plotting code (similar to the Skew-T example provided previously)
    # ... skew.plot(...)
    
    # Save the plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
    plt.savefig(f"skewt_icon_ch1_{timestamp}.png")

if __name__ == "__main__":
    fetch_and_plot()
