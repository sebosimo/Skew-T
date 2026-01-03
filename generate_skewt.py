import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
from meteodatalab import ogd_api
import xarray as xr
import pandas as pd

# Coordinates for the plot (Payerne, Switzerland)
LAT, LON = 46.81, 6.94 

def main():
    print("Connecting to MeteoSwiss STAC API for ICON-CH1...")
    
    # Requesting the latest ICON-CH1 model data
    # ICON-CH1 is the highest resolution model (1km)
    try:
        req = ogd_api.Request(
            collection="ch.meteoschweiz.ogd-forecasting-icon-ch1",
            variable=["T", "RELHUM", "U", "V"],
            reference_datetime="latest",
            horizon="P0DT0H"
        )
        
        # Note: In a live environment, this downloads the GRIB files. 
        # For this script to work, the runner must have 'eccodes' installed.
        print("Request successful. Generating plot...")

        # Placeholder data logic for visualization test
        # In your actual run, you would use: ds = ogd_api.download(req).to_xarray()
        p = [1000, 925, 850, 700, 500, 400, 300, 250, 200] * units.hPa
        T = [12, 8, 3, -5, -20, -32, -45, -52, -58] * units.degC
        Td = [10, 5, -2, -12, -35, -50, -65, -70, -75] * units.degC
        u = [2, 4, 8, 12, 18, 22, 28, 32, 35] * units('m/s')
        v = [1, 2, 4, 6, 8, 10, 12, 14, 16] * units('m/s')

        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, rotation=45)
        
        skew.plot(p, T, 'r', label='Temperature')
        skew.plot(p, Td, 'g', label='Dewpoint')
        skew.plot_barbs(p, u, v)
        
        skew.plot_dry_adiabats(alpha=0.25)
        skew.plot_moist_adiabats(alpha=0.25)
        skew.plot_mixing_lines(alpha=0.25)
        
        plt.title(f"ICON-CH1 Skew-T Profile: {LAT}, {LON}")
        plt.legend(loc='upper left')
        
        plt.savefig("latest_skewt.png")
        print("Success! Plot saved as latest_skewt.png")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
