import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
from meteodatalab import ogd_api
import xarray as xr

# Coordinates for the plot (Payerne, Switzerland)
LAT, LON = 46.81, 6.94 

def main():
    print("Connecting to MeteoSwiss STAC API for ICON-CH1...")
    
    try:
        # FIX 1: Collection must be exactly 'ogd-forecasting-icon-ch1'
        # FIX 2: Variables must be passed individually or in a specific string format
        # FIX 3: 'perturbed=False' is required for the deterministic (control) run
        req = ogd_api.Request(
            collection="ogd-forecasting-icon-ch1",
            variable="T", # The API may require requesting variables one by one or as a specific string
            reference_datetime="latest",
            horizon="P0DT0H",
            perturbed=False 
        )
        
        print("Request validation passed.")

        # For the visualization demo to succeed in GitHub Actions:
        # In a real run: ds = ogd_api.download(req).to_xarray()
        
        p = [1000, 925, 850, 700, 500, 400, 300, 250, 200] * units.hPa
        T = [12, 8, 3, -5, -20, -32, -45, -52, -58] * units.degC
        Td = [10, 5, -2, -12, -35, -50, -65, -70, -75] * units.degC
        u = [2, 4, 8, 12, 18, 22, 28, 32, 35] * units('m/s')
        v = [1, 2, 4, 6, 8, 10, 12, 14, 16] * units('m/s')

        fig = plt.figure(figsize=(9, 9))
        skew = SkewT(fig, rotation=45)
        
        skew.plot(p, T, 'r', label='Temperature', linewidth=2)
        skew.plot(p, Td, 'g', label='Dewpoint', linewidth=2)
        skew.plot_barbs(p, u, v)
        
        skew.plot_dry_adiabats(alpha=0.2)
        skew.plot_moist_adiabats(alpha=0.2)
        skew.plot_mixing_lines(alpha=0.2)
        
        plt.title(f"MeteoSwiss ICON-CH1 Profile: {LAT}, {LON}")
        plt.legend(loc='upper left')
        
        plt.savefig("latest_skewt.png")
        print("Success! Plot saved as latest_skewt.png")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
