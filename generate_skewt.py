import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from meteodatalab import ogd_api
import xarray as xr
import os

# --- Configuration ---
LAT, LON = 46.81, 6.94  # Payerne, Switzerland
# Variables must be requested individually to match API requirements
VARIABLES = ["T", "RELHUM", "U", "V", "P"] 

def main():
    print("Connecting to MeteoSwiss STAC API for ICON-CH1...")
    profile_data = {}

    try:
        for var in VARIABLES:
            print(f"Fetching {var}...")
            # Create the request with required 'perturbed' field
            req = ogd_api.Request(
                collection="ogd-forecasting-icon-ch1",
                variable=var,
                reference_datetime="latest", # Automatically selects newest run
                horizon="P0DT0H",
                perturbed=False 
            )
            
            # FIX: Use 'get_from_ogd' instead of 'download'
            # This returns an xarray.Dataset
            ds = ogd_api.get_from_ogd(req)
            
            # Select the nearest grid point for the high-res 1km ICON-CH1 grid
            # Note: GRIB keys are often lowercase in the resulting xarray
            var_key = var.lower() if var.lower() in ds.data_vars else list(ds.data_vars)[0]
            profile = ds[var_key].sel(lat=LAT, lon=LON, method="nearest").compute()
            profile_data[var] = profile

        # --- Data Extraction & Units ---
        p = profile_data["P"].values * units.Pa
        t = profile_data["T"].values * units.K
        rh = profile_data["RELHUM"].values / 100.0
        u = profile_data["U"].values * units('m/s')
        v = profile_data["V"].values * units('m/s')
        
        # Calculate Dewpoint for the Skew-T profile
        td = mpcalc.dewpoint_from_relative_humidity(t, rh)

        # --- Generate Skew-T Plot ---
        fig = plt.figure(figsize=(10, 10))
        skew = SkewT(fig, rotation=45)
        
        skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2, label='Temperature')
        skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2, label='Dewpoint')
        skew.plot_barbs(p.to(units.hPa), u, v)
        
        skew.plot_dry_adiabats(alpha=0.2)
        skew.plot_moist_adiabats(alpha=0.2)
        skew.plot_mixing_lines(alpha=0.2)
        
        # Add timestamp from metadata
        ref_time = profile_data["T"].attrs.get('forecast_reference_time', 'Latest')
        plt.title(f"ICON-CH1 Skew-T | {ref_time} UTC\nLat: {LAT}, Lon: {LON}", fontsize=14)
        plt.legend(loc='upper left')
        
        plt.savefig("latest_skewt.png", bbox_inches='tight')
        print("Success: Skew-T saved as latest_skewt.png")

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
