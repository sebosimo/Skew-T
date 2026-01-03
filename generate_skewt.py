import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from meteodatalab import ogd_api
import xarray as xr
import pandas as pd

# --- Configuration ---
# Coordinates for Payerne, Switzerland
LAT, LON = 46.81, 6.94 
# Variables needed for a full Skew-T profile
VARIABLES = ["T", "RELHUM", "U", "V", "P"] 

def main():
    print("Connecting to MeteoSwiss STAC API for ICON-CH1...")
    profile_data = {}
    reference_time = "Unknown"

    try:
        for var in VARIABLES:
            print(f"Fetching {var}...")
            req = ogd_api.Request(
                collection="ogd-forecasting-icon-ch1",
                variable=var,
                reference_datetime="latest",
                horizon="P0DT0H",
                perturbed=False 
            )
            
            # Retrieve data from OGD portal
            ds = ogd_api.get_from_ogd(req)
            
            # FIX: Handle cases where the API returns a DataArray instead of a Dataset
            if isinstance(ds, xr.Dataset):
                # If Dataset, pick the first data variable (usually the one requested)
                var_key = list(ds.data_vars)[0]
                data = ds[var_key]
            else:
                # If already a DataArray, use it directly
                data = ds

            # Extract the vertical profile for the chosen coordinates
            profile = data.sel(lat=LAT, lon=LON, method="nearest").compute()
            profile_data[var] = profile
            
            # Capture the model run time from metadata
            if "forecast_reference_time" in profile.attrs:
                reference_time = profile.attrs["forecast_reference_time"]

        # --- Prepare Data for MetPy ---
        # Note: We attach units to the raw values for calculation
        p = profile_data["P"].values * units.Pa
        t = profile_data["T"].values * units.K
        rh = profile_data["RELHUM"].values / 100.0  # Convert 0-100% to 0-1
        u = profile_data["U"].values * units('m/s')
        v = profile_data["V"].values * units('m/s')
        
        # Calculate Dewpoint from Temperature and Relative Humidity
        td = mpcalc.dewpoint_from_relative_humidity(t, rh)

        # --- Generate the Skew-T Plot ---
        fig = plt.figure(figsize=(10, 10))
        skew = SkewT(fig, rotation=45)
        
        # Plot Temperature (Red) and Dewpoint (Green)
        skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2, label='Temperature')
        skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2, label='Dewpoint')
        
        # Plot Wind Barbs on the right side
        skew.plot_barbs(p.to(units.hPa), u, v)
        
        # Add background reference lines (Adiabats and Mixing Lines)
        skew.plot_dry_adiabats(alpha=0.2, color='orangered', linestyle='--')
        skew.plot_moist_adiabats(alpha=0.2, color='blue', linestyle='--')
        skew.plot_mixing_lines(alpha=0.2, color='green', linestyle='--')
        
        # Set Plot Limits
        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-40, 40)
        
        # Formatting and Title
        plt.title(f"ICON-CH1 Skew-T | Run: {reference_time}\nLat: {LAT}, Lon: {LON}", fontsize=14)
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Pressure (hPa)")
        plt.legend(loc='upper left')
        
        # Save the output file
        plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
        print("Success! Skew-T saved as latest_skewt.png")

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
