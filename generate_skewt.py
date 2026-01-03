import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from meteodatalab import ogd_api
import xarray as xr
import os

# --- Configuration ---
LAT, LON = 46.81, 6.94  # Payerne, Switzerland
VARIABLES = ["T", "RELHUM", "U", "V", "P"] # Temperature, Rel Humidity, Wind, Pressure

def main():
    print("Connecting to MeteoSwiss STAC API for ICON-CH1...")
    profile_data = {}

    try:
        # Loop through variables because the API expects strings, not lists
        for var in VARIABLES:
            print(f"Fetching {var}...")
            req = ogd_api.Request(
                collection="ogd-forecasting-icon-ch1",
                variable=var,
                reference_datetime="latest",
                horizon="P0DT0H",
                perturbed=False # Required for the deterministic run
            )
            
            # Download and convert to Xarray
            ds = ogd_api.download(req).to_xarray()
            
            # Select the profile for your coordinates
            # Selecting nearest point in the 1km ICON-CH1 grid
            profile = ds.sel(lat=LAT, lon=LON, method="nearest").compute()
            profile_data[var] = profile

        # --- Extract Data for Plotting ---
        # Note: Variable names in the Xarray may differ slightly from STAC names (e.g., 't', 'p')
        # We extract values and attach MetPy units
        p = profile_data["P"].p.values * units.Pa
        t = profile_data["T"].t.values * units.K
        rh = profile_data["RELHUM"].relhum.values / 100.0 # Convert % to fraction
        u = profile_data["U"].u.values * units('m/s')
        v = profile_data["V"].v.values * units('m/s')
        
        # Calculate Dewpoint from Temperature and Relative Humidity
        td = mpcalc.dewpoint_from_relative_humidity(t, rh)

        # --- Generate Skew-T ---
        fig = plt.figure(figsize=(10, 10))
        skew = SkewT(fig, rotation=45)
        
        # Plot profiles (Convert Pressure to hPa and Temp to Celsius for the plot)
        skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2, label='Temperature')
        skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2, label='Dewpoint')
        skew.plot_barbs(p.to(units.hPa), u, v)
        
        # Add background reference lines
        skew.plot_dry_adiabats(alpha=0.2)
        skew.plot_moist_adiabats(alpha=0.2)
        skew.plot_mixing_lines(alpha=0.2)
        
        # Formatting
        ref_time = profile_data["T"].attrs.get('forecast_reference_time', 'Latest')
        plt.title(f"ICON-CH1 Skew-T | {ref_time} UTC\nLat: {LAT}, Lon: {LON}", fontsize=14)
        plt.legend(loc='upper left')
        
        plt.savefig("latest_skewt.png", bbox_inches='tight')
        print("Successfully saved real-time plot: latest_skewt.png")

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
