import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from meteodatalab import ogd_api
import xarray as xr
import numpy as np

# --- Configuration ---
LAT_TARGET, LON_TARGET = 46.81, 6.94  # Payerne, Switzerland
VARIABLES = ["T", "RELHUM", "U", "V", "P"] 

def main():
    print(f"Connecting to MeteoSwiss for ICON-CH1 at {LAT_TARGET}, {LON_TARGET}...")
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
            
            # get_from_ogd automatically loads horizontal coordinates
            ds = ogd_api.get_from_ogd(req)
            
            # Identify the data array and coordinate names
            data = ds if isinstance(ds, xr.DataArray) else ds[list(ds.data_vars)[0]]
            lat_dim = 'latitude' if 'latitude' in data.coords else 'lat'
            lon_dim = 'longitude' if 'longitude' in data.coords else 'lon'
            
            # --- NEAREST POINT LOGIC ---
            # Calculate distance squared to find the closest grid point
            dist = (data[lat_dim] - LAT_TARGET)**2 + (data[lon_dim] - LON_TARGET)**2
            
            # Flatten dimensions (x/y or cell) and pick the index with minimum distance
            # This handles both 2D regular grids and 1D native icosahedral grids
            flat_idx = dist.argmin().values
            profile = data.stack(gridpoint=data.dims[-2:]).isel(gridpoint=flat_idx).compute()
            profile_data[var] = profile
            
            if "forecast_reference_time" in profile.attrs:
                reference_time = profile.attrs["forecast_reference_time"]

        # --- Unit Conversion & Processing ---
        # Note: ICON pressure (P) is in Pa, Temp (T) in K
        p = profile_data["P"].values * units.Pa
        t = profile_data["T"].values * units.K
        rh = profile_data["RELHUM"].values / 100.0
        u = profile_data["U"].values * units('m/s')
        v = profile_data["V"].values * units('m/s')
        
        # Calculate Dewpoint from Temperature and Humidity
        td = mpcalc.dewpoint_from_relative_humidity(t, rh)

        # --- Create Skew-T ---
        fig = plt.figure(figsize=(10, 10))
        skew = SkewT(fig, rotation=45)
        
        # Plot Temp and Dewpoint (converting to hPa and Celsius)
        skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2, label='Temperature')
        skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2, label='Dewpoint')
        skew.plot_barbs(p.to(units.hPa), u, v)
        
        # Reference Lines
        skew.plot_dry_adiabats(alpha=0.2, color='orangered')
        skew.plot_moist_adiabats(alpha=0.2, color='blue')
        skew.plot_mixing_lines(alpha=0.2, color='green')
        
        skew.ax.set_ylim(1050, 100)
        skew.ax.set_xlim(-40, 40)
        
        plt.title(f"ICON-CH1 Skew-T | {reference_time} UTC\nLat: {LAT_TARGET}, Lon: {LON_TARGET}", fontsize=14)
        plt.legend(loc='upper left')
        
        plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
        print("Success! Skew-T plot generated.")

    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
