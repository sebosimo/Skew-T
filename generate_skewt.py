import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from meteodatalab import ogd_api
import xarray as xr
import numpy as np
import datetime

# --- Configuration ---
LAT_TARGET, LON_TARGET = 46.81, 6.94  # Payerne
# We try both possible names for Humidity to be safe
VARIABLES = ["T", "U", "V", "P", "RELHUM"] 

def fetch_with_fallback(var, ref_time):
    """Tries to fetch a variable for a specific time, returns None if missing."""
    try:
        req = ogd_api.Request(
            collection="ogd-forecasting-icon-ch1",
            variable=var,
            reference_datetime=ref_time,
            horizon="P0DT0H",
            perturbed=False
        )
        return ogd_api.get_from_ogd(req)
    except Exception:
        return None

def main():
    print(f"Connecting to MeteoSwiss for ICON-CH1...")
    profile_data = {}
    
    # Strategy: Try 'latest', if it fails (partial upload), try 3 hours ago
    times_to_try = ["latest", datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=3)]
    
    success = False
    for ref_time in times_to_try:
        print(f"Attempting to fetch data for run: {ref_time}")
        profile_data = {}
        try:
            for var in VARIABLES:
                ds = fetch_with_fallback(var, ref_time)
                if ds is None:
                    raise ValueError(f"Variable {var} missing for this run.")
                
                # Identify data array
                data = ds if isinstance(ds, xr.DataArray) else ds[list(ds.data_vars)[0]]
                lat_dim = 'latitude' if 'latitude' in data.coords else 'lat'
                lon_dim = 'longitude' if 'longitude' in data.coords else 'lon'
                
                # NEAREST POINT LOGIC
                dist = (data[lat_dim] - LAT_TARGET)**2 + (data[lon_dim] - LON_TARGET)**2
                flat_idx = dist.argmin().values
                profile_data[var] = data.stack(gp=data.dims[-2:]).isel(gp=flat_idx).compute()
            
            success = True
            break # We found a complete run!
        except Exception as e:
            print(f"Run {ref_time} incomplete or unavailable: {e}")
            continue

    if not success:
        print("Could not find a complete model run in the last 6 hours.")
        return

    # --- Processing & Units ---
    p = profile_data["P"].values * units.Pa
    t = profile_data["T"].values * units.K
    u = profile_data["U"].values * units('m/s')
    v = profile_data["V"].values * units('m/s')
    rh = profile_data["RELHUM"].values / 100.0
    
    td = mpcalc.dewpoint_from_relative_humidity(t, rh)

    # --- Create Skew-T ---
    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=45)
    
    skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2, label='Temperature')
    skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2, label='Dewpoint')
    skew.plot_barbs(p.to(units.hPa), u, v)
    
    skew.plot_dry_adiabats(alpha=0.2, color='orangered')
    skew.plot_moist_adiabats(alpha=0.2, color='blue')
    skew.plot_mixing_lines(alpha=0.2, color='green')
    
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-30, 30)
    
    ref_ts = profile_data["T"].attrs.get('forecast_reference_time', 'Model Run')
    plt.title(f"ICON-CH1 Skew-T | Run: {ref_ts} UTC\nLat: {LAT_TARGET}, Lon: {LON_TARGET}", fontsize=13)
    plt.legend(loc='upper left')
    
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print("Success! Skew-T saved as latest_skewt.png")

if __name__ == "__main__":
    main()
