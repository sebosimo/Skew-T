import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from meteodatalab import ogd_api
import xarray as xr
import numpy as np
import datetime

# --- Configuration ---
LAT_TARGET, LON_TARGET = 46.81, 6.94  # Payerne, Switzerland
CORE_VARS = ["T", "U", "V", "P"] # Essential variables

def get_nearest_profile(ds, lat_target, lon_target):
    """Finds the closest grid point in the native ICON icosahedral grid."""
    data = ds if isinstance(ds, xr.DataArray) else ds[list(ds.data_vars)[0]]
    lat_dim = 'latitude' if 'latitude' in data.coords else 'lat'
    lon_dim = 'longitude' if 'longitude' in data.coords else 'lon'
    
    # Square distance calculation to find nearest point
    dist = (data[lat_dim] - lat_target)**2 + (data[lon_dim] - lon_target)**2
    flat_idx = dist.argmin().values
    return data.stack(gp=data.dims[-2:]).isel(gp=flat_idx).compute()

def main():
    print(f"Connecting to MeteoSwiss for ICON-CH1...")
    
    # 1. Generate valid 3-hourly reference times
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_valid_run = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # We will try the last 3 standard model runs
    times_to_try = ["latest"] + [latest_valid_run - datetime.timedelta(hours=i*3) for i in range(3)]
    
    success = False
    for ref_time in times_to_try:
        print(f"--- Attempting Model Run: {ref_time} ---")
        profile_data = {}
        try:
            # Fetch core variables
            for var in CORE_VARS:
                print(f"Fetching {var}...")
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                      reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                profile_data[var] = get_nearest_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET)
            
            # Fetch Humidity (Try RELHUM first, then QV)
            hum_found = False
            for hum_var in ["RELHUM", "QV"]:
                print(f"Trying humidity variable: {hum_var}...")
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum_var,
                                            reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                    profile_data["HUM"] = get_nearest_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET)
                    profile_data["HUM_TYPE"] = hum_var
                    hum_found = True
                    break
                except: continue
            
            if not hum_found: raise ValueError("No humidity data found.")
            
            success = True
            break # Found a complete run!
        except Exception as e:
            print(f"Run {ref_time} failed: {e}")

    if not success:
        print("CRITICAL: No complete model runs available in the database.")
        return

    # --- Processing Data ---
    p = profile_data["P"].values * units.Pa
    t = profile_data["T"].values * units.K
    u = profile_data["U"].values * units('m/s')
    v = profile_data["V"].values * units('m/s')
    
    # Calculate Dewpoint based on which humidity variable was found
    if profile_data["HUM_TYPE"] == "RELHUM":
        rh = profile_data["HUM"].values / 100.0
        td = mpcalc.dewpoint_from_relative_humidity(t, rh)
    else:
        qv = profile_data["HUM"].values * units('kg/kg')
        td = mpcalc.dewpoint_from_specific_humidity(p, t, qv)

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=45)
    skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2, label='Temperature')
    skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2, label='Dewpoint')
    skew.plot_barbs(p.to(units.hPa), u, v)
    
    skew.plot_dry_adiabats(alpha=0.15, color='orangered')
    skew.plot_moist_adiabats(alpha=0.15, color='blue')
    skew.plot_mixing_lines(alpha=0.15, color='green')
    
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-35, 35)
    
    ts = profile_data["T"].attrs.get('forecast_reference_time', 'Latest')
    plt.title(f"ICON-CH1 Skew-T | Run: {ts} UTC\nLat: {LAT_TARGET}, Lon: {LON_TARGET}", fontsize=13)
    plt.legend(loc='upper left')
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print("Success! Skew-T saved.")

if __name__ == "__main__":
    main()
