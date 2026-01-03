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
CORE_VARS = ["T", "U", "V", "P"] 

def get_nearest_profile(ds, lat_target, lon_target):
    """Robustly extracts a 1D vertical profile from the ICON grid."""
    data = ds if isinstance(ds, xr.DataArray) else ds[list(ds.data_vars)[0]]
    # Handle varying coordinate names
    lat_dim = 'latitude' if 'latitude' in data.coords else 'lat'
    lon_dim = 'longitude' if 'longitude' in data.coords else 'lon'
    
    # Distance calculation for the native icosahedral grid
    dist = (data[lat_dim] - lat_target)**2 + (data[lon_dim] - lon_target)**2
    flat_idx = dist.argmin().values
    
    # Select the point, compute it, and SQUEEZE to ensure 1D array for MetPy
    profile = data.stack(gp=data.dims[-2:]).isel(gp=flat_idx).compute()
    return profile.squeeze() # Removes singleton dims like (1, 1, 60) -> (60,)

def main():
    print("Connecting to MeteoSwiss for ICON-CH1...")
    
    # 1. Calculate standard 3-hourly run times
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_standard = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # Try the most recent runs (up to 9 hours back)
    times_to_try = [latest_standard - datetime.timedelta(hours=i*3) for i in range(4)]
    
    success = False
    for ref_time in times_to_try:
        print(f"--- Attempting Run: {ref_time.strftime('%Y-%m-%d %H:%M')} UTC ---")
        profile_data = {}
        try:
            # Fetch core vertical variables (T, U, V, Pressure)
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                      reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                profile_data[var] = get_nearest_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET)
            
            # Fetch Humidity (try multiple common variable names)
            for hum_var in ["RELHUM", "QV", "RH"]:
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum_var,
                                            reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                    profile_data["HUM"] = get_nearest_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET)
                    profile_data["HUM_TYPE"] = hum_var
                    break
                except: continue
            
            if "HUM" not in profile_data: raise ValueError("Missing Humidity data.")
            
            success = True
            break # Exit loop once a complete run is found
        except Exception as e:
            print(f"Run incomplete: {e}")

    if not success:
        print("Error: Could not find any complete model runs.")
        return

    # --- Unit Conversion ---
    # Convert ICON native units (Pa, K) to standard units
    p = profile_data["P"].values * units.Pa
    t = profile_data["T"].values * units.K
    u = profile_data["U"].values * units('m/s')
    v = profile_data["V"].values * units('m/s')
    
    if profile_data["HUM_TYPE"] == "RELHUM":
        td = mpcalc.dewpoint_from_relative_humidity(t, profile_data["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, profile_data["HUM"].values * units('kg/kg'))

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=45)
    
    # MetPy now receives clean 1D arrays
    skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2.5, label='Temperature')
    skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2.5, label='Dewpoint')
    skew.plot_barbs(p.to(units.hPa), u, v, x_loc=1.05)
    
    skew.plot_dry_adiabats(alpha=0.15, color='orangered', linestyle='--')
    skew.plot_moist_adiabats(alpha=0.15, color='blue', linestyle='--')
    skew.plot_mixing_lines(alpha=0.15, color='green', linestyle=':')
    
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-30, 30)
    
    plt.title(f"ICON-CH1 Skew-T | {ref_time.strftime('%Y-%m-%d %H:%M')} UTC\nLat: {LAT_TARGET}, Lon: {LON_TARGET}", fontsize=14)
    plt.legend(loc='upper left')
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print("Success! Skew-T saved as latest_skewt.png")

if __name__ == "__main__":
    main()
