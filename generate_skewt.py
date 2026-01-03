import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from meteodatalab import ogd_api
import xarray as xr
import numpy as np
import datetime

# --- Configuration ---
LAT_TARGET, LON_TARGET = 46.81, 6.94  # Payerne sounding station
CORE_VARS = ["T", "U", "V", "P"] 

def get_nearest_profile(ds, lat_target, lon_target):
    """Extracts a 1D vertical profile, ensuring extra dimensions are removed."""
    if ds is None: return None
    
    # Extract the actual data variable (e.g., 't' or 'p')
    data = ds if isinstance(ds, xr.DataArray) else None
    if data is None and hasattr(ds, 'data_vars') and len(ds.data_vars) > 0:
        data = ds[list(ds.data_vars)[0]]
    
    if data is None: return None

    # Handle native ICON grid coordinate names
    lat_dim = 'latitude' if 'latitude' in data.coords else 'lat'
    lon_dim = 'longitude' if 'longitude' in data.coords else 'lon'
    
    # Calculate distance to find nearest grid point
    dist = (data[lat_dim] - lat_target)**2 + (data[lon_dim] - lon_target)**2
    flat_idx = dist.argmin().values
    
    # Stack horizontal dims, select point, and SQUEEZE to force a 1D array
    # We use .flatten() to ensure MetPy gets a clean 1D vector
    profile = data.stack(gp=data.dims[-2:]).isel(gp=flat_idx).compute()
    return profile.squeeze()

def main():
    print("Connecting to MeteoSwiss for ICON-CH1 Skew-T...")
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # Try current and previous 3-hourly runs to find complete data
    times_to_try = [latest_run - datetime.timedelta(hours=i*3) for i in range(4)]
    
    success, profile_data, ref_time_final = False, {}, None
    for ref_time in times_to_try:
        print(f"--- Checking Model Run: {ref_time.strftime('%Y-%m-%d %H:%M')} UTC ---")
        try:
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                      reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                res = get_nearest_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET)
                if res is None or res.size < 5: raise ValueError(f"Variable {var} has insufficient levels.")
                profile_data[var] = res
            
            # Fetch Humidity with fallback
            for hum_var in ["RELHUM", "QV", "RH"]:
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum_var,
                                            reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                    res_h = get_nearest_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET)
                    if res_h is not None and res_h.size >= 5:
                        profile_data["HUM"], profile_data["HUM_TYPE"] = res_h, hum_var
                        break
                except: continue
            
            if "HUM" not in profile_data: raise ValueError("No vertical humidity profile found.")
            success, ref_time_final = True, ref_time
            break 
        except Exception as e: print(f"Run incomplete: {e}")

    if not success:
        print("Error: Could not find a run with full vertical profiles.")
        return

    # --- Unit Conversion and Data Validation ---
    p = profile_data["P"].values * units.Pa
    t = profile_data["T"].values * units.K
    u = profile_data["U"].values * units('m/s')
    v = profile_data["V"].values * units('m/s')
    
    # Print debug info to GitHub log
    print(f"Levels found: {len(p)} | Surface Pressure: {p[0].to(units.hPa):.1f}")
    print(f"Surface Temp: {t[0].to(units.degC):.1f} | Min Temp: {t.min().to(units.degC):.1f}")

    if profile_data["HUM_TYPE"] == "RELHUM":
        td = mpcalc.dewpoint_from_relative_humidity(t, profile_data["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, profile_data["HUM"].values * units('kg/kg'))

    # Sort data by pressure (Required for some MetPy calculations)
    inds = p.argsort()[::-1] # Ensure descending pressure (ground to sky)
    p, t, td, u, v = p[inds], t[inds], td[inds], u[inds], v[inds]

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 12))
    skew = SkewT(fig, rotation=45)
    
    # Plot data with standardized MetPy calls
    skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2.5, label='Temperature')
    skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2.5, label='Dewpoint')
    skew.plot_barbs(p.to(units.hPa)[::2], u[::2], v[::2]) # Decimate barbs for clarity
    
    skew.plot_dry_adiabats(alpha=0.1, color='red')
    skew.plot_moist_adiabats(alpha=0.1, color='blue')
    skew.plot_mixing_lines(alpha=0.1, color='green')
    
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 40)
    
    plt.title(f"ICON-CH1 Sounding | Run: {ref_time_final.strftime('%Y-%m-%d %H:%M')} UTC\nLat: {LAT_TARGET}, Lon: {LON_TARGET}", fontsize=14)
    plt.legend(loc='upper left')
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print("Success! Plot saved.")

if __name__ == "__main__":
    main()
