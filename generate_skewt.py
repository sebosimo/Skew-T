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
CORE_VARS = ["T", "U", "V", "P", "FI"] # FI is geopotential for height

def get_nearest_profile(ds, lat_target, lon_target):
    """Robustly extracts a vertical profile from the native ICON grid."""
    if ds is None: return None
    
    # Safely find the main data variable (often lowercase 't', 'u', etc.)
    var_names = list(ds.data_vars)
    if not var_names:
        print("Warning: Dataset contains no data variables.")
        return None
    data = ds[var_names[0]]

    # Determine horizontal coordinate names
    lat_coord = 'latitude' if 'latitude' in data.coords else 'lat'
    lon_coord = 'longitude' if 'longitude' in data.coords else 'lon'
    
    # Calculate distance to find closest grid cell
    dist = (data[lat_coord] - lat_target)**2 + (data[lon_coord] - lon_target)**2
    
    # Find the dimension that represents the grid (usually 'ncells')
    # If the grid is already 1D, argmin works directly.
    # If not, we stack horizontal dimensions.
    horiz_dims = data.coords[lat_coord].dims
    if len(horiz_dims) == 1:
        idx = dist.argmin().values
        profile = data.isel({horiz_dims[0]: idx})
    else:
        profile = data.stack(gp=horiz_dims).isel(gp=dist.argmin().values)
        
    return profile.squeeze().compute()

def main():
    print("Connecting to MeteoSwiss for Advanced Sounding...")
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_valid = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # Loop back through runs until we find a complete one
    times_to_try = [latest_valid - datetime.timedelta(hours=i*3) for i in range(5)]
    
    success, profile_data, final_time = False, {}, None
    for ref_time in times_to_try:
        print(f"--- Checking Model Run: {ref_time.strftime('%H:%M')} UTC ---")
        try:
            batch = {}
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                      reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                res = get_nearest_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET)
                if res is None: raise ValueError(f"Empty {var}")
                batch[var] = res
            
            # Humidity Fallback
            for h_var in ["RELHUM", "QV", "RH", "q"]:
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=h_var,
                                            reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                    res_h = get_nearest_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET)
                    if res_h is not None:
                        batch["HUM"], batch["HUM_TYPE"] = res_h, h_var
                        break
                except: continue
            
            if "HUM" in batch:
                profile_data, success, final_time = batch, True, ref_time
                break
        except Exception as e: print(f"Run incomplete: {e}")

    if not success: return

    # --- Data Processing ---
    p = profile_data["P"].values * units.Pa
    t = profile_data["T"].values * units.K
    u = profile_data["U"].values * units('m/s')
    v = profile_data["V"].values * units('m/s')
    # Convert Geopotential to Altitude (m)
    h = (profile_data["FI"].values * units('m^2/s^2') / (9.80665 * units('m/s^2'))).to(units.m)
    
    if "RELHUM" in profile_data["HUM_TYPE"].upper():
        td = mpcalc.dewpoint_from_relative_humidity(t, profile_data["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, profile_data["HUM"].values * units('kg/kg'))

    # Sort descending by pressure
    idx = p.argsort()[::-1]
    p, t, td, u, v, h = p[idx], t[idx], td[idx], u[idx], v[idx], h[idx]
    ws = mpcalc.wind_speed(u, v).to(units.knot)

    # --- Plotting ---
    fig = plt.figure(figsize=(12, 10))
    # Skew-T Panel
    skew = SkewT(fig, rotation=45, rect=(0.12, 0.1, 0.60, 0.85))
    skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2.5, label='Temp')
    skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2.5, label='Dewpoint')
    skew.plot_barbs(p.to(units.hPa)[::5], u[::5], v[::5]) 
    
    # Altitude markings in meters
    for m in [1000, 2000, 3000, 5000, 7000, 9000]:
        h_idx = (np.abs(h.m - m)).argmin()
        if p[h_idx].to(units.hPa).m > 100:
            skew.ax.text(-38, p[h_idx].to(units.hPa).m, f"{m}m", color='gray', fontsize=9, ha='right', weight='bold')

    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 40)
    skew.ax.set_ylabel("Pressure (hPa) / Altitude (m)")

    # Wind Panel on right
    ax_wind = fig.add_axes([0.75, 0.1, 0.2, 0.85], sharey=skew.ax)
    ax_wind.plot(ws, p.to(units.hPa), 'k-', linewidth=1.8)
    ax_wind.set_xlabel("Windspeed (kt)")
    ax_wind.grid(True, alpha=0.3)
    ax_wind.set_xlim(0, max(50, ws.m.max() + 5))
    plt.setp(ax_wind.get_yticklabels(), visible=False)
    
    plt.suptitle(f"ICON-CH1 Sounding | Valid: {final_time.strftime('%Y-%m-%d %H:%M')} UTC", fontsize=15, weight='bold')
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print(f"Success! Saved run from {final_time.hour}h UTC.")

if __name__ == "__main__":
    main()
