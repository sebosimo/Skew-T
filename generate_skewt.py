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
CORE_VARS = ["T", "U", "V", "P", "FI"] 

def get_nearest_profile(ds, lat_target, lon_target, var_hint=""):
    """Robustly extracts a vertical profile, handling empty or complex datasets."""
    if ds is None: return None
    
    # Safely extract the DataArray
    if isinstance(ds, xr.DataArray):
        data = ds
    elif hasattr(ds, 'data_vars') and len(ds.data_vars) > 0:
        # Pick the variable that matches our request (case-insensitive)
        var_name = next((v for v in ds.data_vars if v.upper() == var_hint.upper()), list(ds.data_vars)[0])
        data = ds[var_name]
    else:
        return None

    # Identify horizontal dimensions (native ICON grid uses 1D 'ncells')
    lat_coord = 'latitude' if 'latitude' in data.coords else 'lat'
    lon_coord = 'longitude' if 'longitude' in data.coords else 'lon'
    horiz_dims = data.coords[lat_coord].dims
    
    # Calculate nearest point index
    dist = (data[lat_coord] - lat_target)**2 + (data[lon_coord] - lon_target)**2
    flat_idx = dist.argmin().values
    
    # Slice only the horizontal dimension to keep all vertical levels
    profile = data.isel({horiz_dims[0]: flat_idx}) if len(horiz_dims) == 1 else data.stack(gp=horiz_dims).isel(gp=flat_idx)
    
    # SQUEEZE and COMPUTATION: Ensure we have a clean 1D array of values
    return profile.squeeze().compute()

def main():
    print("Connecting to MeteoSwiss for Advanced ICON-CH1 Skew-T...")
    now = datetime.datetime.now(datetime.timezone.utc)
    # Round down to the last valid 3-hourly run (00, 03, 06, 09, 12, 15, 18, 21)
    base_hour = (now.hour // 3) * 3
    latest_valid_start = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # Search backwards through the last 5 runs (15 hours)
    times_to_try = [latest_valid_start - datetime.timedelta(hours=i*3) for i in range(5)]
    
    success, profile_data, final_run_time = False, {}, None
    for ref_time in times_to_try:
        print(f"--- Attempting Model Run: {ref_time.strftime('%H:%M')} UTC ---")
        try:
            batch = {}
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                      reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                res = get_nearest_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET, var)
                if res is None or res.size < 10: raise ValueError(f"Empty or surface-only data for {var}")
                batch[var] = res
            
            # Humidity Fallback logic
            for h_var in ["RELHUM", "QV", "Q"]:
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=h_var,
                                            reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                    res_h = get_nearest_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET, h_var)
                    if res_h is not None and res_h.size >= 10:
                        batch["HUM"], batch["HUM_TYPE"] = res_h, h_var
                        break
                except: continue
            
            if "HUM" in batch:
                profile_data, success, final_run_time = batch, True, ref_time
                break
        except Exception as e:
            print(f"Run {ref_time.hour}h incomplete: {e}")

    if not success:
        print("Error: Could not find a complete model run.")
        return

    # --- Data Processing ---
    p = profile_data["P"].values * units.Pa
    t = profile_data["T"].values * units.K
    u = profile_data["U"].values * units('m/s')
    v = profile_data["V"].values * units('m/s')
    h = (profile_data["FI"].values * units('m^2/s^2') / (9.80665 * units('m/s^2'))).to(units.m) #
    
    if profile_data["HUM_TYPE"] == "RELHUM":
        td = mpcalc.dewpoint_from_relative_humidity(t, profile_data["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, profile_data["HUM"].values * units('kg/kg'))

    # Sort descending by pressure (ground to sky)
    inds = p.argsort()[::-1]
    p, t, td, u, v, h = p[inds], t[inds], td[inds], u[inds], v[inds], h[inds]
    wind_speed = mpcalc.wind_speed(u, v).to(units.knot)

    # --- Multi-Panel Plotting ---
    fig = plt.figure(figsize=(12, 10))
    # Skew-T Panel
    skew = SkewT(fig, rotation=45, rect=(0.12, 0.1, 0.60, 0.85))
    skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2.5, label='Temp')
    skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2.5, label='Dewpoint')
    skew.plot_barbs(p.to(units.hPa)[::5], u[::5], v[::5]) # Decimate for clarity
    
    # Professional reference lines
    skew.plot_dry_adiabats(alpha=0.1, color='red', ls='--')
    skew.plot_moist_adiabats(alpha=0.1, color='blue', ls='--')
    
    # Altitude labels in meters
    for height_m in [1000, 2000, 3000, 5000, 7000, 9000]:
        idx = (np.abs(h.m - height_m)).argmin()
        if p[idx].to(units.hPa).m > 110:
            skew.ax.text(-38, p[idx].to(units.hPa).m, f"{height_m}m", color='gray', fontsize=9, ha='right', weight='bold')

    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 40)
    skew.ax.set_ylabel("Pressure (hPa) / Altitude (m)")

    # Wind Speed Graph (Right panel)
    ax_wind = fig.add_axes([0.75, 0.1, 0.2, 0.85], sharey=skew.ax)
    ax_wind.plot(wind_speed, p.to(units.hPa), 'k-', linewidth=1.8)
    ax_wind.set_xlabel("Windspeed (kt)")
    ax_wind.grid(True, alpha=0.3)
    ax_wind.set_xlim(0, max(50, wind_speed.m.max() + 10))
    plt.setp(ax_wind.get_yticklabels(), visible=False)
    
    plt.suptitle(f"ICON-CH1 Sounding | Valid: {final_run_time.strftime('%Y-%m-%d %H:%M')} UTC", fontsize=15, weight='bold')
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print(f"Success! Plot saved for {final_run_time.hour} UTC run.")

if __name__ == "__main__":
    main()
