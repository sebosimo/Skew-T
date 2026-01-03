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
CORE_VARS = ["T", "U", "V", "P", "FI"] # FI is Geopotential for altitude calculation
STEP = 0 # Generate single plot for the analysis state (+0h)

def get_nearest_profile(ds, lat_target, lon_target):
    """Robustly extracts a vertical profile from regular or native ICON grids."""
    if ds is None: return None
    data = ds if isinstance(ds, xr.DataArray) else ds[list(ds.data_vars)[0]]
    lat_coord = 'latitude' if 'latitude' in data.coords else 'lat'
    lon_coord = 'longitude' if 'longitude' in data.coords else 'lon'
    horiz_dims = data.coords[lat_coord].dims
    dist = (data[lat_coord] - lat_target)**2 + (data[lon_coord] - lon_target)**2
    flat_idx = dist.argmin().values
    
    if len(horiz_dims) == 1:
        profile = data.isel({horiz_dims[0]: flat_idx})
    else:
        profile = data.stack(gp=horiz_dims).isel(gp=flat_idx)
    return profile.squeeze().compute()

def main():
    print("Connecting to MeteoSwiss for Single ICON-CH1 Plot...")
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run_start = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # Try the last few 3-hourly runs to find a completed analysis
    times_to_try = [latest_run_start - datetime.timedelta(hours=i*3) for i in range(4)]
    
    selected_run, profile_data = None, {}
    horizon_str = f"P0DT{STEP}H"

    for ref_time in times_to_try:
        print(f"--- Checking Model Run: {ref_time.strftime('%H:%M')} UTC ---")
        try:
            # Attempt to fetch all core variables for this specific time
            current_batch = {}
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                      reference_datetime=ref_time, horizon=horizon_str, perturbed=False)
                res = get_nearest_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET)
                if res is None or res.size < 5: raise ValueError(f"Missing {var}")
                current_batch[var] = res
            
            # Humidity Fallback logic
            for hum_var in ["RELHUM", "QV"]:
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum_var,
                                            reference_datetime=ref_time, horizon=horizon_str, perturbed=False)
                    res_h = get_nearest_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET)
                    if res_h is not None and res_h.size >= 5:
                        current_batch["HUM"], current_batch["HUM_TYPE"] = res_h, hum_var
                        break
                except: continue
            
            if "HUM" in current_batch:
                profile_data, selected_run = current_batch, ref_time
                break
        except Exception as e: print(f"Run incomplete: {e}")

    if not selected_run:
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

    inds = p.argsort()[::-1] # Sort ground-to-sky
    p, t, td, u, v, h = p[inds], t[inds], td[inds], u[inds], v[inds], h[inds]
    wind_speed = mpcalc.wind_speed(u, v).to(units.knot)

    # --- Plotting ---
    fig = plt.figure(figsize=(12, 10))
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.60, 0.85))
    skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2.5, label='Temp')
    skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2.5, label='Dewpoint')
    skew.plot_barbs(p.to(units.hPa)[::4], u[::4], v[::4]) # Decimate for clarity
    
    # Altitude labels in meters
    for height_m in [1000, 2000, 3000, 5000, 7000, 9000]:
        idx = (np.abs(h.m - height_m)).argmin()
        if p[idx].to(units.hPa).m > 100:
            skew.ax.text(-38, p[idx].to(units.hPa).m, f"{height_m}m", color='gray', fontsize=9, ha='right', va='center')

    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 40)
    skew.ax.set_ylabel("Pressure (hPa) / Altitude (m)")

    # Wind Speed Panel
    ax_wind = fig.add_axes([0.72, 0.1, 0.22, 0.85], sharey=skew.ax)
    ax_wind.plot(wind_speed, p.to(units.hPa), 'k-', linewidth=1.5)
    ax_wind.set_xlabel("Windspeed (kt)")
    ax_wind.grid(True, alpha=0.3)
    ax_wind.set_xlim(0, max(60, wind_speed.m.max() + 5))
    plt.setp(ax_wind.get_yticklabels(), visible=False)
    
    plt.suptitle(f"ICON-CH1 | Valid: {selected_run.strftime('%Y-%m-%d %H:%M')} UTC", fontsize=14)
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=120)
    print(f"Success! Saved single plot: latest_skewt.png")

if __name__ == "__main__":
    main()
