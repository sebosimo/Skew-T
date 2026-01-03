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
# FI = Geopotential, P = Pressure, T = Temp, U/V = Wind components
CORE_VARS = ["T", "U", "V", "P", "FI"] 

def get_nearest_profile(ds, lat_target, lon_target, var_name_hint=""):
    """Extracts a clean 1D vertical profile from ICON datasets."""
    if ds is None: return None
    
    # Identify the data variable safely to avoid IndexError
    if isinstance(ds, xr.DataArray):
        data = ds
    elif hasattr(ds, 'data_vars') and len(ds.data_vars) > 0:
        data = ds[list(ds.data_vars)[0]]
    else:
        print(f"Warning: Dataset for {var_name_hint} contains no data variables.")
        return None

    # Handle icosahedral grid coordinates
    lat_coord = 'latitude' if 'latitude' in data.coords else 'lat'
    lon_coord = 'longitude' if 'longitude' in data.coords else 'lon'
    horiz_dims = data.coords[lat_coord].dims
    
    # Nearest neighbor calculation
    dist = (data[lat_coord] - lat_target)**2 + (data[lon_coord] - lon_target)**2
    flat_idx = dist.argmin().values
    
    # Extract column and squeeze out extra dimensions
    profile = data.isel({horiz_dims[0]: flat_idx}) if len(horiz_dims) == 1 else data.stack(gp=horiz_dims).isel(gp=flat_idx)
    return profile.squeeze().compute()

def main():
    print("Connecting to MeteoSwiss for Advanced ICON-CH1 Plot...")
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run_start = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # Search backwards for a completed model run
    times_to_try = [latest_run_start - datetime.timedelta(hours=i*3) for i in range(5)]
    
    selected_run, profile_data = None, {}
    for ref_time in times_to_try:
        print(f"--- Checking Model Run: {ref_time.strftime('%Y-%m-%d %H:%M')} UTC ---")
        try:
            current_batch = {}
            # Fetch all core variables
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                      reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                res = get_nearest_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET, var)
                if res is None or res.size < 10: raise ValueError(f"Incomplete profile for {var}")
                current_batch[var] = res
            
            # Fetch Humidity with extended fallback names
            for hum_var in ["RELHUM", "QV", "Q"]:
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum_var,
                                            reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                    res_h = get_nearest_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET, hum_var)
                    if res_h is not None and res_h.size >= 10:
                        current_batch["HUM"], current_batch["HUM_TYPE"] = res_h, hum_var
                        break
                except: continue
            
            if "HUM" in current_batch:
                profile_data, selected_run = current_batch, ref_time
                break
        except Exception as e: print(f"Run {ref_time.hour}h incomplete: {e}")

    if not selected_run:
        print("Error: Could not find any completed model runs in the last 12 hours.")
        return

    # --- Processing ---
    p = profile_data["P"].values * units.Pa
    t = profile_data["T"].values * units.K
    u = profile_data["U"].values * units('m/s')
    v = profile_data["V"].values * units('m/s')
    # Height Z = Geopotential / Gravity
    h = (profile_data["FI"].values * units('m^2/s^2') / (9.80665 * units('m/s^2'))).to(units.m)
    
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
    # Main Skew-T Panel
    skew = SkewT(fig, rotation=45, rect=(0.12, 0.1, 0.60, 0.85))
    skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2.5, label='Temp')
    skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2.5, label='Dewpoint')
    skew.plot_barbs(p.to(units.hPa)[::5], u[::5], v[::5]) # Reduced density for clarity
    
    # Altitude Markers (Meters) on the pressure axis
    for height_m in [1000, 2000, 3000, 4000, 5500, 7000, 9000]:
        idx = (np.abs(h.m - height_m)).argmin()
        if p[idx].to(units.hPa).m > 110:
            skew.ax.text(-38, p[idx].to(units.hPa).m, f"{height_m}m", 
                         color='gray', fontsize=9, ha='right', va='center', weight='bold')

    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 40)
    skew.ax.set_ylabel("Pressure (hPa) / Altitude (m)")
    skew.ax.set_xlabel("Temperature (°C)")

    # Wind Speed Graph (Right panel)
    ax_wind = fig.add_axes([0.75, 0.1, 0.2, 0.85], sharey=skew.ax)
    ax_wind.plot(wind_speed, p.to(units.hPa), 'k-', linewidth=1.8)
    ax_wind.set_xlabel("Windspeed (kt)")
    ax_wind.grid(True, alpha=0.3)
    ax_wind.set_xlim(0, max(50, wind_speed.m.max() + 10))
    plt.setp(ax_wind.get_yticklabels(), visible=False) # Share vertical axis
    
    plt.suptitle(f"ICON-CH1 Sounding | Valid: {selected_run.strftime('%Y-%m-%d %H:%M')} UTC", fontsize=15, weight='bold')
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print(f"Success! Plot saved for {selected_run.hour}h UTC run.")

if __name__ == "__main__":
    main()
