import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from meteodatalab import ogd_api
import xarray as xr
import numpy as np
import datetime

# --- Configuration ---
LAT, LON = 46.81, 6.94  # Payerne
# CORE_VARS must include 'FI' (Geopotential) for altitude calculation
VARIABLES = ["T", "RELHUM", "U", "V", "P", "FI"] 

def get_nearest_profile(ds, lat_target, lon_target):
    """Safely extracts a vertical profile using the user's working architecture."""
    if ds is None: return None
    
    # FIX: Handle cases where the API returns a DataArray instead of a Dataset
    if isinstance(ds, xr.Dataset):
        if not ds.data_vars: return None
        data = ds[list(ds.data_vars)[0]]
    else:
        data = ds

    # Finding nearest point using the architecture that worked previously
    lat_dim = 'latitude' if 'latitude' in data.coords else 'lat'
    lon_dim = 'longitude' if 'longitude' in data.coords else 'lon'
    dist = (data[lat_dim] - lat_target)**2 + (data[lon_dim] - lon_target)**2
    flat_idx = dist.argmin().values
    
    # Stack horizontal dims (last two) and select index
    profile = data.stack(gp=data.dims[-2:]).isel(gp=flat_idx).squeeze().compute()
    return profile

def main():
    print("Connecting to MeteoSwiss (Stable Architecture + New Features)...")
    now = datetime.datetime.now(datetime.timezone.utc)
    base_h = (now.hour // 3) * 3
    latest_standard = now.replace(hour=base_h, minute=0, second=0, microsecond=0)
    
    # Fallback logic: check 'latest', then standard runs
    times_to_try = ["latest"] + [latest_standard - datetime.timedelta(hours=i*3) for i in range(4)]
    
    success, profile_data, final_run_time = False, {}, None
    for ref_time in times_to_try:
        print(f"--- Attempting Run: {ref_time} ---")
        try:
            batch = {}
            for var in VARIABLES:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                      reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                res = get_nearest_profile(ogd_api.get_from_ogd(req), LAT, LON)
                if res is None: raise ValueError(f"Empty {var}")
                batch[var] = res
            
            profile_data, success, final_run_time = batch, True, ref_time
            break
        except Exception as e: print(f"Run incomplete: {e}")

    if not success: return

    # --- Processing Data ---
    p = profile_data["P"].values * units.Pa
    t = profile_data["T"].values * units.K
    u, v = profile_data["U"].values * units('m/s'), profile_data["V"].values * units('m/s')
    # Height in meters = Geopotential / standard gravity
    h = (profile_data["FI"].values * units('m^2/s^2') / (9.80665 * units('m/s^2'))).to(units.m)
    rh = profile_data["RELHUM"].values / 100.0
    
    td = mpcalc.dewpoint_from_relative_humidity(t, rh)
    wind_speed = mpcalc.wind_speed(u, v).to(units.knot)

    # Sort ground-to-sky (descending pressure)
    inds = p.argsort()[::-1]
    p, t, td, u, v, h, wind_speed = p[inds], t[inds], td[inds], u[inds], v[inds], h[inds], wind_speed[inds]

    # --- Multi-Panel Plotting ---
    fig = plt.figure(figsize=(12, 10))
    # Main Skew-T Panel
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.65, 0.85))
    skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2.5, label='Temp')
    skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2.5, label='Dewpoint')
    skew.plot_barbs(p.to(units.hPa)[::5], u[::5], v[::5]) #
    
    # Add Altitude Markers (m) to the pressure axis
    for height in [1000, 2000, 3000, 5000, 7000, 9000]:
        idx = (np.abs(h.m - height)).argmin()
        if p[idx].to(units.hPa).m > 120:
            skew.ax.text(-38, p[idx].to(units.hPa).m, f"{height}m", color='blue', alpha=0.6, ha='right', weight='bold')

    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 40)
    skew.ax.set_ylabel("Pressure (hPa) / Altitude (m)")

    # Wind Speed Panel (Right)
    ax_wind = fig.add_axes([0.78, 0.1, 0.18, 0.85], sharey=skew.ax)
    ax_wind.plot(wind_speed, p.to(units.hPa), 'blue', linewidth=1.8)
    ax_wind.set_xlabel("Speed (kt)")
    ax_wind.grid(True, alpha=0.3)
    ax_wind.set_xlim(0, max(50, wind_speed.m.max() + 10))
    plt.setp(ax_wind.get_yticklabels(), visible=False)

    plt.suptitle(f"ICON-CH1 Sounding | Run: {final_run_time} UTC", fontsize=15, weight='bold')
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print("Success! saved latest_skewt.png")

if __name__ == "__main__":
    main()
