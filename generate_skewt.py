import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from meteodatalab import ogd_api
import xarray as xr
import numpy as np
import datetime

# --- Configuration ---
LAT_TARGET, LON_TARGET = 46.81, 6.94  # Payerne
CORE_VARS = ["T", "U", "V", "P"]

def get_nearest_profile(ds, lat_target, lon_target):
    """Correctly extracts a vertical profile from regular or native ICON grids."""
    if ds is None: return None
    data = ds if isinstance(ds, xr.DataArray) else ds[list(ds.data_vars)[0]]
    
    # 1. Identify coordinate and dimension names
    lat_coord = 'latitude' if 'latitude' in data.coords else 'lat'
    lon_coord = 'longitude' if 'longitude' in data.coords else 'lon'
    
    # 2. Find the horizontal dimension
    horiz_dims = data.coords[lat_coord].dims
    
    # 3. Calculate distance to find the closest horizontal point
    dist = (data[lat_coord] - lat_target)**2 + (data[lon_coord] - lon_target)**2
    flat_idx = dist.argmin().values
    
    # 4. Extract column
    if len(horiz_dims) == 1:
        profile = data.isel({horiz_dims[0]: flat_idx})
    else:
        profile = data.stack(gp=horiz_dims).isel(gp=flat_idx)
        
    return profile.squeeze().compute()

def format_pressure_as_km(x, pos):
    """Converts Y-axis Pressure (hPa) to Standard Atmosphere Height (km)."""
    if x <= 0: return ""
    height = mpcalc.pressure_to_height_std(x * units.hPa).to('km')
    return f"{height.m:.1f}"

def main():
    print("Fetching ICON-CH1 data from MeteoSwiss...")
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # Try the last 4 runs to find complete data
    times_to_try = [latest_run - datetime.timedelta(hours=i*3) for i in range(4)]
    
    success, profile_data, ref_time_final = False, {}, None
    for ref_time in times_to_try:
        print(f"--- Attempting Run: {ref_time.strftime('%H:%M')} UTC ---")
        try:
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                     reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                res = get_nearest_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET)
                if res is None or res.size < 5: raise ValueError(f"Empty {var}")
                profile_data[var] = res
            
            # Fetch Humidity with fallback
            for hum_var in ["RELHUM", "QV"]:
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum_var,
                                           reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                    res_h = get_nearest_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET)
                    if res_h is not None and res_h.size >= 5:
                        profile_data["HUM"], profile_data["HUM_TYPE"] = res_h, hum_var
                        break
                except: continue
            
            if "HUM" not in profile_data: raise ValueError("No Humidity")
            success, ref_time_final = True, ref_time
            break 
        except Exception as e: print(f"Run incomplete: {e}")

    if not success:
        print("Error: No complete model runs found.")
        return

    # --- Unit Conversion & Calculation ---
    p = profile_data["P"].values * units.Pa
    t = (profile_data["T"].values * units.K).to(units.degC)
    u = profile_data["U"].values * units('m/s')
    v = profile_data["V"].values * units('m/s')
    
    # Calculate Dewpoint
    if profile_data["HUM_TYPE"] == "RELHUM":
        td = mpcalc.dewpoint_from_relative_humidity(t, profile_data["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, profile_data["HUM"].values * units('kg/kg'))

    # Sort descending (Surface -> Space)
    inds = p.argsort()[::-1] 
    p, t, td, u, v = p[inds], t[inds], td[inds], u[inds], v[inds]
    p_hpa = p.to(units.hPa)

    # Calculate Wind Speed in km/h for the side panel
    wind_speed = mpcalc.wind_speed(u, v).to(units('km/h'))

    # Calculate Parcel Profile
    parcel_prof = mpcalc.parcel_profile(p_hpa, t[0], td[0]).to('degC')

    # --- Plotting Setup ---
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)

    # 1. Skew-T Panel (Left)
    skew = SkewT(fig, rotation=45, subplot=gs[0])

    # Plot Temperature and Dewpoint
    skew.plot(p_hpa, t, 'r', linewidth=2.5, label='Temperature')
    skew.plot(p_hpa, td, 'g', linewidth=2.5, label='Dewpoint')
    
    # Plot Parcel Path
    skew.plot(p_hpa, parcel_prof, 'k', linestyle='--', linewidth=1.5, label='Surface Parcel')
    skew.shade_cape(p_hpa, t, parcel_prof)
    skew.shade_cin(p_hpa, t, parcel_prof, td)

    # Plot barbs (Note: Removed zorder to fix TypeError)
    # Using every 5th level to avoid clutter
    skew.plot_barbs(p_hpa[::5], u[::5], v[::5])

    # Standard Skew-T background lines
    skew.plot_dry_adiabats(alpha=0.1, color='red')
    skew.plot_moist_adiabats(alpha=0.1, color='blue')
    skew.plot_mixing_lines(alpha=0.1, color='green')
    
    # Set Limits
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 40)
    
    # 2. Wind Speed Panel (Right)
    ax_wind = fig.add_subplot(gs[1], sharey=skew.ax)
    ax_wind.plot(wind_speed, p_hpa, color='purple', linewidth=2)
    
    # Formatting the Wind Panel
    ax_wind.set_xlabel("Wind Speed [km/h]")
    ax_wind.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_wind.set_yscale('log')
    ax_wind.set_ylim(1050, 100)
    plt.setp(ax_wind.get_yticklabels(), visible=False)
    
    # 3. Y-Axis Label Transformation (hPa -> km)
    skew.ax.set_ylabel("Altitude (km) [Std. Atm]")
    pressure_levels = [1000, 850, 700, 500, 400, 300, 200, 100]
    skew.ax.set_yticks(pressure_levels)
    skew.ax.yaxis.set_major_formatter(FuncFormatter(format_pressure_as_km))
    
    # Title & Save
    plt.suptitle(f"ICON-CH1 Sounding (Payerne) | {ref_time_final.strftime('%Y-%m-%d %H:%M')} UTC", fontsize=16, y=0.92)
    skew.ax.legend(loc='upper left')
    
    output_filename = "latest_skewt.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"Success! {output_filename} saved.")

if __name__ == "__main__":
    main()
