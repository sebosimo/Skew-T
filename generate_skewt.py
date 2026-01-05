
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
    
    # 2. Find the horizontal dimension (usually 'ncells' or 'grid_index')
    horiz_dims = data.coords[lat_coord].dims
    
    # 3. Calculate distance to find the closest horizontal point
    dist = (data[lat_coord] - lat_target)**2 + (data[lon_coord] - lon_target)**2
    flat_idx = dist.argmin().values
    
    # 4. Extract column: select the index on the horizontal dimension only
    if len(horiz_dims) == 1:
        profile = data.isel({horiz_dims[0]: flat_idx})
    else:
        profile = data.stack(gp=horiz_dims).isel(gp=flat_idx)
        
    return profile.squeeze().compute()

def format_pressure_as_meters(x, pos):
    """
    Callback function for matplotlib ticker.
    Converts Y-axis Pressure (x in hPa) to Standard Atmosphere Height (Meters).
    """
    # Avoid log(0) or negative errors if plot limits get weird
    if x <= 0: return ""
    
    # Calculate standard height from pressure
    height = mpcalc.pressure_to_height_std(x * units.hPa)
    return f"{int(height.m)}"

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
    t = profile_data["T"].values * units.K
    u = profile_data["U"].values * units('m/s')
    v = profile_data["V"].values * units('m/s')
    
    # Calculate Dewpoint
    if profile_data["HUM_TYPE"] == "RELHUM":
        td = mpcalc.dewpoint_from_relative_humidity(t, profile_data["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, profile_data["HUM"].values * units('kg/kg'))

    # Calculate Wind Speed in km/h
    wind_speed = mpcalc.wind_speed(u, v).to(units('km/h'))

    # Sort descending (Surface -> Space)
    inds = p.argsort()[::-1] 
    p, t, td, u, v, wind_speed = p[inds], t[inds], td[inds], u[inds], v[inds], wind_speed[inds]

    # --- Plotting Setup ---
    # Create a figure with GridSpec: 2 columns, Skew-T is wider
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.0)

    # 1. Skew-T Panel (Left)
    skew = SkewT(fig, rotation=45, subplot=gs[0])

    # Plot standard Skew-T layers with zorder
    skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2.5, label='Temperature', zorder=2)
    skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2.5, label='Dewpoint', zorder=2)
    
    # Plot barbs on top with a higher zorder
    skew.plot_barbs(p.to(units.hPa)[::5], u[::5], v[::5], zorder=3)

    skew.plot_dry_adiabats(alpha=0.1, color='red')
    skew.plot_moist_adiabats(alpha=0.1, color='blue')
    skew.plot_mixing_lines(alpha=0.1, color='green')
    
    # Set Limits
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 40)
    
    # 2. Wind Speed Panel (Right)
    ax_wind = fig.add_subplot(gs[1], sharey=skew.ax)
    ax_wind.plot(wind_speed, p.to(units.hPa), color='purple', linewidth=2)
    
    # Formatting the Wind Panel
    ax_wind.set_xlabel("Wind Speed [km/h]")
    ax_wind.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Ensure the wind plot is also Logarithmic to match Skew-T projection
    ax_wind.set_yscale('log')
    ax_wind.set_ylim(1050, 100)

    # Hide the Y-axis labels of the wind plot
    plt.setp(ax_wind.get_yticklabels(), visible=False)
    
    # 3. Y-Axis Label Transformation (hPa -> Meters)
    skew.ax.set_ylabel("Altitude (m) [Std. Atm]")
    
    # Set standard pressure ticks but label them as Height
    pressure_levels = [1000, 900, 850, 800, 700, 600, 500, 400, 300, 200, 150, 100]
    skew.ax.set_yticks(pressure_levels)
    skew.ax.set_yticklabels(pressure_levels)
    
    # Apply the custom formatter
    skew.ax.yaxis.set_major_formatter(FuncFormatter(format_pressure_as_meters))
    
    # Title & Save
    plt.suptitle(f"ICON-CH1 Sounding (Payerne) | {ref_time_final.strftime('%Y-%m-%d %H:%M')} UTC", fontsize=16, y=0.92)
    skew.ax.legend(loc='upper left')
    
    output_filename = "latest_skewt.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"Success! {output_filename} saved.")

if __name__ == "__main__":
    main()
