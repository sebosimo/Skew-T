import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from meteodatalab import ogd_api
import xarray as xr
import numpy as np
import datetime
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# --- Configuration ---
LAT_TARGET, LON_TARGET = 46.81, 6.94  # Payerne
CORE_VARS = ["T", "U", "V", "P"] 

def get_nearest_profile(ds, lat_target, lon_target):
    """Correctly extracts a vertical profile from regular or native ICON grids."""
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

# --- Axis Conversion Functions ---
def p_to_h(p):
    return (44330 * (1 - (p / 1013.25)**(1/5.255)))

def h_to_p(h):
    return (1013.25 * (1 - h / 44330)**5.255)

def main():
    print("Fetching ICON-CH1 data from MeteoSwiss...")
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
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

    # --- Unit Conversion ---
    p = profile_data["P"].values * units.Pa
    t = profile_data["T"].values * units.K
    u = profile_data["U"].values * units('m/s')
    v = profile_data["V"].values * units('m/s')
    wind_speed = mpcalc.wind_speed(u, v).to(units('km/h'))

    if profile_data["HUM_TYPE"] == "RELHUM":
        td = mpcalc.dewpoint_from_relative_humidity(t, profile_data["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, profile_data["HUM"].values * units('kg/kg'))

    inds = p.argsort()[::-1]
    p, t, td, u, v, wind_speed = p[inds], t[inds], td[inds], u[inds], v[inds], wind_speed[inds]

    # --- Plotting ---
    # Taller figure size (10, 12) restores the vertical aspect ratio
    fig = plt.figure(figsize=(10, 12)) 
    
    # Create gridspec with 0 spacing
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    # FORCE zero whitespace between subplots
    fig.subplots_adjust(wspace=0)

    # 1. Skew-T Panel
    ax_skew = fig.add_subplot(gs[0])
    skew = SkewT(fig, rotation=45, subplot=ax_skew)
    
    skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2.5, label='Temperature')
    skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2.5, label='Dewpoint')
    skew.plot_barbs(p.to(units.hPa)[::4], u[::4], v[::4]) 
    
    skew.plot_dry_adiabats(alpha=0.1, color='red')
    skew.plot_moist_adiabats(alpha=0.1, color='blue')
    skew.plot_mixing_lines(alpha=0.1, color='green')
    
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-35, 35)
    
    # Hide default pressure labels and axis title to prevent overlap
    skew.ax.yaxis.set_major_formatter(plt.NullFormatter())
    skew.ax.set_ylabel("") 
    
    # Altitude Axis (Left)
    secax = skew.ax.secondary_yaxis('left', functions=(p_to_h, h_to_p))
    secax.set_ylabel('Altitude [m]', fontsize=11)
    secax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    secax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    secax.tick_params(axis='y', length=6, direction='out')

    plt.title(f"ICON-CH1 Sounding | {ref_time_final.strftime('%Y-%m-%d %H:%M')} UTC", fontsize=14)
    skew.ax.legend(loc='upper left')

    # 2. Wind Speed Panel (Right)
    ax_wind = fig.add_subplot(gs[1], sharey=skew.ax)
    ax_wind.plot(wind_speed, p.to(units.hPa), 'b-', linewidth=2)
    ax_wind.set_xlabel('Wind [km/h]', fontsize=10)
    ax_wind.grid(True)
    
    # Hide Y-axis labels entirely for the wind plot
    ax_wind.yaxis.set_major_formatter(plt.NullFormatter())
    ax_wind.set_ylabel("")
    plt.setp(ax_wind.get_yticklabels(), visible=False)

    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print("Success! Skew-T saved.")

if __name__ == "__main__":
    main()
