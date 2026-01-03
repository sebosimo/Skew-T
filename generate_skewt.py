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
# Adding 'FI' (Geopotential) to the working variable set for altitude
VARIABLES = ["T", "RELHUM", "U", "V", "P", "FI"] 

def main():
    print("Connecting to MeteoSwiss (Working Architecture)...")
    profile_data = {}
    
    # Using 'latest' as the primary target since it worked previously
    # Fallback to 3h ago if 'latest' is still uploading
    times_to_try = ["latest", (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=3)).strftime('%Y-%m-%dT%H:00:00Z')]

    success = False
    for ref_time in times_to_try:
        print(f"--- Attempting Run: {ref_time} ---")
        try:
            for var in VARIABLES:
                req = ogd_api.Request(
                    collection="ogd-forecasting-icon-ch1",
                    variable=var,
                    reference_datetime=ref_time,
                    horizon="P0DT0H",
                    perturbed=False 
                )
                ds = ogd_api.get_from_ogd(req)
                
                # Reverting to the architecture that worked for you previously
                var_key = list(ds.data_vars)[0]
                data = ds[var_key]
                
                # Finding nearest point using your previously working architecture
                lat_dim = 'latitude' if 'latitude' in data.coords else 'lat'
                lon_dim = 'longitude' if 'longitude' in data.coords else 'lon'
                dist = (data[lat_dim] - LAT)**2 + (data[lon_dim] - LON)**2
                flat_idx = dist.argmin().values
                
                # Stack horizontal dims and select index
                profile_data[var] = data.stack(gp=data.dims[-2:]).isel(gp=flat_idx).squeeze().compute()
            
            success = True
            break
        except Exception as e:
            print(f"Run {ref_time} incomplete: {e}")

    if not success:
        print("Error: Could not find a complete model run.")
        return

    # --- Data Processing ---
    p = profile_data["P"].values * units.Pa
    t = profile_data["T"].values * units.K
    u, v = profile_data["U"].values * units('m/s'), profile_data["V"].values * units('m/s')
    rh = profile_data["RELHUM"].values / 100.0
    # Height Z = Geopotential / Gravity
    h = (profile_data["FI"].values * units('m^2/s^2') / (9.80665 * units('m/s^2'))).to(units.m)
    
    td = mpcalc.dewpoint_from_relative_humidity(t, rh)
    wind_speed = mpcalc.wind_speed(u, v).to(units.knot)

    # Sort data for MetPy plotting
    inds = p.argsort()[::-1]
    p, t, td, u, v, h, wind_speed = p[inds], t[inds], td[inds], u[inds], v[inds], h[inds], wind_speed[inds]

    # --- Multi-Panel Plotting ---
    fig = plt.figure(figsize=(12, 10))
    # Main Skew-T (Left)
    skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.65, 0.85))
    skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2.5, label='Temp')
    skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2.5, label='Dewpoint')
    skew.plot_barbs(p.to(units.hPa)[::5], u[::5], v[::5])
    
    # Altitude Markers (Meters) on the vertical axis
    for m in [1500, 3000, 5000, 7000, 9000]:
        idx = (np.abs(h.m - m)).argmin()
        if p[idx].to(units.hPa).m > 120:
            skew.ax.text(-38, p[idx].to(units.hPa).m, f"{m}m", color='blue', alpha=0.6, ha='right', weight='bold')

    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 40)
    skew.ax.set_ylabel("Pressure (hPa) / Altitude (m)")

    # Wind Speed Panel (Right)
    ax_wind = fig.add_axes([0.78, 0.1, 0.18, 0.85], sharey=skew.ax)
    ax_wind.plot(wind_speed, p.to(units.hPa), 'blue', linewidth=2)
    ax_wind.set_xlabel("Speed (kt)")
    ax_wind.grid(True, alpha=0.3)
    ax_wind.set_xlim(0, max(50, wind_speed.m.max() + 5))
    plt.setp(ax_wind.get_yticklabels(), visible=False)

    plt.suptitle(f"ICON-CH1 Sounding | Valid: {profile_data['T'].forecast_reference_time} UTC", fontsize=15, weight='bold')
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print("Success! saved latest_skewt.png")

if __name__ == "__main__":
    main()
