import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from meteodatalab import ogd_api
import xarray as xr
import numpy as np
import datetime

# --- Configuration ---
LAT_TARGET, LON_TARGET = 46.81, 6.94 
CORE_VARS = ["T", "U", "V", "P", "FI"] # FI is Geopotential for height calculation
FORECAST_STEPS = range(0, 27, 3) 

def get_nearest_profile(ds, lat_target, lon_target):
    if ds is None: return None
    data = ds if isinstance(ds, xr.DataArray) else ds[list(ds.data_vars)[0]]
    lat_coord = 'latitude' if 'latitude' in data.coords else 'lat'
    lon_coord = 'longitude' if 'longitude' in data.coords else 'lon'
    horiz_dims = data.coords[lat_coord].dims
    dist = (data[lat_coord] - lat_target)**2 + (data[lon_coord] - lon_target)**2
    flat_idx = dist.argmin().values
    profile = data.isel({horiz_dims[0]: flat_idx}) if len(horiz_dims) == 1 else data.stack(gp=horiz_dims).isel(gp=flat_idx)
    return profile.squeeze().compute()

def main():
    print("Connecting to MeteoSwiss for Advanced ICON-CH1 Forecast Series...")
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    times_to_try = [latest_standard - datetime.timedelta(hours=i*3) for i in range(4)]
    
    selected_run = None
    for ref_time in times_to_try:
        try:
            req_test = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable="T",
                                       reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
            if ogd_api.get_from_ogd(req_test) is not None:
                selected_run = ref_time
                break
        except: continue

    if not selected_run: return

    for step in FORECAST_STEPS:
        horizon_str = f"P0DT{step}H"
        try:
            profile_data = {}
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                      reference_datetime=selected_run, horizon=horizon_str, perturbed=False)
                profile_data[var] = get_nearest_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET)
            
            # Humidity Fallback
            for hum_var in ["RELHUM", "QV"]:
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum_var,
                                            reference_datetime=selected_run, horizon=horizon_str, perturbed=False)
                    profile_data["HUM"], profile_data["HUM_TYPE"] = get_nearest_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET), hum_var
                    break
                except: continue

            # --- Data Processing ---
            p = profile_data["P"].values * units.Pa
            t = profile_data["T"].values * units.K
            u = profile_data["U"].values * units('m/s')
            v = profile_data["V"].values * units('m/s')
            # Calculate height in meters from Geopotential
            h = (profile_data["FI"].values * units('m^2/s^2') / (9.80665 * units('m/s^2'))).to(units.m)
            
            if profile_data["HUM_TYPE"] == "RELHUM":
                td = mpcalc.dewpoint_from_relative_humidity(t, profile_data["HUM"].values / 100.0)
            else:
                td = mpcalc.dewpoint_from_specific_humidity(p, t, profile_data["HUM"].values * units('kg/kg'))

            inds = p.argsort()[::-1]
            p, t, td, u, v, h = p[inds], t[inds], td[inds], u[inds], v[inds], h[inds]
            wind_speed = mpcalc.wind_speed(u, v).to(units.knot)

            # --- Multi-Panel Plotting ---
            fig = plt.figure(figsize=(12, 10))
            # Main Skew-T Panel
            skew = SkewT(fig, rotation=45, rect=(0.1, 0.1, 0.65, 0.85))
            skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2, label='Temp')
            skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2, label='Dewpoint')
            skew.plot_barbs(p.to(units.hPa)[::4], u[::4], v[::4])
            
            # Add height labels in meters on the left of the Pressure axis
            for height in [1000, 2000, 3000, 5000, 7000, 9000]:
                idx = (np.abs(h.m - height)).argmin()
                skew.ax.text(-38, p[idx].to(units.hPa).m, f"{height}m", 
                             color='gray', fontsize=9, ha='right', va='center')

            skew.ax.set_ylim(1050, 100)
            skew.ax.set_xlim(-35, 35)

            # Wind Speed Panel (Right side)
            ax_wind = fig.add_axes([0.76, 0.1, 0.2, 0.85], sharey=skew.ax)
            ax_wind.plot(wind_speed, p.to(units.hPa), 'k-', linewidth=1.5)
            ax_wind.set_xlabel("Windspeed (kt)")
            ax_wind.grid(True)
            ax_wind.set_xlim(0, 60) # Typical max wind range
            
            valid_time = selected_run + datetime.timedelta(hours=step)
            plt.suptitle(f"ICON-CH1 | Run: {selected_run.strftime('%H:%M')} | Valid: {valid_time.strftime('%Y-%m-%d %H:%M')} UTC", fontsize=15)
            
            plt.savefig(f"skewt_f{step:02d}.png", bbox_inches='tight', dpi=120)
            plt.close()
            print(f"Saved step +{step}h")
            
        except Exception as e: print(f"Error at +{step}h: {e}")

if __name__ == "__main__":
    main()
