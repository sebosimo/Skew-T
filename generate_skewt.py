import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from meteodatalab import ogd_api
import xarray as xr
import numpy as np
import datetime
import os

# --- Configuration ---
LAT_TARGET, LON_TARGET = 46.81, 6.94  # Payerne
CORE_VARS = ["T", "U", "V", "P"] 
FORECAST_STEPS = range(0, 27, 3) # 0, 3, 6, ..., 24 hours

def get_nearest_profile(ds, lat_target, lon_target):
    """Robustly extracts a vertical profile from the horizontal grid."""
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
    print("Connecting to MeteoSwiss for ICON-CH1 Forecast Series...")
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # Strategy: Find a complete run initialization first
    times_to_try = [latest_run - datetime.timedelta(hours=i*3) for i in range(4)]
    
    selected_run = None
    for ref_time in times_to_try:
        # Check if the most essential variable (T) exists for the final step (+24h)
        # If +24h is available, the whole run is likely ready
        try:
            req_test = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable="T",
                                       reference_datetime=ref_time, horizon="P0DT24H", perturbed=False)
            if ogd_api.get_from_ogd(req_test) is not None:
                selected_run = ref_time
                break
        except: continue

    if not selected_run:
        print("Error: No complete model runs with 24h forecasts found.")
        return

    print(f"Generating sequence for Model Run: {selected_run.strftime('%H:%M')} UTC")

    for step in FORECAST_STEPS:
        horizon_str = f"P0DT{step}H"
        print(f"Processing Forecast Step: +{step}h ({horizon_str})...")
        
        try:
            profile_data = {}
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                      reference_datetime=selected_run, horizon=horizon_str, perturbed=False)
                profile_data[var] = get_nearest_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET)
            
            for hum_var in ["RELHUM", "QV"]:
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum_var,
                                            reference_datetime=selected_run, horizon=horizon_str, perturbed=False)
                    profile_data["HUM"] = get_nearest_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET)
                    profile_data["HUM_TYPE"] = hum_var
                    break
                except: continue

            # --- Plotting ---
            p = profile_data["P"].values * units.Pa
            t = profile_data["T"].values * units.K
            if profile_data["HUM_TYPE"] == "RELHUM":
                td = mpcalc.dewpoint_from_relative_humidity(t, profile_data["HUM"].values / 100.0)
            else:
                td = mpcalc.dewpoint_from_specific_humidity(p, t, profile_data["HUM"].values * units('kg/kg'))

            inds = p.argsort()[::-1]
            p, t, td = p[inds], t[inds], td[inds]

            fig = plt.figure(figsize=(10, 12))
            skew = SkewT(fig, rotation=45)
            skew.plot(p.to(units.hPa), t.to(units.degC), 'r', linewidth=2.5, label='Temp')
            skew.plot(p.to(units.hPa), td.to(units.degC), 'g', linewidth=2.5, label='Dewpoint')
            skew.plot_barbs(p.to(units.hPa)[::3], profile_data["U"].values[inds][::3], profile_data["V"].values[inds][::3])
            
            skew.plot_dry_adiabats(alpha=0.1, color='red')
            skew.plot_moist_adiabats(alpha=0.1, color='blue')
            skew.plot_mixing_lines(alpha=0.1, color='green')
            
            skew.ax.set_ylim(1050, 100)
            skew.ax.set_xlim(-40, 40)
            
            valid_time = selected_run + datetime.timedelta(hours=step)
            plt.title(f"ICON-CH1 | Run: {selected_run.strftime('%H:%M')} | Forecast: +{step}h\nValid: {valid_time.strftime('%Y-%m-%d %H:%M')} UTC", fontsize=14)
            plt.legend(loc='upper left')
            
            # Save file with step index
            plt.savefig(f"skewt_f{step:02d}.png", bbox_inches='tight', dpi=120)
            plt.close() # Important: close plot to save memory
            
        except Exception as e:
            print(f"Failed step +{step}h: {e}")

    print("Success! Generated forecast sequence.")

if __name__ == "__main__":
    main()
