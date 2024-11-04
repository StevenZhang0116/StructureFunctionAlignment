import numpy as np
import matplotlib.pyplot as plt 

import scienceplots
plt.style.use('science')
plt.style.use(['no-latex'])

from os.path import join as pjoin
import xarray as xr

import activity_helper

session_scan = [[4,7],[8,5],[6,6],[5,3],[5,6],[5,7],[6,2],[7,3],[7,5],[9,3],[9,4],[6,4]]
all_intervals_duration = []

for ss in session_scan:
    session_info, scan_info = ss[0], ss[1]

    save_filename = f"./microns/functional_xr/functional_session_{session_info}_scan_{scan_info}.nc"
    session_ds = xr.open_dataset(save_filename)

    stimulus_name = pjoin("./microns/",'movie_downsample_xr', f'movie_downsample_session_{session_info}_scan_{scan_info}.nc')
    stimulus_ds = xr.open_dataset(stimulus_name)

    stim_on_time = session_ds.stim_on.to_pandas().values
    intervals = activity_helper.find_intervals(stim_on_time)
    intervals = intervals["1"]
    intervals_duration = [(interval[1] - interval[0]) / stimulus_ds.nframes for interval in intervals]
    all_intervals_duration.extend(intervals_duration)

fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.hist(all_intervals_duration, bins=50, density=False)
fig.savefig("zz_stimulus_analysis.png")
