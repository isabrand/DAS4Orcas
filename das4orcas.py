# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# ## DAS4Orcas first Widbey Island visualization attempt

# pip install autoreload if not installed
# import autoreload
# # %load_ext autoreload
# # %autoreload 2
# Imports
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import das4whales as dw # Import the das4whales library, if not installed refer to https://das4whales.readthedocs.io/en/latest/src/install.html

# +
# filepath = 'data/decimator_2024-10-25_23.44.00_UTC_006608.h5'
# filename = 'decimator_22024-10-25_23.44.00_UTC_006608.h5'

# filepath = 'data/decimator_2024-10-29_20.26.00_UTC_012170.h5'
# filename = 'decimator_2024-10-29_20.26.00_UTC_012170.h5'

# filepath = 'data/decimator_2024-10-25_23.46.00_UTC_006610.h5'
# filename = 'decimator_22024-10-25_23.46.00_UTC_006610.h5'

# filepath = 'data/decimator_2024-10-25_23.47.00_UTC_006611.h5'
# filename = 'decimator_22024-10-25_23.47.00_UTC_006611.h5'

# filepath = 'data/decimator_2024-10-25_23.48.00_UTC_006612.h5'
# filename = 'decimator_22024-10-25_23.48.00_UTC_006612.h5'

# filepath = 'data/decimator_2024-10-25_23.49.00_UTC_006613.h5'
# filename = 'decimator_22024-10-25_23.49.00_UTC_006613.h5'
# -

filepath = 'data/decimator_2025-07-21_19.00.00_UTC_126827.h5'
filename = 'decimator_2025-07-21_19.00.00_UTC_126827.h5'

# +
# Read HDF5 files and access metadata
# Get the acquisition parameters for the data folder
metadata = dw.data_handle.get_acquisition_parameters(filepath, interrogator='onyx')
fs, dx, nx, ns, gauge_length, scale_factor = metadata["fs"], metadata["dx"], metadata["nx"], metadata["ns"], metadata["GL"], metadata["scale_factor"]

print(f'Sampling frequency: {metadata["fs"]} Hz')
print(f'Channel spacing: {metadata["dx"]} m')
print(f'Gauge length: {metadata["GL"]} m')
print(f'File duration: {metadata["ns"] / metadata["fs"]} s')
print(f'Cable max distance: {metadata["nx"] * metadata["dx"]/1e3:.1f} km')
print(f'Number of channels: {metadata["nx"]}')
print(f'Number of time samples: {metadata["ns"]}')

# +
selected_channels_m_north = [0, 15000, 7]  # list of values in meters corresponding to the starting,
                                                # ending and step wanted channels along the FO Cable
                                                # selected_channels_m = [ChannelStart_m, ChannelStop_m, ChannelStep_m]
                                                # in meters

                                                # ### Select the desired channels and channel interval

selected_channels = [int(selected_channels_m // dx) for selected_channels_m in
                selected_channels_m_north]  # list of values in channel number (spatial sample) corresponding to the starting, ending and step wanted
                                        # channels along the FO Cable
                                        # selected_channels = [ChannelStart, ChannelStop, ChannelStep] in channel
                                        # numbers

print('Begin channel #:', selected_channels[0], 
', End channel #: ',selected_channels[1], 
', step: ',selected_channels[2], 
'equivalent to ',selected_channels[2]*dx,' m')
# -

tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)
metadata["fileBeginTimeUTC"] = fileBeginTimeUTC.strftime("%Y-%m-%d_%H:%M:%S")

# Bandpass the data between 0.1 and 1 Hz
b, a = sp.butter(4, [400,950], 'bp', fs=fs)
# b, a = sp.butter(2, 30, 'high', fs=fs)
tr = sp.filtfilt(b, a, tr, axis=1)

# Useful commands to free memory (jupyter notebook does not free memory after a cell is executed)
import gc
gc.collect()

# Adapt the scale of the data to the plot
min_tr = np.min(tr)
max_tr = np.max(tr)
scale = np.min([np.abs(min_tr), np.abs(max_tr)]) *0.0000001

# Plot the data
plt.figure(figsize=(10, 6))
plt.imshow(tr, origin='lower', aspect='auto', cmap='RdBu_r', extent=[0, time[-1], dist[0], dist[-1]], vmin=-scale, vmax=scale)
plt.ylabel('Optical distance (m)')
plt.xlabel('Time (s)')
# plt.ylim(3500,4000)
# plt.xlim(45, 52)
plt.colorbar(label='Strain [-]')
plt.show()
