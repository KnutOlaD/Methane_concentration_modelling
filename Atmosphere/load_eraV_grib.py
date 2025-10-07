'''
script that loads the eraV grib data, creates some plots and saves the data in a netcdf file

author: Knut Ola Dølven
'''

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import pickle
import xarray as xr
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *

#change plotting style to dark
plt.style.use('dark_background')

##############################
### LOAD DATA USING xarray ###
##############################

ds = xr.open_dataset(ATMOSPHERE_PATHS['maydata'], engine='cfgrib')

u10May = ds['u10']
v10May = ds['v10']
sstMay = ds['sst']
timeMay = ds['time']
lonsMay = ds['longitude']
latsMay = ds['latitude']

#get arrays
u10_array_MAY = u10May.values
v10_array_MAY = v10May.values
sst_array_MAY = sstMay.values
time_array_MAY = timeMay.values
lonsMay = lonsMay.values
latsMay = latsMay.values
#calculate magnitude
wsMay = np.sqrt(u10_array_MAY**2 + v10_array_MAY**2)


ds = xr.open_dataset(ATMOSPHERE_PATHS['junedata'], engine='cfgrib')

u10June = ds['u10']
v10June = ds['v10']
sstJune = ds['sst']
timeJune = ds['time']
lonsJune = ds['longitude']  
latsJune = ds['latitude']

#get arrays
u10_array_JUNE = u10June.values
v10_array_JUNE = v10June.values
sst_array_JUNE = sstJune.values
time_array_JUNE = timeJune.values
lonsJune = lonsJune.values
latsJune = latsJune.values

#calculate magnitude
wsJune = np.sqrt(u10_array_JUNE**2 + v10_array_JUNE**2)


# Merge the data for MAY and JUNE
u10merge = np.concatenate((u10_array_MAY, u10_array_JUNE), axis=0)
v10merge = np.concatenate((v10_array_MAY, v10_array_JUNE), axis=0)
sstmerge = np.concatenate((sst_array_MAY, sst_array_JUNE), axis=0)
timemerge = np.concatenate((time_array_MAY, time_array_JUNE), axis=0)

wsmerge = np.sqrt(u10merge**2 + v10merge**2)




#save these to a pickle file
import pickle
with open(ATMOSPHERE_PATHS['output_path'], 'wb') as f:
    pickle.dump([lonsJune, latsJune, timemerge, sstmerge, u10merge, v10merge, wsmerge], f)

#load the pickel file
#with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\ERAV_all_2018.pickle', 'rb') as f:
#    lons, lats, times, sst, u10, v10 = pickle.load(f)

ws = wsmerge
lons = lonsJune
lats = latsJune
times = timemerge

#limits
levels = np.arange(0, 21, 2)

colormap = plt.cm.get_cmap('magma', 20)

import scipy.ndimage

zoom_factor = 3

#make a figure using contour plots instead
#make a figure using contour plots instead
plt.figure(figsize = (7, 7))
ws_zoomed = scipy.ndimage.zoom(np.mean(ws,axis=0), zoom_factor)
lats_zoomed = scipy.ndimage.zoom(lats, zoom_factor)
lons_zoomed = scipy.ndimage.zoom(lons, zoom_factor)
plt.contourf(lons_zoomed, lats_zoomed, ws_zoomed,levels = np.arange(0, 20, 2))
#set colormap
plt.set_cmap(colormap)
cbar = plt.colorbar()
cbar.set_label('[m/s]')
plt.title('Average wind speed [m/s], May 20 - June 20, 2018')
#add contourlines
contour = plt.contour(lons_zoomed, lats_zoomed, ws_zoomed, levels = np.arange(0, 20, 2), colors = 'w', linewidths = 0.5)
#add labels to the contours
plt.clabel(contour, inline=True, fontsize=10)
#put labels on the x axis , y axis and the colorbar
plt.xlabel('Longitude')
plt.ylabel('Latitude')
#add colorbar and set its label

plt.show()

#create a gif of the wind field by looping over the time dimension
import imageio
import matplotlib.gridspec as gridspec

### IF PLOTTING MODEL DOMAIN WIND DATA ###
#load the model pickle file to get the data
#with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\interpolated_wind_sst_fields_test.pickle', 'rb') as f:
#    ws_interp,sst_interp,bin_x_mesh,bin_y_mesh,ocean_time_unix = pickle.load(f)

#levels_w = np.arange(-1, 24, 2)
#levels_sst = np.arange(np.round(np.nanmin(sst_interp))-2, np.round(np.nanmax(sst_interp))+1, 1)
#do the same plot but just on lon lat coordinates
#convert bin_x_mesh and bin_y_mesh to lon/lat
#lat_mesh,lon_mesh = utm.to_latlon(bin_x_mesh,bin_y_mesh,zone_number=33,zone_letter='V')
#colormap = 'magma'

#ws = ws_interp #switch to sst if you want to plot that instead. 

###########################

zoom_factor = 3

images = []
time_steps  = ws.shape[0]
for i in range(time_steps):
    ws_zoomed = scipy.ndimage.zoom(ws[i], zoom_factor)
    lats_zoomed = scipy.ndimage.zoom(lats, zoom_factor)
    lons_zoomed = scipy.ndimage.zoom(lons, zoom_factor)

    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.05])  # Create a GridSpec object

    ax1 = plt.subplot(gs[0])  # Create the first subplot for the contour plot
    contourf = ax1.contourf(lons_zoomed, lats_zoomed, ws_zoomed, levels=levels)
    cbar = plt.colorbar(contourf, ax=ax1)
    cbar.set_label('[m/s]')
    cbar.set_ticks(np.arange(0,21,2))
    ax1.set_title('Wind speed, '+str(times[i])[:10])
    contour = ax1.contour(lons_zoomed, lats_zoomed, ws_zoomed, levels = levels, colors = 'w', linewidths = 0.2)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    ax2 = plt.subplot(gs[1])  # Create the second subplot for the progress bar
    ax2.set_position([0.12,0.12,0.6246,0.03])
    ax2.set_xlim(0, time_steps)  # Set the limits to match the number of time steps
    #ax2.plot([i, i], [0, 1], color='w')  # Plot a vertical line at the current time step
    ax2.fill_between([0, i], [0, 0], [1, 1], color='grey')
    ax2.set_yticks([])  # Hide the y-axis ticks
    ax2.set_xticks([0, time_steps])  # Set the x-axis ticks at the start and end
    ax2.set_xticklabels(['May 20, 2018', 'June 20, 2018'])  # Set the x-axis tick labels to the start and end time

    #set tight layout
    #plt.tight_layout()
    #save the figure
    plt.savefig(ATMOSPHERE_PATHS['results_path'] / 'create_gif' / f'wind_field{i}.png')
    images.append(imageio.imread(ATMOSPHERE_PATHS['results_path'] / 'create_gif' / f'wind_field{i}.png'))
    plt.close()

imageio.mimsave(ATMOSPHERE_PATHS['results_path'] / 'create_gif' / 'wind_field.gif', images, duration = 0.1)

#make a plot that shows average wind speed over time for the whole field
plt.figure(figsize = (7, 7))
plt.plot(times, np.mean(ws, axis=(1,2)),linewidth = 2, color = 'w')
plt.xlabel('Time')
plt.ylabel('Wind speed [m/s]')
plt.title('Average wind speed whole domain')
plt.show()

#save figure
plt.savefig(ATMOSPHERE_PATHS['results_path'] / 'wind_speed_over_time.png')



