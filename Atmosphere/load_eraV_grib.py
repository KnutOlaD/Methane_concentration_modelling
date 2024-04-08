'''
script that loads the eraV grib data, creates some plots and saves the data in a netcdf file

author: Knut Ola Dølven
'''

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import pickle
import pygrib
import xarray as xr

#change plotting style to dark
plt.style.use('dark_background')

##############################
### LOAD DATA USING PYGRIB ###
##############################

import xarray as xr

ds = xr.open_dataset('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\ERAV_May_2018.grib', engine='cfgrib')

#get the time values

# Open the GRIB file
with pygrib.open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\ERAV_June_2018.grib') as grb:
    
    # Create lists to store the data and times
    u10_list = []
    v10_list = []
    sst_list = []
    time_list_June = []
    # Iterate over the messages in the GRIB file
    for message in grb:
        print(message)
        pause = input('Press enter to continue')
        #if the time_list more than one hour since the last one, then add the previous fields to 
        #the list and a timestep to the time_array until the time_list is continuous
        # Check the name of the message and append the data to the appropriate list
        if message.name == '10 metre U wind component':
            u10_list.append(message.values)
            time_list_June.append(pd.Timestamp(message.analDate) + pd.Timedelta(hours=message.hour))
        elif message.name == '10 metre V wind component':
            v10_list.append(message.values)
        elif message.name == 'Sea surface temperature':
            sst_list.append(message.values)
    # Convert the lists into 3D numpy arrays and a 1D array
    u10_array_JUNE = np.array(u10_list)
    v10_array_JUNE = np.array(v10_list)
    sst_array_JUNE = np.array(sst_list)
    time_array_JUNE = np.array(time_list_June)

print(u10_array_JUNE.shape)
print(v10_array_JUNE.shape)
print(sst_array_JUNE.shape)
print(time_array_JUNE.shape)

# Get the latitude and longitude values
lats, lons = message.latlons()

# Merge the data for MAY and JUNE
u10_array = np.concatenate((u10_array_MAY, u10_array_JUNE), axis=0)
v10_array = np.concatenate((v10_array_MAY, v10_array_JUNE), axis=0)
sst_array = np.concatenate((sst_array_MAY, sst_array_JUNE), axis=0)
time_list = np.concatenate((time_array_MAY, time_array_JUNE), axis=0)
#create a time array with all times
times = pd.date_range(start='2018-05-20', end='2018-06-20', freq='H')

#create u10, v10 and sst arrays with the same time dimension as the times array
u10merge = np.zeros((len(times), u10_array.shape[1], u10_array.shape[2]))
v10merge = np.zeros((len(times), v10_array.shape[1], v10_array.shape[2]))
sstmerge = np.zeros((len(times), sst_array.shape[1], sst_array.shape[2]))


#Go through the time_list and find missing times. If there are missing times, then add the previous fields to the list
#fill in with the previous fields
time_list_count = 0
for i in range(1, len(times)):
    #fill in the fields into u10merge, v10merge and sstmerge
    if times[i] == time_list[time_list_count]:
        u10merge[i] = u10_array[time_list_count]
        v10merge[i] = v10_array[time_list_count]
        sstmerge[i] = sst_array[time_list_count]
        time_list_count += 1
    elif times[i] != time_list[time_list_count] and time_list_count !=0:
        u10merge[i] = u10_array[time_list_count-1]
        v10merge[i] = v10_array[time_list_count-1]
        sstmerge[i] = sst_array[time_list_count-1]
    else:
        u10merge[i] = u10_array[time_list_count]
        v10merge[i] = v10_array[time_list_count]
        sstmerge[i] = sst_array[time_list_count]
        time_list_count += 1


print(u10_array.shape)

#save these to a pickle file
import pickle
with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\ERAV_all_2018.pickle', 'wb') as f:
    pickle.dump([lons2, lats2, timemerge, sstmerge, u10merge, v10merge], f)

#load the pickel file
#with open('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\data\\atmosphere\\ERAV_all_2018.pickle', 'rb') as f:
#    lons, lats, times, sst, u10, v10 = pickle.load(f)

#calculate the wind speed
ws = np.sqrt(u10**2 + v10**2)

#limits
levels = np.arange(0, 21, 2)

colormap = plt.cm.get_cmap('magma', 20)

import scipy.ndimage

import cfgrib

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

plt.savefig('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\wind_field_average.png')

plt.show()
#save figure

#create a gif of the wind field by looping over the time dimension
import imageio
import matplotlib.gridspec as gridspec

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
    plt.savefig('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\create_gif\\wind_field'+str(i)+'.png')
    images.append(imageio.imread('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\create_gif\\wind_field'+str(i)+'.png'))
    plt.close()




imageio.mimsave('C:\\Users\\kdo000\\Dropbox\\post_doc\\project_modelling_M2PG1_hydro\\results\\atmosphere\\create_gif\\wind_field.gif', images, duration = 0.1)







