'''
Script that loads and plots opendrift data

Author: Knut Ola Dølven

'''

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import netCDF4 as nc
import utm
from scipy.sparse import coo_matrix as coo_matrix
from scipy.sparse import csr_matrix as csr_matrix
import pickle
import numba
#add path to load_plot_opendrift
import sys
sys.path.append(r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\src\Dispersion_modelling\concentration_estimation')
from load_plot_opendrift import *
import utm 

datapath = r'C:\Users\kdo000\Dropbox\post_doc\project_modelling_M2PG1_hydro\data\OpenDrift\drift_norkyst.nc'

ODdata = nc.Dataset(datapath)

number = 7*24

#load only final timestep
particles = {'lon':ODdata.variables['lon'][:,number],
                    'lat':ODdata.variables['lat'][:,number],
                    'z':ODdata.variables['z'][:,number],
                    'time':ODdata.variables['time'][number],
                    'status':ODdata.variables['status'][:,number]} #this is 

#add utm coordinates to the dictionary
#get only the particles that are active (they are nonmasked)
particles['lon'] = particles['lon'][~particles['lon'].mask]
particles['lat'] = particles['lat'][~particles['lat'].mask]
particles['z'] = particles['z'][~particles['z'].mask]
particles['status'] = particles['status'][~particles['status'].mask]

print(np.shape(particles['lon']))


UTM = utm.from_latlon(particles['lat'],particles['lon'])
particles['UTM_x'] = UTM[0]
particles['UTM_y'] = UTM[1]

#plot only particles where particles['z'] is larger than -10
plt.figure()
plt.scatter(particles['UTM_x'][particles['z']>-10],particles['UTM_y'][particles['z']>-10],s=1)



#make a scatterplot with very small, filled and transparent (alpha=0.01) points

#plt.scatter(particles['lon'],particles['lat'],s=0.1)


#plot the data as a heatmap of the number of particles in each grid cell
#find the grid cell size
