'''
Load and plot opendrift data

Author: Knut Ola Dølven

'''
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import netCDF4 as nc
import utm
from scipy.sparse import coo_matrix as coo_matrix

#List of variables in the script:
#datapath: path to the netcdf file containing the opendrift data
#ODdata: netcdf file containing the opendrift data
#particles: dictionary containing information about the drift particles
#time: time vector
#x: x-position of the particles
#y: y-position of the particles

#Load data
datapath = '/home/knut/Documents/masteroppgave/analysis/od_data/2016-04-01_2016-04-30.nc'

ODdata = nc.Dataset(datapath)

