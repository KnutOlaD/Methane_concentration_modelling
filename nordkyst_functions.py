'''
Functions used to load and process data from the Nordkyst-800 model

Author: Knut Ola Dølven

Includes the following functions:

load_nordkyst_data(year,month,day)
    Loads the Nordkyst-800 data for the specified date
    Input: year, month, day: The year, month and day of the data to be loaded
    Output: nc: The netcdf file containing the Nordkyst-800 data



'''

from datetime import datetime
import numpy as np
import netCDF4

#Find path to the netcdf file containing the Nordkyst-800 data
date    = datetime(2018, 5, 20)
https   = 'https://thredds.met.no/thredds/dodsC/fou-hi/new_norkyst800m/his/ocean_his.an.'
year    = str(date.year)
month = '{:02d}'.format(date.month)
day     = '{:02d}'.format(date.day)
netcdf_file = f'{https}{year}{month}{day}.nc'

#Load the netcdf file using the netCDF4 module
nc = netCDF4.Dataset(netcdf_file)

#Find the variable names in the netcdf file
print(nc.variables.keys())

file_keys = nc.variables.keys()