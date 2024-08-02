#!/usr/bin/env python
"""
FVCOM: Using model input from unstructured grid
===============================================
"""

from datetime import timedelta
import urllib.request as urllib_request
import numpy as np
from opendrift.readers import reader_netCDF_CF_unstructured, reader_global_landmask, reader_shape
from opendrift.models.oceandrift import OceanDrift

# Single file used in the OpenDrit documentation
def original_example():
    o = OceanDrift(loglevel=20)  # Set loglevel to 0 for debug information

    proj = "EPSG:32633"
    fvcom = reader_netCDF_CF_unstructured.Reader(filename = 'https://thredds.met.no/thredds/dodsC/metusers/knutfd/thredds/netcdf_unstructured_samples/AkvaplanNiva_sample_lonlat_fixed.nc', proj4 = proj)
    o.add_reader(fvcom)
    print(fvcom)

    # Seed elements at defined positions, depth and time
    N = 1000
    z = -10*np.random.uniform(0, 1, N)
    o.seed_elements(lon=18.0, lat=69.8, radius=2000, number=N,
                    z=z, time=fvcom.start_time)

    #%%
    # Running model
    o.run(time_step=1800, duration=timedelta(hours=12))

    #%%
    # Print and plot results
    print(o)

    #%%
    # Animation (current as background not yet working).
    o.animation(color='z')

    #%%
    # .. image:: /gallery/animations/example_fvcom_0.gif

    o.plot(fast=True, buffer = 1.)


# Multiple files using a filelist as input
def filelist_example():
    '''
    Here, the coastline is made for each FVCOM grid
    '''
    o = OceanDrift(loglevel=20)  # Set loglevel to 0 for debug information

    proj = "EPSG:32633"

    # You can also use a standard reader
    coast = '/cluster/home/hes001/OpenDrift/run/NordLand/shape/NorL3_coast.shp'
    reader_coast = reader_shape.Reader.from_shpfiles(coast)
    o.add_reader(reader_coast)

    # List FVCOM files to read. Here, we use a model with 1m nodes and 2m cells
    filelist = [f'/cluster/work/users/hes001/NorLand3D/output_201806/NorL3_{n:04d}.nc' for n in range(1,4)]
    fvcom = reader_netCDF_CF_unstructured.Reader(filelist = filelist, proj4 = proj)
    o.add_reader(fvcom)

    # No beaching, no sedimentation
    o.set_config('general:use_auto_landmask', False)
    o.set_config('general:coastline_action', 'previous')
    o.set_config('general:seafloor_action', 'lift_to_seafloor')
    print(fvcom)

    # Seed elements at defined positions, depth and time
    N = 10000
    z = -10*np.random.uniform(0, 1, N)
    o.seed_elements(lon=14.2572579, lat=68.2684919, radius=50, number=N,
                    z=z, time=fvcom.start_time)

    o.run(time_step=60, duration=timedelta(days=3))
    print(o)
    o.animation(color='z')
    o.plot(fast=True, buffer = 1.)

    # Performance on a Betzy login node: 3 days simulation, 1s timestep: 4320 steps in 15 minutes