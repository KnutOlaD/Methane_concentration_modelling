import sys
#sys.path.append('/cluster/home/hes001/opendrift/') 
# clone opendrift here: https://github.com/OpenDrift/opendrift
import opendrift
import matplotlib.pyplot as plt
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.models.oceandrift import OceanDrift
from datetime import datetime, timedelta
import os
import pandas as pd

def run(hours_pr_release = 1):
    '''
    Release NorKyst particles
    - hours_pr_release : how often to release particles
    '''
    o = OceanDrift(loglevel=20)
    e = pd.read_excel('fluxes_knut_ola.xlsx')
    reader_norkyst = reader_netCDF_CF_generic.Reader('https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be')
    o.add_reader(reader_norkyst)
    positions = e.Longitude, e.Latitude
    zlevels   = np.arange(-200, 0, 10)
    
    start = datetime(2020, 1, 1)
    stop  = datetime(2020, 2, 1)
    timesteps = int((stop-start)/timedelta(hours=hours_pr_release))
    start_times = [start + timedelta(hours=hours_pr_release*n) for n in range(timesteps)]

    # Initialize particles
    for t in start_times:
        for z in zlevels:
            o.seed_elements(lon=e['Longitude'].to_numpy(), lat=e['Latitude'].to_numpy(), time=t, z=z, wind_drift_factor=0)

    o.run(
        time_step=600,
        duration=timedelta(days=(stop-start).days), 
        time_step_output=3600,
        outfile=f'drift_test.nc',
        export_buffer_length=12,
        export_variables=['time', 'lon', 'lat', 'z'],
    )