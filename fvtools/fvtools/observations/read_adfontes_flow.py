'''
Read noise-processed AdFontes velocity data
'''
import sys
import numpy as np 
import pandas as pd
import scipy.io as sio
import utide
import datetime
import matplotlib.pyplot as plt

from functools import cached_property
from fvtools.grid.tools import date2num
from icecream import ic

# Handle things that are standard for all data
class AdFontesFlowBase:
    '''
    Used to read flow measurements from AdFontes
    - this base processes information found in all AdFontes exported files
    '''
    @property
    def instrument(self):
        return self.input['inst_name'][0]
    
    @property
    def lon(self):
        return self.input['lon']

    @property
    def lat(self):
        return self.input['lat']

    @property
    def sensor(self):
        return self.input['sensor'][0,0]

    @property
    def sensor_keys(self):
        return self.sensor.dtype.names

    @property
    def settings(self):
        return self.input['settings'][0,0]

    @property
    def settings_names(self):
        return self.settings.dtype.names

    @property
    def depth(self):
        '''
        Either reads the depth directly from the sensor, or calculates it based on recorded pressure
        '''
        if 'depth' in self.sensor_keys:
            return self.sensor['depth'].flatten()[self.valid_time]

        else:
            return ((self.pressure[self.valid_time]-self.airpressure)*10**4)/(1024*9.81)

    @property
    def airpressure(self):
        if 'airpress' in self.sensor_keys:
            return self.sensor['airpress'].flatten()
        else:
            return 0

    @property
    def pressure(self):
        return self.sensor[self._pressure_key].flatten()

    @cached_property
    def datetime(self):
        '''
        datetime with valid time indices (not flagged as nan)
        '''
        time = np.array(
            [
            [self.sensor[self.time_keys[key]][i][0] for key in self.time_keys.keys()] 
            for i in range(len(self.sensor[0]))
            ]
            )
        self.valid_time = ~np.isnan(time[:,0])
        time = time[self.valid_time]
        time[:,0] += self._year_offset
        dates = np.array([datetime.datetime(*tuple(t.astype(int))) for t in time])
        assert np.diff(dates).min().seconds > 0, 'the time vector is not monotonically increasing'
        return dates

class AdFontes2D(AdFontesFlowBase):
    '''
    Base for instruments that record data from a fixed depth
    '''
    @property
    def u(self):
        return self.input['u'].flatten()[self.valid_time]

    @property
    def v(self):
        return self.input['v'].flatten()[self.valid_time]

class AdFontes3D(AdFontesFlowBase):
    '''
    Base for instruments that record data profiled over the water column
    '''
    @property
    def u(self):
        return self.input['u'][self.valid_time, :]

    @property
    def v(self):
        return self.input['v'][self.valid_time, :]

    @property
    def ua(self):
        return np.nanmean(self.u, axis = 1)

    @property
    def va(self):
        return np.nanmean(self.v, axis = 1)

# Now, prepare one reader for each type of instrument
# ----
class AanderaaPoint(AdFontes2D):
    '''
    Reads processed data from the Aanderaa Point instrument (https://www.aanderaa.com/single-point-current-meter) 
    '''
    time_keys = {'year':'year', 'month':'month', 'day':'day', 'hour':'hour', 'minute':'min', 'second':'sec'}
    _pressure_key = 'press'
    _year_offset = 2000
    def __init__(self, data):
        '''
        Reads an AdFontes file, checks if it is an Anderaa Point instrument and initializes fields
        '''
        self.input = data
        assert self.instrument == 'Aanderaa Point', f'instrument type: "{self.instrument}"", not Aanderaa Point'
        _ = self.datetime

class DCSBlue(AdFontes2D):
    '''
    Reads processed data from the Aanderaa DCSBlue (https://www.aanderaa.com/dcs-blue-perfect-tool-measure-currents-aquaculture)
    - This instrument is specifically designed for measuring currents in aquaculture farms
    - At the time of writing (oct 11, 2023), there seems to be a bug in the processing of this instrument in AdFontes
    '''
    time_keys = {'year':'year', 'month':'day', 'day':'month', 'hour':'hour', 'minute':'min', 'second':'sec'}
    _pressure_key = 'press'
    _year_offset = 2000
    def __init__(self, data):
        '''
        Reads an AdFontes file, checks if it is an Anderaa Point instrument and initializes fields
        '''
        self.input = data
        assert self.instrument == 'DCSBlue', f'instrument type: "{self.instrument}"", is not a DCSBlue'
        _time = self.datetime

class NortekProfiler(AdFontes3D):
    '''
    Loads data from a Nortek Aquadopp profiler (https://www.nortekgroup.com/products/aquadopp-profiler-400-khz)
    - A highly versatile ADCP
    '''
    time_keys = {'year':'year', 'month':'month', 'day':'day', 'hour':'hour', 'minute':'min', 'second':'sec'}
    _pressure_key = 'press'
    _year_offset = 0
    def __init__(self, data):
        '''
        Reads an AdFontes file, checks if it is a Nortek Profiler instrument and initializes fields
        '''
        self.input = data
        assert self.instrument == 'Nortek Profiler', f'instrument type: "{self.instrument}", not Nortek Profiler'
        _ = self.datetime

    @property
    def cell_depths(self):
        return self.settings['cell_depths'].flatten()

# Gather all instruments to one list
# ----
class CurrentInstruments:
    '''
    Load multiple instruments
    '''
    def __init__(self, files):
        if type(files) != list:
            files = [files]
            
        sources = []
        for file in files:
            input = sio.loadmat(file)['file'][0][0]
            name = input['inst_name'][0]
            try:
                if name == 'Nortek Profiler':
                    sources.append(NortekProfiler(input))
                elif name == 'DCSBlue':
                    sources.append(DCSBlue(input))
                elif name == 'Aanderaa Point':
                    sources.append(AanderaaPoint(input))
                else:
                    ic(f'Can not read: {file}\nsince we do not have a reader for: {name}')
            except:
                raise ValueError(f'{file} can not be read')
        self.sources = sources

# Methods used to compare the data to FVCOM
# ----
class CompareToFVCOM:
    '''
    Class used to compare FVCOM results to field data
    '''
    def __init__(self, instruments, fvcom):
        '''
        Used to compare FVCOM results to field data
        - results can be CTD or velocity measurements
        '''
        pass

    def compare_tidespeed(self):
        '''
        plot measured speed vs speed from tidal analysis
        '''
        plt.figure()
        plt.plot(self.datetime, self.sp, label = 'Observed')
        plt.plot(self.datetime, self.sp_tide, label = 'Tidal analysis')
        plt.title(f'Speed from {self.name}')
        plt.xticks(rotation = 30)
        plt.ylabel('m/s')
        plt.legend(loc = 'upper right')
        plt.show(block = False)

# Deprecated, to be moved into other classes
# ---
class AdFontesFlow:
    '''
    Class for reading data from a .mat output-file from AdFontes
    -> Not necessarilly a need to store as .npy if this is relatively quick
    '''
    def __init__(self, adfontes_file, name):
        self.tidal_analysis()
        try:
            self.tidal_analysis_speed()
        except:
            ic('Something went wrong with the tidal analysis of speed')


    def tidal_analysis(self, const = ['M2', 'S2', 'N2', 'K1']):
        '''
        Perform a tidal analysis to find amplitude and phase of the sea surface elevation
        '''
        fit_const = ['M2', 'S2', 'N2', 'K1']# 'MSF', 'O1', 'OO1', 'M4', 'MS4', 'SK3', '3MK7', 'M6', '2SK5', 'S4', 'M8', 'M3', '2MS6', '2MK5', '2SM6']
        self.amp   = np.nan*np.zeros((len(const)))
        self.phase = np.nan*np.zeros((len(const)))
        tide = utide.solve(self.time, self.depth-np.nanmean(self.depth), lat = self.lat, verbose = False, epoch = '1858-11-17') #, constit = fit_const)
        for j, c in enumerate(const):
            ind      = np.where(tide['name'] == c)[0]
            if len(ind) > 0:
                self.amp[j]   = tide['A'][ind[0]]
                self.phase[j] = tide['g'][ind[0]]
            else:
                self.amp[j]   = np.nan
                self.phase[j] = np.nan

    def tidal_analysis_speed(self, const = ['M2', 'S2', 'N2', 'K1']):
        '''
        Perform a tidal analysis on velocity data, return reconstructed timeseries
        '''
        if len(self.u.shape)>1:
            tide = utide.solve(self.time, self.ua, self.va, lat = self.lat, verbose = False, epoch = '1858-11-17')
        else:
            tide = utide.solve(self.time, self.u, self.v, lat = self.lat, verbose = False, epoch = '1858-11-17')

        self.sp_amp   = np.nan*np.zeros((len(const)))
        self.sp_phase = np.nan*np.zeros((len(const)))

        for j, c in enumerate(const):
            ind      = np.where(tide['name'] == c)[0]
            if len(ind) > 0:
                self.sp_amp[j]   = tide['Lsmaj'][ind[0]]
                self.sp_phase[j] = tide['g'][ind[0]]

            else:
                self.sp_amp[j]   = np.nan
                self.sp_phase[j] = np.nan

        rec  = utide.reconstruct(self.time, tide, verbose = False, epoch = '1858-11-17')
        self.u_tide = rec.u
        self.v_tide = rec.v
        self.sp_tide = np.sqrt(self.u_tide**2 + self.v_tide**2)