import pandas as pd
import datetime
import seawater

from fvtools.grid.tools import Filelist, num2date
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.observations.read_fvcom_hydrography import ModelCTD
from netCDF4 import Dataset

class AdFontesCTD:
    '''
    Reads excel sheets of CTD data
    - hacky, no idea if one can always expect lon, lat to be found that way -- or if we can expect "rensa" 
      to be the standard sheet name for that matter
    '''
    def __init__(self, adfontes_file):
        '''
        Class for reading CTD casts stored to excel files
        '''
        # find the keys
        d = pd.read_excel(adfontes_file, None)
        key = [key for key in d.keys() if 'rensa' in key][0]

        # Get the actual datasheet
        self.data = pd.read_excel(adfontes_file, key, skiprows = 7)

        # Find the location
        key2 = d[key].columns[0]
        self.lat = float(d[key][key2].iloc[2].split(': ')[-1])
        self.lon = float(d[key][key2].iloc[3].split(': ')[-1])

    def __repr__(self):
        return f'AdFontes CTD profile at lon = {self.lon}, lat = {self.lat}'

    def __str__(self):
        return f'{self.data}'

    @property
    def datetime(self):
        '''
        Create a datetime array
        '''
        tidspunkt = [
        datetime.datetime(dato.year, dato.month, dato.day, tid.hour, tid.minute, tid.second, tzinfo = datetime.timezone.utc) 
        for dato, tid in zip(self.data.Dato, self.data.Tid)
        ]
        return tidspunkt

    @property
    def T(self):
        '''in C'''
        return self.data[self.data.columns[4]]

    @property
    def S(self):
        '''in psu'''
        return self.data[self.data.columns[5]]

    @property
    def pressure(self):
        '''in desibar'''
        return self.data[self.data.columns[3]]

    @property
    def density(self):
        '''in kg/m3'''
        return self.data[self.data.columns[7]]+1000
    
    @property
    def calculated_density(self):
        return seawater.dens0(self.S, self.T)
    
    @property
    def depth(self):
        '''simplified depth (just divide by local density)'''
        return -self.pressure*10**4/(9.81*self.density)

class CompareToFVCOMHydrography:
    '''
    '''
    def __init__(self, CTD: AdFontesCTD, filelist: list):
        '''
        Compare AdFontes observations to FVCOM results
        '''
        self.Observation = CTD

        # Find nearest FVCOM time, use it for the comparison
        target_time = self.Observation.datetime[-1]
        self.fl = Filelist(filelist)
        closest_time = self.return_closest(fl.datetime, target_time)
        index = np.where(fl.datetime == closest_time)[0][0]

        # Get what we need to find the data in FVCOM
        self.fvcom_offset = target_time - self.fl.datetime[index]

        # Find closest FVCOM point
        self.M    = FVCOM_grid(self.fl.path[index])        
        self.Observation.x, self.Observation.y = self.M.Proj(self.Observation.lon, self.Observation.lat)
        grid = self.M.find_nearest(self.Observation.x, self.Observation.y)

        # Make model datastructure
        self.Model = ModelCTD(self.fl.path[index], self.fl.datetime[index], self.fl.index[index], grid)

    def return_closest(self, item, pivot):
        '''
        from stack overlow: https://stackoverflow.com/questions/32237862/find-the-closest-date-to-a-given-date
        '''
        return min(item, key = lambda x: abs(x - pivot))