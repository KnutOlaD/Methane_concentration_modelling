# Model hydrography reader
from netCDF4 import Dataset
from fvtools.grid.tools import num2date

class ModelCTD:
    def __init__(self, file, date, time_index, grid_index):
        self.path = file
        self.grid = grid_index
        with Dataset(self.path, 'r') as d:
            self.S = d['salinity'][time_index, :, self.grid]
            self.T = d['temp'][time_index, :, self.grid]
            self.zeta = d['zeta'][time_index, self.grid]
            self.h = d['h'][self.grid]
            self.siglay = d['siglay'][:, self.grid]
            self.time = d['time'][time_index]
            self.time_units = d['time'].units
            self.x = d['x'][self.grid]
            self.y = d['y'][self.grid]

    @property
    def depth(self):
        return self.siglay*(self.h+self.zeta)

    @property
    def datetime(self):
        return num2date(self.time, units = self.time_units)
    
    @property
    def density(self):
        return seawater.dens0(self.S, self.T)