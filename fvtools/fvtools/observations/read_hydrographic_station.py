import numpy as np
import pandas as pd
import datetime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from fvtools.grid.tools import Filelist
from fvtools.grid.fvcom_grd import FVCOM_grid
from fvtools.observations.read_fvcom_hydrography import ModelCTD
from random import sample
import matplotlib._color_data as mc

class HydrographicStationPositions:
    '''
    positions of hydrographic stations
    '''
    @property
    def YtreUtsira(self):
        return np.array([4.8, 59.3167])

    @property
    def IndreUtsira(self):
        return np.array([4.9833, 59.3167])

    @property
    def Sognesjoen(self):
        return np.array([4.765, 60.98])

    @property
    def Skrova(self):
        return np.array([14.65, 68.1167])

    @property
    def Eggum(self):
        return np.array([13.6333, 68.3667])
        
    @property
    def Balsfjord(self):
        return np.array([np.nan, np.nan])

    @property
    def Ingoy(self):
        return np.array([24.0167, 71.1333])


class HydrographicStationNC:
    '''
    There are 8 permanent hydorgaphic stations in Norway:  http://www.imr.no/forskning/forskningsdata/stasjoner/index.html
    Data from these stations can be downloaded from NMDC:  http://metadata.nmdc.no/UserInterface/#/
    '''
    def __init__(self, ncfile: str):
        '''
        Stores data from ncfile to a class that communicates well with FVCOM
        '''
        with Dataset(ncfile, 'r') as d:
            self.time = d['TIME'][:].data
            self.time_units = d['TIME'].units.replace('T', ' ').replace('Z', '')
            self.lon = d['LONGITUDE'][0].data
            self.lat = d['LATITUTE'][0].data
            self.depth = d['DEPTH'][:].data
            self.temp = d['TEMP'][:].data
            self.salinity = d['PSL'][:].data

class HydrographicStationTXT:
    '''
    There are 8 permanent hydorgaphic stations in Norway:  
        - http://www.imr.no/forskning/forskningsdata/stasjoner/index.html

    From these, you can download single text-files for each depth, covering a certain timespan:
        - http://www.imr.no/forskning/forskningsdata/stasjoner/view/initdownload
        - Choose a timespan, station and depth.

    This routine will probably not work for stations with a space in their name ('Ytre Utsira, 'Indre Utsira'),
    since I had to hack my way through reading the text files...
    '''
    def __init__(self, textfiles: list, lon: float, lat: float):
        '''
        Stores data from singular textfiles together, and is designed to communicate well with the FVCOM validation toolbox
        '''
        # There must be a better way (?), but this works
        self.lon, self.lat = lon, lat
        self.data = pd.DataFrame()
        for file in textfiles:
            d = pd.read_csv(file)
            data = d.to_numpy()
            values = []
            for row in data:
                tmp = row[0].split(' ')
                tmp = [value for value in tmp if value != '']
                values.append(tmp)
            df = pd.DataFrame(values, columns=['Stasjon', 'Dato', 'Tid', 'Dyp', 
                                               'Temperatur', 'Salt', 
                                               'Kalibrert temperatur', 'Kalibrert salt'])
            self.data = pd.concat([self.data, df])

        # Now we re-format some of the values from string to float
        keys = ['Dyp', 'Temperatur', 'Salt', 'Kalibrert temperatur', 'Kalibrert salt']
        for key in keys:
            self.data[key] = self.data[key].astype(float)

        # And we group by dates
        self.groups = self.data.groupby('Dato')

    @property
    def group_names(self):
        names = [str(name) for name in self.groups.groups.keys()]
        return names

    @property
    def columns(self):
        return self.data.columns

    @property
    def datetime(self):
        '''
        time for each group as a datetime object referenced in the utc timeformat
        - we use the "centre time" in the CTD cast
        '''
        datetimes =  []
        for group in self.group_names:
            date      = [int(num) for num in reversed(group.split('.'))]
            tid       = self.get_value(group, 'Tid')
            timetuple = [[int(f) for f in t.split(':')] for t in tid]
            date_time = [datetime.datetime(*(date + t), tzinfo=datetime.timezone.utc) for t in timetuple]
            datetimes.append(date_time[int(len(date_time)/2)])
        datetimes.sort()
        return datetimes

    def get_value(self, group: str, key: str, sort = 'Dyp'):
        '''
        Returns the variable "key" in the "group"
        '''
        return self.get_group(group, sort = sort)[key].to_numpy()

    def get_group(self, group: str, sort = 'Dyp'):
        '''
        Returns a group sorted by "sort"
        '''
        group = self.groups.get_group(group)
        group = group.sort_values(sort)
        return group

    def plot(self, group_names):
        '''
        Plot all observations
        '''
        fig, ax = plt.subplots(1, 2, figsize=[12,6], sharey = True)
        for group in group_names:
            ax[0].plot(self.get_value(group, 'Salt'), -self.get_value(group, 'Dyp'), label = group)
            ax[1].plot(self.get_value(group, 'Temperatur'), -self.get_value(group, 'Dyp'), label = group)
        ax[1].set_xlabel('Salinity [psu]')
        ax[0].set_xlabel(r'Temperatur [$^{\degree}$C]')
        
        ax[0].set_ylabel('Depth below sea surface [m]')
        ax[1].legend()
        plt.suptitle('CTD casts at Skrova hydrological station')

    def make_hovmoller(self):
        '''
        Make a hovmoller diagram over the data period
        '''
        # For later, if needbe
        pass

class CompareIMRHydrographyToFVCOM:
    '''
    Methods used to compare CTD casts from FVCOM to 
    '''
    def __init__(self, IMR: HydrographicStationTXT, filelist: list):
        '''
        Loads a HydrographicStationTXT object, finds equivalent CTD casts in FVCOM and
        stores to self
        - construct a method that removes data far from our period of interest
        '''
        self.Observation = IMR
        self.fl = Filelist(filelist)

        # Find matching dates
        closest_date = []
        for date in self.Observation.datetime:
            closest_date.append(self.return_closest(self.fl.datetime, date))

        index = []
        for date in closest_date:
            index.append(np.where(self.fl.datetime == date)[0][0])

        # Find closest node
        self.M = FVCOM_grid(self.fl.path[index[0]])
        self.Observation.x, self.Observation.y = self.M.Proj(self.Observation.lon, self.Observation.lat)
        grid = self.M.find_nearest(self.Observation.x, self.Observation.y)

        # Get model data
        self.Model = []
        for timeind in index:
            self.Model.append(
                ModelCTD(
                    self.fl.path[timeind], 
                    self.fl.datetime[timeind], 
                    self.fl.index[timeind], 
                    grid
                    )
                )

        # Make it very clear and obvious how close we are
        print('Absolute time difference between model and observation')
        for i in range(len(self.Observation.datetime)):
            print(f'{abs(self.Model[i].datetime - self.Observation.datetime[i])}')

    def return_closest(self, item, pivot):
        '''
        from stack overlow: https://stackoverflow.com/questions/32237862/find-the-closest-date-to-a-given-date
        '''
        return min(item, key = lambda x: abs(x - pivot))

    def spagetti(
        self, 
        extent: tuple, 
        georef_extent: tuple = [0.75, 0.6, 0.3, 0.3],
        url: str = 'https://openwms.statkart.no/skwms1/wms.topo4.graatone?service=wms&request=getcapabFilities', 
        layers: tuple = ['topo4graatone_WMS'],
        group_names: list = None,
        ):
        '''
        Compare CTD and model output
        '''
        fig, ax = plt.subplots(1, 2, figsize = [12,6], sharey = True)

        if group_names is not None:
            model = [model for model, name in zip(self.Model, self.Observation.group_names) if name in group_names]
            iterable = zip(group_names, model)
        else:
            iterable = zip(self.Observation.group_names, self.Model)
        
        for i, (group, Model) in enumerate(iterable):
            line, = ax[0].plot(self.Observation.get_value(group, 'Salt'), -self.Observation.get_value(group, 'Dyp'), label = f'IMR {group}')
            ax[1].plot(self.Observation.get_value(group, 'Temperatur'), -self.Observation.get_value(group, 'Dyp'), color = line.get_color(), label = f'IMR {group}')
            ax[0].plot(Model.S, Model.depth, '--', color = line.get_color(), label = f'FVCOM {group}')
            ax[1].plot(Model.T, Model.depth, '--', color = line.get_color(), label = f'FVCOM {group}')

        ax[0].set_ylabel('Depth below sea surface [m]')
        ax[0].set_xlabel(r'Temperature [$^{\degree}$C]')
        ax[1].set_xlabel('Salinity [psu]')
        ax[0].legend()
        plt.suptitle('CTD casts vs FVCOM')

        if type(georef_extent) == list:
            ax2 = fig.add_axes(georef_extent, projection = ccrs.epsg(int(self.M.reference.split(':')[-1])))
            ax2.add_wms(url, layers = layers)
            if 'raster' not in url:
                ax2.add_wms('https://wms.geonorge.no/skwms1/wms.dybdedata2?service=WMS&request=GetCapabilities', layers = ['Dybdedata2'])
            ax2.scatter(self.Observation.x, self.Observation.y, 200, c='r', marker='*', edgecolor='k', label = 'Observation')
            ax2.scatter(self.Model[0].x, self.Model[0].y, 100, c='k', marker='X', label = 'FVCOM')
            ax2.set_xlim([self.Observation.x - extent, self.Observation.x + extent])
            ax2.set_ylim([self.Observation.y - extent, self.Observation.y + extent])
            ax2.legend()


    def georeference(self, extent):
        '''
        Compare measurement to model
        '''
        import cartopy.crs as ccrs
        M = FVCOM_grid(self.fl.path[0])
        x, y = M.Proj(self.Observation.lon, self.Observation.lat)
        grid = M.find_nearest(x, y)

        fig = plt.figure()
        ax = plt.axes(projection = ccrs.epsg(int(M.reference.split(':')[-1])))
        ax.add_wms('https://openwms.statkart.no/skwms1/wms.topo4.graatone?service=wms&request=getcapabilities', layers = ['topo4graatone_WMS'])
        ax.add_wms('https://wms.geonorge.no/skwms1/wms.dybdedata2?service=WMS&request=GetCapabilities', layers = ['Dybdedata2'])
        ax.scatter(self.Observation.x, self.Observation.y, 100, c = 'r', marker = '*', edgecolor = 'k', label = 'Observation')
        ax.scatter(M.x[self.Model[0].grid], M.y[self.Model[0].grid], c = 'k', marker = 'X', edgecolor = 'k', label = 'Model')
        ax.set_xlim([compare.Observation.x-extent, compare.Observation.x+extent])
        ax.set_ylim([compare.Observation.y-extent, compare.Observation.y+extent])
        ax.legend()