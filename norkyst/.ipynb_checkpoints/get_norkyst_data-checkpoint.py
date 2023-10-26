# Mal for å laste ned data fra ROMS
import roms_grid as rg
import pandas as pd
import numpy as np
from pykdtree.kdtree import KDTree
from pyproj import Proj
from netCDF4 import Dataset
from datetime import datetime, timedelta

# Min måte å laste ned data fra ROMS
M = rg.get_roms_grid('MET-NK', Proj('EPSG:32633')) # her kan du også hente data fra andre havmodeller hos met, feks NorShelf - 'NS'
M.load_grid()
lon, lat = 13.5939525, 68.8354794 # tilfeldig punkt

def get_ind(lon_grid: np.ndarray, 
            lat_grid: np.ndarray, 
            lon: float, 
            lat: float):
    grid_lonrav = lon_grid.ravel()
    grid_latrav = lat_grid.ravel()
    tree = KDTree(np.vstack([grid_lonrav, grid_latrav]).T)
    _, ind = tree.query(np.vstack([lon, lat]).T)
    bool_rav = np.zeros((grid_lonrav.shape), dtype=bool)
    bool_rav[ind[0]] = True
    bool_grid = np.reshape(bool_rav, lon_grid.shape)
    grid_ind = np.where(bool_grid)
    return grid_ind[0][0], grid_ind[1][0]

# salt, temperatur og overflatehevning lagres på rho punkter
rho_ind = get_ind(M.lon_rho, M.lat_rho, lon, lat)
u_ind = get_ind(M.lon_u, M.lat_u, lon, lat)
v_ind = get_ind(M.lon_v, M.lat_v, lon, lat)

# Sjekk om det gir mening
def check_dist(
    M,
    lon_grid: float, 
    lat_grid: float, 
    lon: float, 
    lat: float,
):
    xp, yp = M.Proj(lon, lat)
    xg, yg = M.Proj(lon_grid, lat_grid)
    print(f'Avstand til nærmeste ROMS punkt: {np.sqrt((xp-xg)**2 + (yp-yg)**2):.2f} meter')

check_dist(M, M.lon_rho[rho_ind[0], rho_ind[1]], M.lat_rho[rho_ind[0], rho_ind[1]], lon, lat)
check_dist(M, M.lon_u[u_ind[0], u_ind[1]], M.lat_u[v_ind[0], u_ind[1]], lon, lat)
check_dist(M, M.lon_u[u_ind[0], u_ind[1]], M.lat_v[v_ind[0], v_ind[1]], lon, lat)

# Finn filer som dekker ønsker tisrom
date_range = [datetime(2018,1,1) + timedelta(days=n) for n in range(6)]
files = []
for day in date_range:
    try:
        files.append(M.test_day(day))
    except Exception:
        pass

# Og nå kan man hente data fra filene som dekker simuleringen
# ---
with Dataset(files[0]) as d:
    salt = d['salt'][:, :, rho_ind[0], rho_ind[1]]
    temp = d['temp'][:, :, rho_ind[0], rho_ind[1]]
    u = d['u_eastward'][:, :, u_ind[0], u_ind[1]]
    v = d['v_northward'][:, :, v_ind[0], v_ind[1]]

# Håper det er til hjelp :)
