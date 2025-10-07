'''
Configuration file for the concentration and diffusive flux modeling projects.
adjust this file as needed in forked repositories.

Author: Knut Ola Dølven
'''

from pathlib import Path
path = Path(__file__).parent.parent.parent #

# PROJECT ROOT
PROJECT_ROOT = path

# PATHS TO ERAV DATA HANDLING (for load_eraV_grib.py)
ATMOSPHERE_PATHS = {'maydata': PROJECT_ROOT / "data" / "atmosphere" / "ERAV_May_2018.grib",
                    'junedata': PROJECT_ROOT / "data" / "atmosphere" / "ERAV_June_2018.grib",
                    'output_path': PROJECT_ROOT / "data" / "atmosphere" / "ERAV_all_2018.pickle",
                    'results_path': PROJECT_ROOT / "results" / "atmosphere"}

# PATHS FOR CONCENTRATION MODELING (for concentration_flux_modelling.py)
CONCENTRATION_MODELING_PATHS = {'interpolated_wind_sst_data': PROJECT_ROOT / "data" / "atmosphere" / "model_grid" / "interpolated_wind_sst_fields_test.pickle",
                                'PDMD': PROJECT_ROOT / "data" / "OpenDrift" / "drift_norkyst_unlimited_vdiff_30s_fb_-15.nc",
                                'bathymetry_data': PROJECT_ROOT / "data" / "bathymetry" / "IBCAO_v4_400m.nc",
                                'interpolated_bathymetry': PROJECT_ROOT / "data" / "bathymetry" / "interpolated_bathymetry.pickle",
                                'interpolated_gt_vel_data': PROJECT_ROOT / "data" / "atmosphere" / "model_grid" / "gt_vel" / "GRID_gt_vel.pickle",
                                'concentration_grid_output': PROJECT_ROOT / "data" / "data_02diff" / "GRID.pickle",
                                'output_data_path': PROJECT_ROOT / "data" / "diss_atm_flux" / "test_run",
                                'results_path': PROJECT_ROOT / "results" / "concentration_flux_modeling"}