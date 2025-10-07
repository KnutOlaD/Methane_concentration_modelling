This README file was generated on 2022-04-05 by Knut Ola Dølven.

-------------------
GENERAL INFORMATION
-------------------
// Repository for code and data accompanying Dølven et al., (2025) (see bottom for publication info)
// For the script to work, it needs to call the akd_estimator function located here: https://github.com/KnutOlaD/akd_estimator
// See concentration_flux_modelling.py for the project source path to make sure the import is working correctly
// There is a future plan for making a package for the akd_estimator, which will make this process easier... 
// Contents: 

// Contact Information
     // Name: Knut Ola Dølven	
     // Institution: UiT, The Arctic University of Tromsø	
     // Email: knut.o.dolven@uit.no
     // ORCID: 0000-0002-5315-4834

// Contributors (code): Knut Ola Dølven, Håvard Espenes (https://github.com/HvardE)
// Contributors (method developement): Knut Ola Dølven, Håvard Espenes, Alfred Hanssen, Muhammed Fatih Sert, Magnus Drivdal, Achim Randelhoff, Bénédicte Férré

// For date of data collection, geographic location, funding sources, and description of data: See Dølven et al. (2025) 

--------------------------
METHODOLOGICAL INFORMATION
--------------------------

A model framework for simulating dissolved gas dispersion in marine environments using a 
Lagrangian particle approach with adaptive kernel density estimation (KDE) and flexible inclusion of process modules
See Dølven et al. (2025) for a full detailed description.

--------------------
DATA FILES, AND FOLDERS OVERVIEW
--------------------

### Main Folders

- **Atmosphere/** - Contains atmospheric data and related processing scripts
- **concentration_estimation/** - Core implementation of concentration modeling and visualization
- **figures/** - Figures showing the effect of using different fallback values for diffusivity in OpenDrift
- **fvtools/** - Support scripts used when setting up particle tracking scenatios
- **opendrift/** - OpenDrift is a software package for modeling the trajectories and fate of objects or substances drifting in the ocean (or even in the atmosphere).
- **particle_profiles/** - Particle release profiles (derived from M2PG1 output)
- **norkyst/** - Contains a python class to access ROMS data (the NorKyst model) stored on norwegian met office servers
- **pythondefs/** - Contains a definitions file to create a singularity container used to run python
- **runscripts/** - Scripts used to run the particle tracking (opendrift) experiment explored in Dølvel et a. (2025)

### Key Scripts

#### Concentration Estimation
- `concentration_flux_modelling.py` - Main concentration modeling implementation with process modules
- `geographic_plotter.py` - Visualization functions for concentration model results
- `load_eraV_grib` - Loads eraV data, converts to .nc and interpolates onto modeling grid. 

#### Configuration
- `config.py` - Configuration and path definitions for scripts related to the concentration estimation (`concentration_flux_modelling.py` and associated scripts/functions)
- `environment.yml` - Conda environment specification

### Dependencies

This project requires the akd_estimator package, available at: https://github.com/KnutOlaD/akd_estimator

---------------------------
PUBLICATION AND HOW TO CITE
---------------------------

Dølven, K. O., Espenes, H., Hanssen, A., Sert, M. F., Drivdal, M., Randelhoff, A., Ferré, B.: Modeling water column gas transformation, migration and
atmospheric flux from seafloor seepage (...)
