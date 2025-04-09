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
// Controbutors (method developement): Knut Ola Dølven, Håvard Espenes, Alfred Hanssen, Muhammed Fatih Sert, Magnus Drivdal, Achim Randelhoff, Bénédicte Férré

// For date of data collection, geographic location, funding sources, and description of data: See Dølven et al. (2025) 

--------------------------
METHODOLOGICAL INFORMATION
--------------------------

A model framework for simulating dissolved gas dispersion in marine environments using a 
Lagrangian particle approach with adaptive kernel density estimation (KDE) and flexible inclusion of process modules
See Dølven et al. (2025) for a full detailed description.

--------------------
DATA & FILES OVERVIEW
--------------------

(...)

concentration_estimation/concentration_flux_modelling.py - Python file containing the main functions and script 
for concentration modeling, including process modules and a visualization section. Requries the the particle position dataset
located here: (...) to work as well as the akd_estimator.py script ready for export. 

concentration_estimation/geographic_plotter.py - Python file containing plotting functions used to visualize results 
from concentration_flux_modelling.py.

---------------------------
PUBLICATION AND HOW TO CITE
---------------------------

Dølven, K. O., Espenes, H., Hanssen, A., Sert, M. F., Drivdal, M., Randelhoff, A., Ferré, B.: Modeling water column gas transformation, migration and
atmospheric flux from seafloor seepage (...)
