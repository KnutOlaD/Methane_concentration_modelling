# Tools used to retrieve observations, and compare them to FVCOM
# ----

For example;
We want to compare model data to the permanent hydrographic station near Skrova:
```python
from fvtools.observations.read_hydrographic_station import HydrographicStationTXT, CompareIMRHydrographyToFVCOM
import os
base = '/cluster/home/hes001/stasjonsdata_skrova/'
textfiles = [f'{base}{file}' for file in os.listdir(base)]
H = HydrographicStationTXT(textfiles, 14.65, 68.1167) # location of the Skrova station
filelist = '/cluster/work/users/hes001/NorLand3D/filelist_full_2023.txt' # Results from the 2023 version of the Nordland model

# The comare object automatically finds the closest FVCOM result
compare = CompareIMRHydrographyToFVCOM(H, filelist)

# We can thereafter plot a comparison
compare.spagetti(
    25000, 
    georef_extent = [0.75, 0.7, 0.3, 0.3],
    url = 'http://openwms.statkart.no/skwms1/wms.toporaster4?version=1.3.0&service=wms&request=getcapabilities',
    layers = ['toporaster'],
)
plt.suptitle('CTD casts vs FVCOM')
```


