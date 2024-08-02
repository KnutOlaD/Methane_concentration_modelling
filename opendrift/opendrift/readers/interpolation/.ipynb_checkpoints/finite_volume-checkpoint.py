# ReaderBlockUnstruct -- in-memory caching of velocity fields

import logging

logger = logging.getLogger(__name__)

from opendrift.readers.basereader.consts import *


class FiniteVolumeReaderBlock:
    """
    An unstructured ReaderBlock for Finite Volume ocean models with irregularly gridded data on an Arakawa B-grid

    Class to store and interpolate the data from a finite volume, sigma coordinate, reader.
    This uses ideas from the ReaderBlock (regular grid), but applies them in a way appropriate
    for the unstructured grid of FVCOM.

    A block essentially is just a cache with interpolation methods from
    the mesh to particles in the horizontal and vertical.

    Does __not__ include methods to extract environment profiles at the moment, will have to be
    implemented by someone who needs it :)

    Here, cells = faces
    """

    logger = logging.getLogger("opendrift")  # using common logger

    def __init__(self, data_dict, *argv, **kwarg):
        """
        Reads
        - data dict (from the readers get_variables function)
        - Block of data and particle position information

        Future?
        - Add support for other methods than "nearest" in the horizontal
        """
        super().__init__(*argv, **kwarg)
        self.data_dict = data_dict

    @property
    def time(self):
        return self.data_dict["time"]

    def interpolate(self, nodes, cells, variables=None):
        """
        Interpolate data from the finite volume block to the particles

        input:
            x, y:      particle positions
            z:         particles depth
            variables: velocities, scalar properties on mesh

        Warning:
            Profiles not yet implemented, will just be passed through
        """
        node_variables = [var for var in variables if var in self.node_variables]
        cell_variables = [var for var in variables if var in self.face_variables]
        all_variables_expected =  (len(node_variables) + len(cell_variables)) == len(variables)
        assert all_variables_expected, "missing variables requested"

        env_dict = {}
        env_dict = self.__interpolate_variables(env_dict, node_variables, nodes)
        env_dict = self.__interpolate_variables(env_dict, cell_variables, cells)
        return env_dict

    def __interpolate_variables(self, env_dict, variables, grid):
        """
        Fetch data from nearest neighbor horizontally, linear interpolation vertically
        """
        if variables:
            for var in variables:
                logger.debug("Interpolating: %s" % (var))
                if len(self.data_dict[var].shape) > 1:
                    sig_1 = self.data_dict[var][grid["sigma_ind"], grid["id"]]
                    sig_2 = self.data_dict[var][grid["sigma_next"], grid["id"]]
                    env_dict[var] = sig_1*grid['weight_sigma_ind'] + sig_2*grid['weight_sigma_next']
                else:
                    env_dict[var] = self.data_dict[var][grid["id"]]
        return env_dict
