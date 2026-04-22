# -*- coding: utf-8

"""Module for free fluid property engine.
Aurora/tools/fluid_properties/free_fluid_engine.py

SPDX-License-Identifier: MIT
"""

from Aurora.tools.fluid_properties.properties_reference_data import MOLTEN_SALT_REFERENCE_DATA as salt_data
from Aurora.tools.fluid_properties.properties_reference_data import molar_mass_dict
from Aurora.tools.fluid_properties.properties_reference_data import atomic_masses
from Aurora.tools.fluid_properties.properties_reference_data import compound_groups_masses
from Aurora.tools.fluid_properties.properties_reference_data import Customized_Fluid


class FreeFluidEngine:
    """
    Supply properties for fluid composition defined by user.
    """
    def __init__(self, fluid):
        self.fluid = fluid
        self.__generate_fluid_properties_data__()

    def __generate_fluid_properties_data__(self):
        self.T_melt = None