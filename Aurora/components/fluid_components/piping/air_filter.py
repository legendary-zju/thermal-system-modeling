# -*- coding: utf-8

"""Module for class AirFilter.
"""

import numpy as np

from Aurora.components.component import component_registry
from Aurora.components.fluid_components.piping.valve import Valve
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.fluid_properties import h_mix_pQ
from Aurora.tools import helpers as hlp
from Aurora.tools import logger


@component_registry
class AirFilter(Valve):
    pass

