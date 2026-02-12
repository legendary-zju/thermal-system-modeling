# -*- coding: utf-8
"""Module of class ElectricConnection.
"""

import numpy as np

from Aurora.components.component import Component
from Aurora.connections.connection import Connection
from Aurora.tools import fluid_properties as fp
from Aurora.tools import logger
from Aurora.tools.data_containers import DataContainer as dc
from Aurora.tools.data_containers import FluidComposition as dc_flu
from Aurora.tools.data_containers import FluidProperties as dc_prop
from Aurora.tools.data_containers import ReferencedFluidProperties as dc_ref
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
from Aurora.tools.fluid_properties import CoolPropWrapper
from Aurora.tools.fluid_properties import Q_mix_ph

from Aurora.tools.fluid_properties import T_sat_p
from Aurora.tools.fluid_properties import dh_mix_dpQ
from Aurora.tools.fluid_properties import d2h_mix_d2p_Q

from Aurora.tools.fluid_properties import T_mix_ph
from Aurora.tools.fluid_properties import dT_mix_dph
from Aurora.tools.fluid_properties import dT_mix_pdh
from Aurora.tools.fluid_properties import dT_mix_ph_dfluid
from Aurora.tools.fluid_properties import d2T_mix_d2p_h
from Aurora.tools.fluid_properties import d2T_mix_p_d2h
from Aurora.tools.fluid_properties import d2T_mix_ph_d2fluid
from Aurora.tools.fluid_properties import d2T_mix_dpdh
from Aurora.tools.fluid_properties import d2T_mix_dp_h_dfluid
from Aurora.tools.fluid_properties import d2T_mix_p_dh_dfluid
from Aurora.tools.fluid_properties import d2T_mix_ph_dfluid1_dfluid2

from Aurora.tools.fluid_properties import dT_sat_dp
from Aurora.tools.fluid_properties import d2T_sat_d2p

from Aurora.tools.fluid_properties import v_mix_ph
from Aurora.tools.fluid_properties import dv_mix_dph
from Aurora.tools.fluid_properties import dv_mix_pdh
from Aurora.tools.fluid_properties import dv_mix_ph_dfluid
from Aurora.tools.fluid_properties import d2v_mix_d2p_h
from Aurora.tools.fluid_properties import d2v_mix_p_d2h
from Aurora.tools.fluid_properties import d2v_mix_ph_d2fluid
from Aurora.tools.fluid_properties import d2v_mix_dp_dh
from Aurora.tools.fluid_properties import d2v_mix_dp_h_dfluid
from Aurora.tools.fluid_properties import d2v_mix_p_dh_dfluid
from Aurora.tools.fluid_properties import d2v_mix_ph_dfluid1_dfluid2

from Aurora.tools.fluid_properties import h_mix_pQ
from Aurora.tools.fluid_properties import h_mix_pT
from Aurora.tools.fluid_properties import h_mix_pv

from Aurora.tools.fluid_properties import p_mix_hT
from Aurora.tools.fluid_properties import p_mix_hv
from Aurora.tools.fluid_properties import p_mix_hQ

from Aurora.tools.fluid_properties import s_mix_ph
from Aurora.tools.fluid_properties import phase_mix_ph
from Aurora.tools.fluid_properties import p_critical_fluids

from Aurora.tools.fluid_properties import viscosity_mix_ph

from Aurora.tools.fluid_properties.functions import p_sat_T
from Aurora.tools.fluid_properties.helpers import get_mixture_temperature_range
from Aurora.tools.fluid_properties.helpers import get_number_of_fluids
from Aurora.tools.global_vars import ERR
from Aurora.tools.global_vars import min_derive
from Aurora.tools.global_vars import fluid_property_data as fpd
from Aurora.tools.global_vars import electromagnetic_property_data as epd
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.helpers import AURORAConnectionError
from Aurora.tools.helpers import AURORANetworkError
from Aurora.tools.helpers import convert_from_SI


class ElectricConnection(Connection):
    r"""
    Class connection is the container for fluid properties between components.

    Parameters
    ----------
    U : float, Aurora.connections.connection.Ref
        Voltage specification.

    U0 : float
        Starting value specification for voltage.

    I : float, Aurora.connections.connection.Ref
        Electricity specification.

    I0 : float
        Starting value specification for electricity.

    f : float, Aurora.connections.connection.Ref
        Frequency specification.

    f0 : float
        Starting value specification for frequency.

    design : list
        List containing design parameters (stated as string).

    offdesign : list
        List containing offdesign parameters (stated as string).

    design_path : str
        Path to individual design case for this connection.

    local_offdesign : boolean
        Treat this connection in offdesign mode in a design calculation.

    local_design : boolean
        Treat this connection in design mode in an offdesign calculation.

    printout : boolean
        Include this connection in the network's results printout.

    label : str
        Label of the connection. The default value is:
        :code:`'source:source_id_target:target_id'`.

    Note
    ----
    - The fluid balance parameter applies a balancing of the fluid vector on
      the specified conntion to 100 %. For example, you have four fluid
      components (a, b, c and d) in your vector, you set two of them
      (a and b) and want the other two (components c and d) to be a result of
      your calculation. If you set this parameter to True, the equation
      (0 = 1 - a - b - c - d) will be applied.

    - The specification of values for design and/or offdesign is used for
      automatic switch from design to offdesign calculation: All parameters
      given in 'design', e.g. :code:`design=['T', 'p']`, are unset in any
      offdesign calculation, parameters given in 'offdesign' are set for
      offdesign calculation.

    - The property state is applied on pure fluids only. If you specify the
      desired state of the fluid at a connection the convergence check will
      adjust the enthalpy values of that connection for the first
      iterations in order to meet the state requirement.

    """

    def connection_type(self):
        return 'electric'






