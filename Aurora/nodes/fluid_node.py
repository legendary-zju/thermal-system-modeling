import numpy as np
from Aurora.nodes.node import Node
from Aurora.tools import fluid_properties as fp
from Aurora.tools import logger
from Aurora.tools.data_containers import DataContainer as dc
from Aurora.tools.data_containers import FluidComposition as dc_flu
from Aurora.tools.data_containers import FluidProperties as dc_prop
from Aurora.tools.data_containers import ReferencedFluidProperties as dc_ref
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
from Aurora.tools.data_containers import ComponentProperties as dc_cp
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
from Aurora.tools.fluid_properties import p_crit_

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


class FluidNode(Node):
    """Define fluid node for fluid components"""
    def __init__(self, component, comp_type, **kwargs):
        super().__init__(component, comp_type, **kwargs)
        self.fluid_reference = None

    def set_properties(self):
        return {
            "m": dc_prop(
                is_result=False,
                property_data=fpd['m'],
                SI_unit=fpd['m']['SI_unit'],
                scale=ps['m']['scale']),
            "p": dc_prop(
                is_result=True,
                property_data=fpd['p'],
                SI_unit=fpd['p']['SI_unit'],
                scale=ps['p']['scale']),
            "h": dc_prop(
                is_result=True,
                property_data=fpd['h'],
                SI_unit=fpd['h']['SI_unit'],
                scale=ps['h']['scale']),
            "T": dc_prop(
                is_result=True,
                val_SI=300,
                property_data=fpd['T'],
                SI_unit=fpd['T']['SI_unit'],
                scale=ps['T']['scale']),
            "x": dc_prop(
                is_result=True,
                val_SI=0,
                property_data=fpd['x'],
                SI_unit=fpd['x']['SI_unit'],
                scale=ps['x']['scale']),
            "fluid": dc_flu(
                scale=ps['fluid']['scale']),
            'n': dc_cp(
                is_result=True,
            )

        }

    def calc_T(self):
        """Calculate temperature"""
        try:
            self.T.val_SI = T_mix_ph(self.p.val_SI, self.h.val_SI,
                                     self.fluid_reference.fluid_data,
                                     self.fluid_reference.mixing_rule,
                                     T0=self.fluid_reference.T.val_SI)  #
        except ValueError as e:
            self.calc_temperature_bounds_()
            min_enthalpy = self.calc_min_enthalpy_()
            max_enthalpy = self.calc_max_enthalpy_()
            if self.h.val_SI < min_enthalpy:
                # msg = f'node enthalpy: {self.h.val_SI} < min_enthalpy: {min_enthalpy}--{e}'
                # logger.error(msg)
                min_enthalpy += 1
                self.T.val_SI = (T_mix_ph(self.p.val_SI, min_enthalpy,
                                         self.fluid_reference.fluid_data,
                                         self.fluid_reference.mixing_rule,
                                         T0=self.fluid_reference.T.val_SI) +
                                 (self.h.val_SI - min_enthalpy) * dT_mix_pdh(self.p.val_SI, min_enthalpy,
                                                                             self.fluid_reference.fluid_data,
                                                                             self.fluid_reference.mixing_rule,))
            elif self.h.val_SI > max_enthalpy:
                # msg = f'node enthalpy: {self.h.val_SI} > max_enthalpy: {max_enthalpy}--{e}'
                # logger.error(msg)
                max_enthalpy -= 1
                self.T.val_SI = (T_mix_ph(self.p.val_SI, max_enthalpy,
                                          self.fluid_reference.fluid_data,
                                          self.fluid_reference.mixing_rule,
                                          T0=self.fluid_reference.T.val_SI) +
                                 (self.h.val_SI - max_enthalpy) * dT_mix_pdh(self.p.val_SI, max_enthalpy,
                                                                             self.fluid_reference.fluid_data,
                                                                             self.fluid_reference.mixing_rule, ))

    def calc_min_enthalpy_(self):
        """Calculate minimum enthalpy"""
        min_enthalpy = h_mix_pT(self.p.val_SI,
                                self.T.min_val,
                                self.fluid_reference.fluid_data,
                                self.fluid_reference.mixing_rule)
        return min_enthalpy

    def calc_max_enthalpy_(self):
        """Calculate maximum enthalpy"""
        max_enthalpy = h_mix_pT(self.p.val_SI,
                                self.T.max_val,
                                self.fluid_reference.fluid_data,
                                self.fluid_reference.mixing_rule)
        return max_enthalpy

    def calc_temperature_bounds_(self):
        """Calculate minimum temperature"""
        Tmin = max(
            [w._T_min for f, w in self.fluid.wrapper.items() if self.fluid.val[f] > ERR]
        ) * 1.01
        Tmax = min(
            [w._T_max for f, w in self.fluid.wrapper.items() if self.fluid.val[f] > ERR]
        ) * 0.99
        self.T.min_val, self.T.max_val = Tmin, Tmax

    def calc_x(self):
        if len(self.fluid.val.items()) > 1:
            self.x.val_SI = np.nan
        else:
            if self.p.val_SI > p_crit_(self.fluid_reference.fluid_data):
                self.x.val_SI = np.nan
            h_0 = h_mix_pQ(self.p.val_SI, 0, self.fluid_reference.fluid_data)
            h_1 = h_mix_pQ(self.p.val_SI, 1, self.fluid_reference.fluid_data)
            if self.h.val_SI >= h_0 - ERR and self.h.val_SI <= h_1 + ERR:
                self.x.val_SI = Q_mix_ph(self.p.val_SI, self.h.val_SI, self.fluid_reference.fluid_data)
            else:
                self.x.val_SI = np.nan

    def calc_properties(self):
        """Calculate fluid properties"""
        self.calc_T()
        self.calc_x()
