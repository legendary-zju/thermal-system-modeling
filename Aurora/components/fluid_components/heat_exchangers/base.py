# -*- coding: utf-8

"""Module of class HeatExchanger.
"""
import math

import numpy as np

from Aurora.components.component import Component
from Aurora.components.fluid_components.fluid_component import FluidComponent
from Aurora.components.component import component_registry
from Aurora.nodes.node import Node
from Aurora.tools import logger
from Aurora.tools.characteristics import CharLine
from Aurora.tools.data_containers import ComponentCharacteristics as dc_cc
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import GroupedComponentCharacteristics as dc_gcc
from Aurora.tools.data_containers import FitCoefficient as dc_fit
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import fluid_property_data as fpd
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.document_models import generate_latex_eq

from Aurora.tools.fluid_properties import T_mix_ph, p_mix_hT, p_sat_T
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

from Aurora.tools.fluid_properties import h_mix_pT
from Aurora.tools.fluid_properties import h_mix_pQ
from Aurora.tools.fluid_properties import T_sat_p
from Aurora.tools.fluid_properties import s_mix_ph
from Aurora.tools.global_vars import ERR
from Aurora.tools.global_vars import min_derive
from Aurora.tools.helpers import AURORANetworkError


@component_registry
class HeatExchanger(FluidComponent):
    r"""
    Class for counter current heat exchanger.

    The component HeatExchanger is the parent class for the components:

    - :py:class:`AURORA.components.heat_exchangers.condenser.Condenser`
    - :py:class:`AURORA.components.heat_exchangers.desuperheater.Desuperheater`

    **Mandatory Equations**

    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.energy_balance_func`

    **Optional Equations**

    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.energy_balance_hot_func`
    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.kA_func`
    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.kA_char_func`
    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.DTU_func`
    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.DTL_func`
    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.eff_cold_func`
    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.eff_hot_func`
    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.eff_max_func`
    - hot side :py:meth:`AURORA.components.component.Component.pr_func`
    - cold side :py:meth:`AURORA.components.component.Component.pr_func`
    - hot side :py:meth:`AURORA.components.component.Component.zeta_func`
    - cold side :py:meth:`AURORA.components.component.Component.zeta_func`

    Inlets/Outlets

    - in1, in2 (index 1: hot side, index 2: cold side)
    - out1, out2 (index 1: hot side, index 2: cold side)

    Parameters
    ----------
    label : str
        The label of the component.

    design : list
        List containing design parameters (stated as String).

    offdesign : list
        List containing offdesign parameters (stated as String).

    design_path : str
        Path to the components design case.

    local_offdesign : boolean
        Treat this component in offdesign mode in a design calculation.

    local_design : boolean
        Treat this component in design mode in an offdesign calculation.

    char_warnings : boolean
        Ignore warnings on default characteristics usage for this component.

    printout : boolean
        Include this component in the network's results printout.

    Q : float, dict
        Heat transfer, :math:`Q/\text{W}`.

    pr1 : float, dict, :code:`"var"`
        Outlet to inlet pressure ratio at hot side, :math:`pr/1`.

    pr2 : float, dict, :code:`"var"`
        Outlet to inlet pressure ratio at cold side, :math:`pr/1`.

    dp1 : float, dict, :code:`"var"`
        Inlet to outlet pressure delta at hot side, unit is the network's
        pressure unit!.

    dp2 : float, dict, :code:`"var"`
        Inlet to outlet pressure delta at cold side, unit is the network's
        pressure unit!.

    zeta1 : float, dict, :code:`"var"`
        Geometry independent friction coefficient at hot side,
        :math:`\frac{\zeta}{D^4}/\frac{1}{\text{m}^4}`.

    zeta2 : float, dict, :code:`"var"`
        Geometry independent friction coefficient at cold side,
        :math:`\frac{\zeta}{D^4}/\frac{1}{\text{m}^4}`.

    DTL : float, dict
        Lower terminal temperature difference :math:`DT_\mathrm{L}/\text{K}`.

    DTU : float, dict
        Upper terminal temperature difference :math:`DT_\mathrm{U}/\text{K}`.

    DTN_min : float, dict
        Minimum terminal temperature difference :math:`DTN_\mathrm{min}/\text{K}`.

    DTM: float, dict
        Pinch temperature difference :math:`DT_\mathrm{U}/\text{K}`

    eff_cold : float, dict
        Cold side heat exchanger effectiveness :math:`eff_\text{cold}/\text{1}`.

    eff_hot : float, dict
        Hot side heat exchanger effectiveness :math:`eff_\text{hot}/\text{1}`.

    eff_max : float, dict
        Max value of hot and cold side heat exchanger effectiveness values
        :math:`eff_\text{max}/\text{1}`.

    kA : float, dict
        Area independent heat transfer coefficient,
        :math:`kA/\frac{\text{W}}{\text{K}}`.

    KDTA : float, dict
        Area independent heat transfer coefficient,
        :math:`KDTA/\frac{\text{W}}{\text{K}}`.

    kA_char : dict
        Area independent heat transfer coefficient characteristic.

    kA_char1 : AURORA.tools.characteristics.CharLine, dict
        Characteristic line for hot side heat transfer coefficient.

    kA_char2 : AURORA.tools.characteristics.CharLine, dict
        Characteristic line for cold side heat transfer coefficient.

    """
    @classmethod
    def correct_nodes_num(cls, num):
        if num < 8:
            num = 8
        return super().correct_nodes_num(num)

    @staticmethod
    def is_differential_component():
        return True

    def generate_nodes_constructure(self):
        node_properties = []
        node_temp = Node(self, 'fluid')
        for key in node_temp.properties.keys():
            data = node_temp.get_attr(key)
            if hasattr(data, 'is_result') and data.is_result:
                node_properties.append(key)
        for side in range(max(len(self.inlets()), len(self.outlets()))):
            side_names = ['hot_side', 'cold_side']
            self.nodes_properties[side] = {'side_name': side_names[side], 'node_properties': node_properties}
            for node_col in range(self.nodes_num):
                self.nodes.loc[side, node_col] = Node(self, 'fluid')

    def simplify_nodes_topology(self):
        for row in self.nodes.index:
            for col in range(self.nodes_num):
                node = self.nodes.loc[row, col]
                node.n.min_val = self.nodes_num
                node.m = self.inl[row].m
                node.fluid = self.inl[row].fluid
                node.fluid_reference = self.inl[row]

    @staticmethod
    def component():
        return 'heat exchanger'

    @staticmethod
    def inlets():
        return ['in1', 'in2']

    @staticmethod
    def outlets():
        return ['out1', 'out2']

    def spread_forward_pressure_values(self, inconn):
        conn_idx = self.inl.index(inconn)  # the index of branch
        outconn = self.outl[conn_idx]  # outconn: object of outlet connection.
        if conn_idx == 0:  # hot liquid side
            if inconn.p.is_set and not outconn.p.is_set and ((self.pr1.is_set and self.pr1_fit.rule in ['constant', 'static'])
                                                             or (self.dp1.is_set and self.dp1_fit.rule in ['constant', 'static'])):
                if self.pr1.is_set:
                    outconn.p.val_SI = inconn.p.val_SI * self.pr1.val_SI
                    outconn.p.is_set = True
                    outconn.p.is_var = False
                    self.pr1.is_set = False
                elif self.dp1.is_set:
                    outconn.p.val_SI = inconn.p.val_SI - self.dp1.val_SI
                    outconn.p.is_set = True
                    outconn.p.is_var = False
                    self.dp1.is_set = False
                if outconn not in self.network.connections_spread_pressure_container:
                    self.network.connections_spread_pressure_container.append(outconn)
                    outconn.target.spread_forward_pressure_values(outconn)
                    outconn.spread_pressure_reference_check()
            return
        else:  # cold liquid side
            if inconn.p.is_set and not outconn.p.is_set and ((self.pr2.is_set and self.pr2_fit.rule in ['constant', 'static'])
                                                             or (self.dp2.is_set and self.dp2_fit.rule in ['constant', 'static'])):
                if self.pr2.is_set:
                    outconn.p.val_SI = inconn.p.val_SI * self.pr2.val_SI
                    outconn.p.is_set = True
                    outconn.p.is_var = False
                    self.pr2.is_set = False
                elif self.dp2.is_set:
                    outconn.p.val_SI = inconn.p.val_SI - self.dp2.val_SI
                    outconn.p.is_set = True
                    outconn.p.is_var = False
                    self.dp2.is_set = False
                if outconn not in self.network.connections_spread_pressure_container:
                    self.network.connections_spread_pressure_container.append(outconn)
                    outconn.target.spread_forward_pressure_values(outconn)
                    outconn.spread_pressure_reference_check()
            return

    def spread_backward_pressure_values(self, outconn):
        conn_idx = self.outl.index(outconn)
        inconn = self.inl[conn_idx]
        if conn_idx == 0:  # hot liquid side
            if not inconn.p.is_set and outconn.p.is_set and ((self.pr1.is_set and self.pr1_fit.rule in ['constant', 'static'])
                                                             or (self.dp1.is_set and self.dp1_fit.rule in ['constant', 'static'])):
                if self.pr1.is_set:
                    inconn.p.val_SI = outconn.p.val_SI / self.pr1.val_SI
                    inconn.p.is_set = True
                    inconn.p.is_var = False
                    self.pr1.is_set = False
                elif self.dp1.is_set:
                    inconn.p.val_SI = outconn.p.val_SI + self.dp1.val_SI
                    inconn.p.is_set = True
                    inconn.p.is_var = False
                    self.dp1.is_set = False
                if inconn not in self.network.connections_spread_pressure_container:
                    self.network.connections_spread_pressure_container.append(inconn)
                    inconn.source.spread_backward_pressure_values(inconn)
                    inconn.spread_pressure_reference_check()
            return
        else:  # cold liquid side
            if not inconn.p.is_set and outconn.p.is_set and ((self.pr2.is_set and self.pr2_fit.rule in ['constant', 'static'])
                                                             or (self.dp2.is_set and self.dp2_fit.rule in ['constant', 'static'])):
                if self.pr2.is_set:
                    inconn.p.val_SI = outconn.p.val_SI / self.pr2.val_SI
                    inconn.p.is_set = True
                    inconn.p.is_var = False
                    self.pr2.is_set = False
                elif self.dp2.is_set:
                    inconn.p.val_SI = outconn.p.val_SI + self.dp2.val_SI
                    inconn.p.is_set = True
                    inconn.p.is_var = False
                    self.dp2.is_set = False
                if inconn not in self.network.connections_spread_pressure_container:
                    self.network.connections_spread_pressure_container.append(inconn)
                    inconn.source.spread_backward_pressure_values(inconn)
                    inconn.spread_pressure_reference_check()
            return

    def set_pressure_initial_factor(self, branch_index=0):
        inconn = self.inl[branch_index]
        outconn = self.outl[branch_index]
        if branch_index == 0:
            if self.pr1.is_set:
                return self.pr1.val_SI
            elif self.dp1.is_set:
                if inconn.p.is_set:
                    return (inconn.p.val_SI - self.dp1.val_SI) / inconn.p.val_SI
                elif outconn.p.is_set:
                    return outconn.p.val_SI / (outconn.p.val_SI + self.dp1.val_SI)
        elif branch_index == 1:
            if self.pr2.is_set:
                return self.pr2.val_SI
            elif self.dp2.is_set:
                if inconn.p.is_set:
                    return (inconn.p.val_SI - self.dp2.val_SI) / inconn.p.val_SI
                elif outconn.p.is_set:
                    return outconn.p.val_SI / (outconn.p.val_SI + self.dp2.val_SI)
        return 0.98

    def get_parameters(self):
        self.two_phase_hot_side = False
        self.two_phase_cold_side = False
        return {
            'Q': dc_cp(
                max_val=0,
                func=self.energy_balance_hot_func,
                variables_columns=self.energy_balance_hot_variables_columns,
                solve_isolated=self.energy_balance_hot_solve_isolated,
                deriv=self.energy_balance_hot_deriv,
                tensor=self.energy_balance_hot_tensor,
                latex=self.energy_balance_hot_func_doc,
                num_eq=1,
                property_data=cpd['Q'],
                SI_unit=cpd['Q']['SI_unit'],
                scale=ps['m']['scale'] * ps['h']['scale'],
                var_scale=ps['m']['scale'] * ps['h']['scale']),
            'kA': dc_cp(
                min_val=0,
                func=self.kA_func,
                variables_columns=self.kA_variables_columns,
                solve_isolated=self.KA_solve_isolated,
                latex=self.kA_func_doc,
                deriv=self.kA_deriv,
                tensor=self.kA_tensor,
                num_eq=1,
                property_data=cpd['kA'],
                SI_unit=cpd['kA']['SI_unit'],
                scale=ps['m']['scale'] * ps['h']['scale'],
                var_scale=ps['m']['scale'] * ps['h']['scale'] / ps['DT']['scale']),
            'KDTA': dc_cp(
                min_val=0,
                func=self.KDTA_func,
                variables_columns=self.KDTA_variables_columns,
                solve_isolated=self.KDTA_solve_isolated,
                latex=self.KDTA_func_doc,
                deriv=self.KDTA_deriv,
                repair_matrix=self.KDTA_repair_matrix,
                tensor=self.KDTA_tensor,
                num_eq=2,
                property_data=cpd['kA'],
                SI_unit=cpd['kA']['SI_unit'],
                scale=ps['h']['scale'],
                var_scale=ps['m']['scale'] * ps['h']['scale'] / ps['DT']['scale']),
            'DTM': dc_cp(
                min_val=0,
                func=self.DTM_func,
                variables_columns=self.DTM_variables_columns,
                solve_isolated=self.DTM_solve_isolated,
                deriv=self.DTM_deriv,
                repair_matrix=self.DTM_repair_matrix,
                tensor=None,
                latex=self.DTM_func_doc,
                num_eq=1,
                property_data=cpd['DT'],
                SI_unit=cpd['DT']['SI_unit'],
                scale=ps['DT']['scale'],
                var_scale=ps['DT']['scale']
            ),
            'DT_log': dc_cp(
                min_val=0,
                is_result=True,
                property_data=cpd['DT'],
                SI_unit=cpd['DT']['SI_unit'],
                scale=ps['DT']['scale']),
            'DTU': dc_cp(
                min_val=0,
                func=self.DTU_func,
                variables_columns=self.DTU_variables_columns,
                solve_isolated=self.DTU_solve_isolated,
                deriv=self.DTU_deriv,
                repair_matrix=self.DTU_repair_matrix,
                tensor=self.DTU_tensor,
                latex=self.DTU_func_doc,
                num_eq=1,
                property_data=cpd['DT'],
                SI_unit=cpd['DT']['SI_unit'],
                scale=ps['DT']['scale'],
                var_scale=ps['DT']['scale']
                ),
            'DTL': dc_cp(
                min_val=0,
                func=self.DTL_func,
                variables_columns=self.DTL_variables_columns,
                solve_isolated=self.DTL_solve_isolated,
                deriv=self.DTL_deriv,
                repair_matrix=self.DTL_repair_matrix,
                tensor=self.DTL_tensor,
                latex=self.DTL_func_doc,
                num_eq=1,
                property_data=cpd['DT'],
                SI_unit=cpd['DT']['SI_unit'],
                scale=ps['DT']['scale'],
                var_scale=ps['DT']['scale']
            ),
            'DTN_min': dc_cp(
                min_val=0,
                num_eq=1,
                func=self.DTN_min_func,
                variables_columns=self.DTN_min_variables_columns,
                solve_isolated=self.DTN_min_solve_isolated,
                deriv=self.DTN_min_deriv,
                tensor=self.DTN_min_tensor,
                property_data=cpd['DT'],
                SI_unit=cpd['DT']['SI_unit'],
                scale=ps['DT']['scale'],
                var_scale=ps['DT']['scale']
            ),
            'pr1': dc_cp(
                min_val=1e-4,
                max_val=1,
                func=self.pr_func,
                variables_columns=self.pr_variables_columns,
                solve_isolated=self.pr_solve_isolated,
                deriv=self.pr_deriv,
                tensor=self.pr_tensor,
                latex=self.pr_func_doc,
                func_params={'pr': 'pr1', 'inconn': 0, 'outconn': 0},
                num_eq=1,
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
                scale=ps['p']['scale'],
                var_scale=ps['pr']['scale']),
            'pr2': dc_cp(
                min_val=1e-4,
                max_val=1,
                func=self.pr_func,
                variables_columns=self.pr_variables_columns,
                solve_isolated=self.pr_solve_isolated,
                latex=self.pr_func_doc,
                deriv=self.pr_deriv,
                tensor=self.pr_tensor,
                func_params={'pr': 'pr2', 'inconn': 1, 'outconn': 1},
                num_eq=1,
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
                scale=ps['p']['scale'],
                var_scale=ps['pr']['scale']),
            'dp1': dc_cp(
                min_val=0,
                max_val=1e15,
                num_eq=1,
                deriv=self.dp_deriv,
                variables_columns=self.dp_variables_columns,
                solve_isolated=self.dp_solve_isolated,
                func=self.dp_func,
                tensor=self.dp_tensor,
                func_params={'dp': 'dp1', 'inconn': 0, 'outconn': 0},
                property_data=cpd['dp'],
                SI_unit=cpd['dp']['SI_unit'],
                scale=ps['p']['scale'],
                var_scale=ps['p']['scale']
            ),
            'dp2': dc_cp(
                min_val=0,
                max_val=1e15,
                num_eq=1,
                deriv=self.dp_deriv,
                variables_columns=self.dp_variables_columns,
                solve_isolated=self.dp_solve_isolated,
                func=self.dp_func,
                tensor=self.dp_tensor,
                func_params={'dp': 'dp2', 'inconn': 1, 'outconn': 1},
                property_data=cpd['dp'],
                SI_unit=cpd['dp']['SI_unit'],
                scale=ps['p']['scale'],
                var_scale=ps['p']['scale']
            ),
            'pr1_fit': dc_fit(
                rule='constant',
                constant=self.pr_constant_func_,
                default=self.pr_default_func_,
            ),
            'dp1_fit': dc_fit(
                rule='constant',
                constant=self.dp_constant_func_,
                default=self.dp_default_func_,
            ),
            'pr2_fit': dc_fit(
                rule='constant',
                constant=self.pr_constant_func_,
                default=self.pr_default_func_,
            ),
            'dp2_fit': dc_fit(
                rule='constant',
                constant=self.dp_constant_func_,
                default=self.dp_default_func_,
            ),
            'kA_fit': dc_fit(
                choice=[],
                static=self.kA_static_func_,
                constant=self.kA_constant_func_,
                default=self.kA_default_func_,
                charline=self.kA_charline_func_,
            ),
            'kA_char1': dc_cc(
                param='m',
                char_params={'type': 'rel', 'inconn': 0, 'outconn': 0}
            ),
            'kA_char2': dc_cc(
                param='m',
                char_params={'type': 'rel', 'inconn': 1, 'outconn': 1}
            ),
            'KDTA_fit': dc_fit(
                choice=[],
                static=self.KDTA_static_func_,
                constant=self.KDTA_constant_func_,
                default=self.KDTA_default_func_,
                charline=self.KDTA_charline_func_,
                charmap=self.KDTA_charmap_func_,
                self_defined=self.KDTA_self_defined_func_,
            ),
            'KDTA_char1': dc_cc(
                param='m',
                char_params={'type': 'rel', 'inconn': 0, 'outconn': 0}
            ),
            'KDTA_char2': dc_cc(
                param='m',
                char_params={'type': 'rel', 'inconn': 1, 'outconn': 1}
            ),
            'hf1': dc_cp(
                val=0,
                val_SI=50,
                min_val=0,
                is_property=True,
                is_result=True,
                property_data=cpd['hf'],
                SI_unit=cpd['hf']['SI_unit'],
            ),
            'hf2': dc_cp(
                val=0,
                val_SI=500,
                min_val=0,
                is_property=True,
                is_result=True,
                property_data=cpd['hf'],
                SI_unit=cpd['hf']['SI_unit'],
            ),
            'fA': dc_cp(
                val=0,
                min_val=0,
                is_property=True,
                is_result=True,
                property_data=cpd['fA'],
                SI_unit=cpd['fA']['SI_unit'],
            ),
            'exm1': dc_cp(
                val=0,
                val_SI=0.61,
                min_val=0,
                is_property=True,
                is_result=True,
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
            ),
            'exm2': dc_cp(
                val=0,
                val_SI=0.784,
                min_val=0,
                is_property=True,
                is_result=True,
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
            ),
            'zeta1': dc_cp(
                min_val=0,
                max_val=1e15,
                is_property=True,
                is_result=True,
                property_data=cpd['zeta'],
                SI_unit=cpd['zeta']['SI_unit'],
            ),
            'zeta2': dc_cp(
                min_val=0,
                max_val=1e15,
                is_property=True,
                is_result=True,
                property_data=cpd['zeta'],
                SI_unit=cpd['zeta']['SI_unit'],
            ),
            'eff_cold': dc_cp(
                min_val=0,
                max_val=1,
                func=self.eff_cold_func,
                variables_columns=self.eff_cold_variables_columns,
                solve_isolated=self.eff_cold_solve_isolated,
                deriv=self.eff_cold_deriv,
                tensor=self.eff_cold_tensor,
                num_eq=1,
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
                scale=ps['h']['scale'],
                var_scale=ps['eff']['scale']
            ),
            'eff_hot': dc_cp(
                min_val=0,
                max_val=1,
                func=self.eff_hot_func,
                variables_columns=self.eff_hot_variables_columns,
                solve_isolated=self.eff_hot_solve_isolated,
                deriv=self.eff_hot_deriv,
                tensor=self.eff_hot_tensor,
                num_eq=1,
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
                scale=ps['h']['scale'],
                var_scale=ps['eff']['scale']
            ),
            'eff_max': dc_cp(
                min_val=0,
                max_val=1,
                func=self.eff_max_func,
                variables_columns=self.eff_max_variables_columns,
                solve_isolated=self.eff_max_solve_isolated,
                deriv=self.eff_max_deriv,
                tensor=self.eff_max_tensor,
                num_eq=1,
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
                scale=ps['eff']['scale'],
                var_scale=ps['eff']['scale']
            )
        }

    def get_mandatory_constraints(self):
        if self.KDTA.is_set:
            return {}
        return {
            'energy_balance_constraints': dc_cons(
                func=self.energy_balance_func,
                variables_columns=self.energy_balance_variables_columns,
                solve_isolated=self.energy_balance_solve_isolated,
                deriv=self.energy_balance_deriv,
                tensor=self.energy_balance_tensor,
                constant_deriv=False,
                latex=self.energy_balance_func_doc,
                num_eq=1,
                scale=ps['m']['scale'] * ps['h']['scale'])
        }

    def get_bypass_constraints(self):
        return {
            'pressure_equality_constraints': dc_cons(
                func=self.pressure_equality_func,
                variables_columns=self.pressure_equality_variables_columns,
                solve_isolated=self.pressure_equality_solve_isolated,
                deriv=self.pressure_equality_deriv,
                tensor=self.pressure_equality_tensor,
                constant_deriv=False,
                latex=self.pressure_equality_func_doc,
                num_eq=self.num_i,
                scale=ps['p']['scale']
            ),
            'enthalpy_equality_constraints': dc_cons(
                func=self.enthalpy_equality_func,
                variables_columns=self.enthalpy_equality_variables_columns,
                solve_isolated=self.enthalpy_equality_solve_isolated,
                deriv=self.enthalpy_equality_deriv,
                tensor=self.enthalpy_equality_tensor,
                constant_deriv=False,
                latex=self.enthalpy_equality_func_doc,
                num_eq=self.num_i,
                scale=ps['h']['scale']
            )
        }

    def energy_balance_func(self):
        r"""
        Equation for heat exchanger energy balance.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = \dot{m}_{in,1} \cdot \left(h_{out,1} - h_{in,1} \right) +
                \dot{m}_{in,2} \cdot \left(h_{out,2} - h_{in,2} \right)
        """
        return (
            self.inl[0].m.val_SI
            * (self.outl[0].h.val_SI - self.inl[0].h.val_SI)
            + self.inl[1].m.val_SI
            * (self.outl[1].h.val_SI - self.inl[1].h.val_SI)
        )

    def energy_balance_variables_columns(self):
        variables_columns1 = []
        for _c_num, i in enumerate(self.inl):
            o = self.outl[_c_num]
            variables_columns1 += [data.J_col for data in [i.m, i.h, o.h] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def energy_balance_take_effect(self):
        pass

    def energy_balance_solve_isolated(self):
        return False

    def energy_balance_func_doc(self, label):
        r"""
        Equation for heat exchanger energy balance.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        latex = (
            r'0 = \dot{m}_\mathrm{in,1} \cdot \left(h_\mathrm{out,1} -'
            r' h_\mathrm{in,1} \right) +\dot{m}_\mathrm{in,2} \cdot '
            r'\left(h_\mathrm{out,2} - h_\mathrm{in,2} \right)')
        return generate_latex_eq(self, latex, label)

    def energy_balance_deriv(self, increment_filter, k):
        r"""
        Partial derivatives of energy balance function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        if self.inl[0].m == self.inl[1].m:
            if self.is_variable(self.inl[0].m, increment_filter):
                self.network.jacobian[k, self.inl[0].m.J_col] = (
                        (self.outl[0].h.val_SI - self.inl[0].h.val_SI)
                        + (self.outl[1].h.val_SI - self.inl[1].h.val_SI)
                )
        else:
            for _c_num, i in enumerate(self.inl):
                o = self.outl[_c_num]
                if self.is_variable(i.m, increment_filter):
                    self.network.jacobian[k, i.m.J_col] = o.h.val_SI - i.h.val_SI

        for _c_num, i in enumerate(self.inl):
            o = self.outl[_c_num]
            if self.is_variable(i.h, increment_filter):
                self.network.jacobian[k, i.h.J_col] = -i.m.val_SI
            if self.is_variable(o.h, increment_filter):
                self.network.jacobian[k, o.h.J_col] = i.m.val_SI

    def energy_balance_tensor(self, increment_filter, k):
        if self.inl[0].m == self.inl[1].m:
            if self.is_variable(self.inl[0].m, increment_filter):
                if self.is_variable(self.outl[0].h, increment_filter):
                    self.network.tensor[self.inl[0].m.J_col, self.outl[0].h.J_col, k] = 1
                    self.network.tensor[self.outl[0].h.J_col, self.inl[0].m.J_col, k] = 1
                if self.is_variable(self.outl[1].h, increment_filter):
                    self.network.tensor[self.inl[0].m.J_col, self.outl[1].h.J_col, k] = 1
                    self.network.tensor[self.outl[1].h.J_col, self.inl[0].m.J_col, k] = 1
                if self.is_variable(self.inl[0].h, increment_filter):
                    self.network.tensor[self.inl[0].m.J_col, self.inl[0].h.J_col, k] = -1
                    self.network.tensor[self.inl[0].h.J_col, self.inl[0].m.J_col, k] = -1
                if self.is_variable(self.inl[1].h, increment_filter):
                    self.network.tensor[self.inl[0].m.J_col, self.inl[1].h.J_col, k] = -1
                    self.network.tensor[self.inl[1].h.J_col, self.inl[0].m.J_col, k] = -1
        else:
            for _c_num, i in enumerate(self.inl):
                o = self.outl[_c_num]
                if self.is_variable(i.m, increment_filter):
                    if self.is_variable(i.h, increment_filter):
                        self.network.tensor[i.m.J_col, i.h.J_col, k] = -1
                        self.network.tensor[i.h.J_col, i.m.J_col, k] = -1
                    if self.is_variable(o.h, increment_filter):
                        self.network.tensor[i.m.J_col, o.h.J_col, k] = 1
                        self.network.tensor[o.h.J_col, i.m.J_col, k] = 1

    def energy_balance_hot_func(self):
        r"""
        Equation for hot side heat exchanger energy balance.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 =\dot{m}_{in,1} \cdot \left(h_{out,1}-h_{in,1}\right)-\dot{Q}
        """
        return self.inl[0].m.val_SI * (
            self.outl[0].h.val_SI - self.inl[0].h.val_SI
        ) - self.Q.val_SI

    def energy_balance_hot_variables_columns(self):
        i = self.inl[0]
        o = self.outl[0]
        variables_columns1 = [data.J_col for data in [i.m, i.h, o.h] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def energy_balance_hot_take_effect(self):
        pass

    def energy_balance_hot_solve_isolated(self):
        i = self.inl[0]
        o = self.outl[0]
        if sum([1 if data.is_var else 0 for data in [i.m, i.h, o.h]]) > 1:
            return False
        if i.m.is_var:
            i.m.val_SI = self.Q.val_SI / (o.h.val_SI - i.h.val_SI)
            i.m.is_set = True
            i.m.is_var = False
            self.Q.is_set = False
            return True
        elif i.h.is_var:
            i.h.val_SI = o.h.val_SI - self.Q.val_SI / i.m.val_SI
            i.h.is_set = True
            i.h.is_var = False
            self.Q.is_set = False
            return True
        elif o.h.is_var:
            o.h.val_SI = i.h.val_SI + self.Q.val_SI / i.m.val_SI
            o.h.is_set = True
            o.h.is_var = False
            self.Q.is_set = False
            return True
        else:
            self.Q.is_set = False
            return True

    def energy_balance_hot_func_doc(self, label):
        r"""
        Equation for hot side heat exchanger energy balance.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        latex = (
            r'0 =\dot{m}_{in,1} \cdot \left(h_{out,1}-'
            r'h_{in,1}\right)-\dot{Q}')
        return generate_latex_eq(self, latex, label)

    def energy_balance_hot_deriv(self, increment_filter, k):
        r"""
        Partial derivatives for hot side heat exchanger energy balance.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        i = self.inl[0]
        o = self.outl[0]
        if self.is_variable(i.m, increment_filter):
            self.network.jacobian[k, i.m.J_col] = o.h.val_SI - i.h.val_SI
        if self.is_variable(i.h, increment_filter):
            self.network.jacobian[k, i.h.J_col] = -i.m.val_SI
        if self.is_variable(o.h, increment_filter):
            self.network.jacobian[k, o.h.J_col] = i.m.val_SI

    def energy_balance_hot_tensor(self, increment_filter, k):
        i = self.inl[0]
        o = self.outl[0]
        if self.is_variable(i.m, increment_filter):
            if self.is_variable(i.h, increment_filter):
                self.network.tensor[i.m.J_col, i.h.J_col, k] = -1
                self.network.tensor[i.h.J_col, i.m.J_col, k] = -1
            if self.is_variable(o.h, increment_filter):
                self.network.tensor[i.m.J_col, o.h.J_col, k] = 1
                self.network.tensor[o.h.J_col, i.m.J_col, k] = 1

    def calc_DTM(self):
        i1 = self.inl[0]
        o1 = self.outl[0]
        i2 = self.inl[1]
        o2 = self.outl[1]
        if len(i2.fluid.val) > 1 and len(i1.fluid.val) > 1:
            return np.nan
        h_mid_l = h_mix_pQ((i2.p.val_SI + o2.p.val_SI) / 2, 0, i2.fluid_data)
        if o2.h.val_SI > h_mid_l > i2.h.val_SI:
            # temperature of saturated liquid point in cold side
            T_lower = T_sat_p((i2.p.val_SI + o2.p.val_SI) / 2, i2.fluid_data, mixing_rule=i2.mixing_rule)
            # enthalpy of corresponding point in hot side
            h_mid_u = o1.h.val_SI + ((h_mid_l - i2.h.val_SI) * i2.m.val_SI / o1.m.val_SI)
            if h_mid_u < i1.h.val_SI:
                T_uper = T_mix_ph((o1.p.val_SI + i1.p.val_SI) / 2, h_mid_u, o1.fluid_data, o1.mixing_rule,
                                  T0=o1.T.val_SI)
            else:
                T_uper = T_mix_ph(i1.p.val_SI, i1.h.val_SI, o1.fluid_data, o1.mixing_rule, T0=o1.T.val_SI)
            delta_T = T_uper - T_lower
        elif "H2O" in i1.fluid.val or 'wator' in i1.fluid.val:
            logger.debug(f"the mid point above range of {self.__class__.__name__}: {self.label}")
            if "H2O" in i1.fluid.val:
                fluid_c = "H2O"
            else:
                fluid_c = 'wator'
            fluid_dict = {fluid_c: {
                "wrapper": i1.fluid.wrapper[fluid_c],
                "mass_fraction": 1}}
            T_wator_sat = T_sat_p(i1.p.val_SI, fluid_dict, mixing_rule=i1.mixing_rule)
            h_mid_u = h_mix_pT(i1.p.val_SI, T_wator_sat, i1.fluid_data, i1.mixing_rule)
            if i1.h.val_SI > h_mid_u > o1.h.val_SI:
                h_mid_l = o2.h.val_SI - (i1.h.val_SI - h_mid_u) * i1.m.val_SI / o2.m.val_SI
                T_uper = T_wator_sat
                if h_mid_l > i2.h.val_SI:
                    T_lower = T_mix_ph((i2.p.val_SI + o2.p.val_SI) / 2, h_mid_l, i2.fluid_data, i2.mixing_rule,
                                       T0=i2.T.val_SI)
                else:
                    T_lower = T_mix_ph(i2.p.val_SI, i2.h.val_SI, i2.fluid_data, i2.mixing_rule, T0=i2.T.val_SI)
                delta_T = T_uper - T_lower
            else:
                delta_T = np.nan
        else:
            delta_T = np.nan
        return delta_T

    def DTM_func(self):
        """
        Calculate pinch temperature difference.

        :return:
        """
        i1 = self.inl[0]
        o1 = self.outl[0]
        i2 = self.inl[1]
        o2 = self.outl[1]

        def calculate_common_DTM():
            if self.DTU.is_set:
                return o1.calc_T()-i2.calc_T()
            elif self.DTL.is_set:
                return i1.calc_T()-o2.calc_T()
            elif self.DTU.is_set and self.DTL.is_set:
                msg = f"{self.__class__.__name__}: {self.label} has been over constrained in [DTU, DTL, DTM]"
                logger.error(msg)
                raise AURORANetworkError(msg)
            else:
                return min(i1.calc_T()-o2.calc_T(), o1.calc_T()-i2.calc_T())

        h_mid_l = h_mix_pQ((o2.p.val_SI + i2.p.val_SI) / 2, 0, i2.fluid_data)
        if o2.h.val_SI > h_mid_l > i2.h.val_SI:
            # temperature of saturated liquid point in cold side
            T_lower = T_sat_p((o2.p.val_SI + i2.p.val_SI) / 2, i2.fluid_data, mixing_rule=i2.mixing_rule)
            # enthalpy of corresponding point in hot side
            h_mid_u1 = o1.h.val_SI + ((h_mid_l - i2.h.val_SI) * i2.m.val_SI / o1.m.val_SI)
            h_mid_u2 = i1.h.val_SI - ((o2.h.val_SI - h_mid_l) * i2.m.val_SI / o1.m.val_SI)
            h_mid_u = (h_mid_u1 + h_mid_u2) / 2
            T_uper = T_mix_ph((o1.p.val_SI + i1.p.val_SI) / 2, h_mid_u, o1.fluid_data, o1.mixing_rule, T0=o1.T.val_SI)
            # if o1.h.val_SI < h_mid_u < i1.h.val_SI:
            #     T_uper = T_mix_ph((o1.p.val_SI + i1.p.val_SI) / 2, h_mid_u, o1.fluid_data, o1.mixing_rule, T0=o1.T.val_SI)
            # else:
            #     T_uper = T_mix_ph(i1.p.val_SI, i1.h.val_SI, o1.fluid_data, o1.mixing_rule, T0=o1.T.val_SI)
            delta_T = T_uper - T_lower
        elif "H2O" in i1.fluid.val or 'water' in i1.fluid.val:
            logger.debug(f"the mid point above range of {self.__class__.__name__}: {self.label}")
            if "H2O" in i1.fluid.val:
                fluid_c = "H2O"
            else:
                fluid_c = 'water'
            fluid_dict = {fluid_c: {
                "wrapper": i1.fluid.wrapper[fluid_c],
                "mass_fraction": 1}}
            T_wator_sat = T_sat_p((o1.p.val_SI + i1.p.val_SI) / 2, fluid_dict, mixing_rule=i1.mixing_rule)
            h_mid_u = h_mix_pT((o1.p.val_SI + i1.p.val_SI) / 2, T_wator_sat, i1.fluid_data, i1.mixing_rule)
            if i1.h.val_SI > h_mid_u > o1.h.val_SI:
                h_mid_l1 = o2.h.val_SI - (i1.h.val_SI - h_mid_u) * i1.m.val_SI / o2.m.val_SI
                h_mid_l2 = i2.h.val_SI + ((h_mid_u - o1.h.val_SI) * i1.m.val_SI / o2.m.val_SI)
                h_mid_l = (h_mid_l1 + h_mid_l2) / 2
                T_uper = T_wator_sat
                T_lower = T_mix_ph((i2.p.val_SI + o2.p.val_SI) / 2, h_mid_l, i2.fluid_data, i2.mixing_rule, T0=i2.T.val_SI)
                # if h_mid_l > i2.h.val_SI:
                #     T_lower = T_mix_ph((i2.p.val_SI + o2.p.val_SI) / 2, h_mid_l, i2.fluid_data, i2.mixing_rule, T0=i2.T.val_SI)
                # else:
                #     T_lower = T_mix_ph(i2.p.val_SI, i2.h.val_SI, i2.fluid_data, i2.mixing_rule, T0=i2.T.val_SI)
                delta_T = T_uper - T_lower
            else:
                delta_T = calculate_common_DTM()
        else:
            delta_T = calculate_common_DTM()

        return self.DTM.val_SI - delta_T

    def DTM_variables_columns(self):
        i1 = self.inl[0]
        o1 = self.outl[0]
        i2 = self.inl[1]
        o2 = self.outl[1]
        variables_columns1 = []
        data_list = [data for c in [i1, o1, i2, o2] for data in [c.h]]  # [c.p, c.h]
        # if i1.m == i2.m:
        #     data_list += [i1.m]
        # else:
        #     data_list += [i1.m, i2.m]
        variables_columns1 += [data.J_col for data in data_list if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def DTM_take_effect(self):
        pass

    def DTM_solve_isolated(self):
        return False

    def DTM_func_doc(self, label):
        latex = f'nothing now'
        return generate_latex_eq(self, latex, label)

    def DTM_deriv(self, increment_filter, k):
        f = self.DTM_func
        i1 = self.inl[0]
        o1 = self.outl[0]
        i2 = self.inl[1]
        o2 = self.outl[1]
        if self.is_variable(i1.h):
            self.network.jacobian[k, i1.h.J_col] = self.numeric_deriv(f, 'h', i1)
        if self.is_variable(o1.h):
            self.network.jacobian[k, o1.h.J_col] = self.numeric_deriv(f, 'h', o1)
        if self.is_variable(i2.h):
            self.network.jacobian[k, i2.h.J_col] = self.numeric_deriv(f, 'h', i2)
        if self.is_variable(o2.h):
            self.network.jacobian[k, o2.h.J_col] = self.numeric_deriv(f, 'h', o2)

    def DTM_repair_matrix(self, property_):
        pass

    def calculate_DT_log(self):
        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0]
        o2 = self.outl[1]
        # temperature value manipulation for convergence stability
        T_i1 = i1.calc_T()
        T_i2 = i2.calc_T()
        T_o1 = o1.calc_T()
        T_o2 = o2.calc_T()
        DTU = T_i1 - T_o2
        DTL = T_o1 - T_i2
        if DTU == DTL:
            DT_log = DTL
        else:
            if DTU / DTL < 0:
                DT_log = (DTU + DTL) / 2
            else:
                DT_log = (DTL - DTU) / math.log((DTL) / (DTU))
        return DT_log

    def kA_func(self):
        r"""
        Calculate heat transfer from heat transfer coefficient.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                \[0={\left( {{{\dot{m}}}_{in,1}}\cdot \left( {{h}_{out,1}}-
                {{h}_{in,1}} \right)-{{{\dot{m}}}_{in,2}}\cdot \left( {{h}_{out,2}}-{{h}_{in,2}} \right) \right)}/{2}\;
                +kA\cdot \frac{{{T}_{out,1}}-
                {{T}_{in,2}}-{{T}_{in,1}}+{{T}_{out,2}}}
                {\ln \frac{{{T}_{out,1}}-{{T}_{in,2}}}{{{T}_{in,1}}-{{T}_{out,2}}}}\]

        """
        return (
            (self.inl[0].m.val_SI * (self.outl[0].h.val_SI - self.inl[0].h.val_SI) -
             self.inl[1].m.val_SI * (self.outl[1].h.val_SI - self.inl[1].h.val_SI)) / 2
            + self.kA.val_SI * self.calculate_DT_log()
        )

    def kA_variables_columns(self):
        i = self.inl[0]
        o = self.outl[0]
        variables_columns1 = []
        if i.m.is_var:
            variables_columns1.append(i.m.J_col)
        variables_columns1 += [data.J_col for c in self.inl + self.outl for data in [c.h] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def KA_take_effect(self):
        pass

    def KA_solve_isolated(self):
        return False

    def kA_func_doc(self, label):
        r"""
        Calculate heat transfer from heat transfer coefficient.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        latex = (
            r'0 = \dot{m}_\mathrm{in,1} \cdot \left( h_\mathrm{out,1} - '
            r'h_\mathrm{in,1}\right)+ kA \cdot \frac{T_\mathrm{out,1} - '
            r'T_\mathrm{in,2} - T_\mathrm{in,1} + T_\mathrm{out,2}}'
            r'{\ln{\frac{T_\mathrm{out,1} - T_\mathrm{in,2}}'
            r'{T_\mathrm{in,1} - T_\mathrm{out,2}}}}'
        )
        return generate_latex_eq(self, latex, label)

    def kA_deriv(self, increment_filter, k):
        r"""
        Partial derivatives of heat transfer coefficient function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        f = self.kA_func
        i = self.inl[0]
        o = self.outl[0]
        # if self.is_variable(i.m):
        #     self.network.jacobian[k, i.m.J_col] = o.h.val_SI - i.h.val_SI
        for c in self.inl + self.outl:
            # if self.is_variable(c.p):
            #     self.network.jacobian[k, c.p.J_col] = self.numeric_deriv(f, 'p', c)
            if self.is_variable(c.h):
                self.network.jacobian[k, c.h.J_col] = self.numeric_deriv(f, 'h', c)

    def kA_tensor(self, increment_filter, k):
        f = self.kA_func
        i = self.inl[0]
        o = self.outl[0]
        if self.is_variable(i.m):
            if self.is_variable(i.h, increment_filter):
                self.network.tensor[i.m.J_col, i.h.J_col, k] = -1
                self.network.tensor[i.h.J_col, i.m.J_col, k] = -1
            if self.is_variable(o.h, increment_filter):
                self.network.tensor[i.m.J_col, o.h.J_col, k] = 1
                self.network.tensor[o.h.J_col, i.m.J_col, k] = 1
        numeric_variables_list = ([('p', self.is_variable(c.p), c, c.p.J_col) for c in self.inl + self.outl] +
                                  [('h', self.is_variable(c.h), c, c.h.J_col) for c in self.inl + self.outl])
        self.generate_numerical_tensor(f, k, numeric_variables_list)

    def calc_KDTA(self):
        if self.KDTA.is_set:  # !!
            pass
        else:
            if self.kA.val_SI and self.kA.val_SI > 0:
                self.KDTA.val_SI = self.kA.val_SI
        self.distribute_nodes_properties_()
        KDTA_rule = self.KDTA_fit.rule
        self.KDTA_fit.rule = 'static'
        iteration = 0
        while True:
            residual = self.calculate_KDTA_residual_()
            deriv = self.calculate_KDTA_deriv_(self.KDTA)
            increment = [re / dv for re, dv in zip(residual, deriv)]
            self.KDTA.val_SI -= sum(increment) / len(increment)
            if self.KDTA.val_SI < self.KDTA.min_val:
                self.KDTA.val_SI = self.KDTA.min_val + 1e0
            if iteration > 40:
                msg = (f'The KDTA calculation of {self.__class__.__name__}: {self.label} '
                       f'has not converged after {iteration} iterations.')
                logger.debug(msg)
                break
            if np.linalg.norm(residual) < ERR ** 0.5:
                break
            iteration += 1
        self.KDTA_fit.rule = KDTA_rule
        return self.KDTA.val_SI

    def distribute_nodes_properties_(self):
        """
        Set initial properties of nodes.
        """
        pressure_data = {
            0: self.generate_interpolated_array_(self.inl[0].p.val_SI, self.outl[0].p.val_SI, self.nodes_num),
            1: self.generate_interpolated_array_(self.outl[1].p.val_SI, self.inl[1].p.val_SI, self.nodes_num),
        }
        enthalpy_data = {
            0: self.generate_interpolated_array_(self.inl[0].h.val_SI, self.outl[0].h.val_SI, self.nodes_num),
            1: self.generate_interpolated_array_(self.outl[1].h.val_SI, self.inl[1].h.val_SI, self.nodes_num),
        }
        for row in self.nodes.index:
            for col in range(self.nodes_num):
                node = self.nodes.loc[row, col]
                node.p.val_SI = pressure_data[row][col]
                node.h.val_SI = enthalpy_data[row][col]

    def calculate_KDTA_residual_(self):
        """
        Calculate enthalpy residual of mid-nodes.

        :return:
        enthalpy_residual : list
        """
        mid_idx = self.nodes_num // 2
        # left differential calculate
        for col in range(0, mid_idx-1):
            node_u_f = self.nodes.loc[0, col]
            node_l_f = self.nodes.loc[1, col]
            node_u_f.calc_T()
            node_l_f.calc_T()
            dQ = self.KDTA_fit() * (node_u_f.T.val_SI - node_l_f.T.val_SI) / (self.nodes_num - 2)
            node_u_c = self.nodes.loc[0, col+1]
            node_l_c = self.nodes.loc[1, col+1]
            node_u_c.h.val_SI = node_u_f.h.val_SI - dQ / node_u_f.m.val_SI
            node_l_c.h.val_SI = node_l_f.h.val_SI - dQ / node_l_f.m.val_SI
        # right differential calculate
        for col_ in range(-1, -mid_idx, -1):
            col = col_ + self.nodes_num
            node_u_f = self.nodes.loc[0, col]
            node_l_f = self.nodes.loc[1, col]
            node_u_f.calc_T()
            node_l_f.calc_T()
            dQ = self.KDTA_fit() * (node_u_f.T.val_SI - node_l_f.T.val_SI) / (self.nodes_num - 2)
            node_u_c = self.nodes.loc[0, col-1]
            node_l_c = self.nodes.loc[1, col-1]
            node_u_c.h.val_SI = node_u_f.h.val_SI + dQ / node_u_f.m.val_SI
            node_l_c.h.val_SI = node_l_f.h.val_SI + dQ / node_l_f.m.val_SI
        node_mid_u_left = self.nodes.loc[0, mid_idx-1]
        node_mid_u_right = self.nodes.loc[0, mid_idx]
        node_mid_l_left = self.nodes.loc[1, mid_idx-1]
        node_mid_l_right = self.nodes.loc[1, mid_idx]
        for node in [node_mid_u_left, node_mid_u_right, node_mid_l_left, node_mid_l_right]:
            node.calc_T()
        return [node_mid_u_left.h.val_SI - node_mid_u_right.h.val_SI, node_mid_l_left.h.val_SI - node_mid_l_right.h.val_SI]

    def calculate_KDTA_deriv_(self, property_):
        property_.val_SI += property_.property_data['differ']
        self.distribute_nodes_properties_()
        residual_u = self.calculate_KDTA_residual_()
        property_.val_SI -= property_.property_data['differ'] * 2
        self.distribute_nodes_properties_()
        residual_l = self.calculate_KDTA_residual_()
        property_.val_SI += property_.property_data['differ']
        deriv = [(ru - rl) / (property_.property_data['differ'] * 2)
                 for ru, rl in zip(residual_u, residual_l)]
        return np.array(deriv)

    def KDTA_func(self):
        self.distribute_nodes_properties_()
        return self.calculate_KDTA_residual_()

    def KDTA_variables_columns(self):
        i1 = self.inl[0]
        i2 = self.inl[1]
        variables_columns1 = []
        variables_columns1 += [data.J_col for c in self.inl + self.outl for data in [c.h] if data.is_var]
        variables_columns2 = variables_columns1.copy()
        if i1.m.is_var:
            variables_columns1.append(i1.m.J_col)
        if i2.m.is_var:
            variables_columns2.append(i2.m.J_col)
        variables_columns1.sort()
        variables_columns2.sort()
        return [variables_columns1, variables_columns2]

    def KDTA_take_effect(self):
        pass

    def KDTA_solve_isolated(self):
        return False

    def KDTA_func_doc(self, label):
        pass

    def KDTA_deriv(self, increment_filter, k):
        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0]
        o2 = self.outl[1]
        self.distribute_nodes_properties_()
        if i1.m == i2.m:
            if i1.m.is_var:
                self.network.jacobian[k: k + 2, i1.m.J_col] = self.calculate_KDTA_deriv_(i1.m)
        else:
            if i1.m.is_var:
                self.network.jacobian[k: k + 2, i1.m.J_col] = self.calculate_KDTA_deriv_(i1.m)
            if i2.m.is_var:
                self.network.jacobian[k: k + 2, i2.m.J_col] = self.calculate_KDTA_deriv_(i2.m)
        for c in [i1, i2, o1, o2]:
            if c.h.is_var:
                self.network.jacobian[k: k + 2, c.h.J_col] = self.calculate_KDTA_deriv_(c.h)

    def KDTA_repair_matrix(self, property_, row):
        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0]
        o2 = self.outl[1]
        for c in [i1, o1]:
            if property_ == c.h and row == 1:
                h0 = h_mix_pQ(c.p.val_SI, 0, c.fluid_data)
                h1 = h_mix_pQ(c.p.val_SI, 1, c.fluid_data)
                self.distribute_nodes_properties_()
                return -abs(self.calculate_KDTA_residual_()[row]) / max(c.h.val_SI - h0, h1 - c.h.val_SI)
        for c in [i2, o2]:
            if property_ == c.h and row == 0:
                h0 = h_mix_pQ(c.p.val_SI, 0, c.fluid_data)
                h1 = h_mix_pQ(c.p.val_SI, 1, c.fluid_data)
                self.distribute_nodes_properties_()
                return abs(self.calculate_KDTA_residual_()[row]) / max(c.h.val_SI - h0, h1 - c.h.val_SI)
        if property_ == i1.h and row == 0:
            return 1
        elif property_ == o1.h and row == 0:
            return -1
        elif property_ == o2.h and row == 1:
            return 1
        elif property_ == i2.h and row == 1:
            return -1
        msg = (f'Has no repair method for {property_.label} '
               f'about enthalpy difference in KDTA calculation at {self.nodes_properties[row]["side_name"]} in {self.__class__.__name__}: {self.label}')
        logger.error(msg)
        raise ValueError(msg)

    def KDTA_tensor(self, increment_filter, k):
        pass

    def kA_static_func_(self, **kwargs):
        return self.kA.val_SI

    def kA_constant_func_(self, **kwargs):
        if self.fA.is_set and self.hf1.is_set and self.hf2.is_set:
            return self.fA.val_SI / (1 / self.hf1.val_SI + 1 / self.hf2.val_SI)
        else:
            return self.kA.val_SI

    def kA_default_func_(self, **kwargs):
        alfa1 = (self.inl[0].m.val_SI / self.inl[0].m.design) ** self.exm1.val_SI
        alfa2 = (self.inl[1].m.val_SI / self.inl[1].m.design) ** self.exm2.val_SI
        fkA = 2 / (1 / alfa1 + 1 / alfa2)  # total kA
        return self.kA.design * fkA

    def kA_charline_func_(self, **kwargs):
        p1 = self.kA_char1.param
        p2 = self.kA_char2.param
        f1 = self.get_char_expr(p1, **self.kA_char1.char_params)
        f2 = self.get_char_expr(p2, **self.kA_char2.char_params)
        fkA1 = self.kA_char1.char_func.evaluate(f1)
        fkA2 = self.kA_char2.char_func.evaluate(f2)
        fkA = 2 / (1 / fkA1 + 1 / fkA2)  # total kA
        return self.kA.design * fkA

    def KDTA_static_func_(self, **kwargs):
        return self.KDTA.val_SI

    def KDTA_constant_func_(self, **kwargs):
        if self.fA.is_set and self.hf1.is_set and self.hf2.is_set:
            return self.fA.val_SI / (1 / self.hf1.val_SI + 1 / self.hf2.val_SI)
        else:
            return self.KDTA.val_SI

    def KDTA_default_func_(self, **kwargs):
        alfa1 = (self.inl[0].m.val_SI / self.inl[0].m.design) ** self.exm1.val_SI
        alfa2 = (self.inl[1].m.val_SI / self.inl[1].m.design) ** self.exm2.val_SI
        kA = self.fA.design / (1 / (self.hf1.design * alfa1) + 1 / (self.hf2.design * alfa2))
        return kA

    def KDTA_charline_func_(self, **kwargs):
        if not self.KDTA_char1.is_set:
            self.KDTA_char1.char_func = CharLine(x=[0, 1], y=[1, 1])
        if not self.KDTA_char2.is_set:
            self.KDTA_char2.char_func = CharLine(x=[0, 1], y=[1, 1])
        p1 = self.KDTA_char1.param
        p2 = self.KDTA_char2.param
        f1 = self.get_char_expr(p1, **self.KDTA_char1.char_params)
        f2 = self.get_char_expr(p2, **self.KDTA_char2.char_params)
        alfa1 = self.KDTA_char1.char_func.evaluate(f1)
        alfa2 = self.KDTA_char2.char_func.evaluate(f2)
        kA = self.fA.design / (1 / (self.hf1.design * alfa1) + 1 / (self.hf2.design * alfa2))
        return kA

    def KDTA_charmap_func_(self, **kwargs):
        pass

    def KDTA_self_defined_func_(self, **kwargs):
        pass

    def DTU_func(self):
        r"""
        Equation for upper terminal temperature difference.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = DT_{U} - T_{in,1} + T_{out,2}
        """
        i = self.inl[0]
        o = self.outl[1]
        T_i1 = i.calc_T()
        T_o2 = o.calc_T()
        return self.DTU.val_SI - T_i1 + T_o2

    def DTU_variables_columns(self):
        variables_colmns1 = [data.J_col for c in [self.inl[0], self.outl[1]] for data in [c.h] if data.is_var]  # [c.p, c.h]
        variables_colmns1.sort()
        return [variables_colmns1]

    def DTU_take_effect(self):
        pass

    def DTU_solve_isolated(self):
        i = self.inl[0]
        o = self.outl[1]
        if i.fluid.is_var or o.fluid.is_var:
            return False
        if sum([1 if data.is_var else 0 for data in [i.p, i.h, o.p, o.h]]) > 1:  # [i.p, i.h, o.p, o.h]
            return False
        if i.h.is_var:
            T_i1 = self.DTU.val_SI + o.calc_T()
            i.h.val_SI = h_mix_pT(i.p.val_SI, T_i1, i.fluid_data, i.mixing_rule)
            i.h.is_set = True
            i.h.is_var = False
            self.DTU.is_set = False
            return True
        elif i.p.is_var:
            T_i1 = self.DTU.val_SI + o.calc_T()
            i.p.val_SI = p_mix_hT(i.h.val_SI, T_i1, i.fluid_data, i.mixing_rule)
            i.p.is_set = True
            i.p.is_var = False
            self.DTU.is_set = False
            return True
        elif o.p.is_var:
            T_o2 = i.calc_T() - self.DTU.val_SI
            o.p.val_SI = p_mix_hT(o.h.val_SI, T_o2, o.fluid_data, o.mixing_rule)
            o.p.is_set = True
            o.p.is_var = False
            self.DTU.is_set = False
            return True
        elif o.h.is_var:
            T_o2 = i.calc_T() - self.DTU.val_SI
            o.h.val_SI = h_mix_pT(o.p.val_SI, T_o2, o.fluid_data, o.mixing_rule)
            o.h.is_set = True
            o.h.is_var = False
            self.DTU.is_set = False
            return True
        else:
            self.DTU.is_set = False
            return True

    def DTU_func_doc(self, label):
        r"""
        Equation for upper terminal temperature difference.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        latex = r'0 = DT_\mathrm{U} - T_\mathrm{in,1} + T_\mathrm{out,2}'
        return generate_latex_eq(self, latex, label)

    def DTU_deriv(self, increment_filter, k):
        """
        Calculate partial derivates of upper terminal temperature function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        i = self.inl[0]
        o = self.outl[1]
        # if self.is_variable(i.p, increment_filter):
        #     self.network.jacobian[k, i.p.J_col] = -dT_mix_dph(i.p.val_SI, i.h.val_SI, i.fluid_data, i.mixing_rule, T0=i.T.val_SI)
        # if self.is_variable(o.p, increment_filter):
        #     self.network.jacobian[k, o.p.J_col] = dT_mix_dph(o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule, T0=o.T.val_SI)
        if self.is_variable(i.h, increment_filter):
            self.network.jacobian[k, i.h.J_col] = -dT_mix_pdh(i.p.val_SI, i.h.val_SI, i.fluid_data, i.mixing_rule, T0=i.T.val_SI)
        if self.is_variable(o.h, increment_filter):
            self.network.jacobian[k, o.h.J_col] = dT_mix_pdh(o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule, T0=o.T.val_SI)

    def DTU_repair_matrix(self, property_):
        """
        Repair upper terminal temperature derivative.

        :param property_:
        :return:
        """
        i = self.inl[0]
        o = self.outl[1]
        if property_ == i.h:
            h0 = h_mix_pQ(i.p.val_SI, 0, i.fluid_data)
            h1 = h_mix_pQ(i.p.val_SI, 1, i.fluid_data)
            return -abs(self.DTU.val_SI - i.calc_T() + o.calc_T()) / max(i.h.val_SI - h0, h1 - i.h.val_SI)
        elif property_ == o.h:
            h0 = h_mix_pQ(o.p.val_SI, 0, o.fluid_data)
            h1 = h_mix_pQ(o.p.val_SI, 1, o.fluid_data)
            return abs(self.DTU.val_SI - i.calc_T() + o.calc_T()) / max(o.h.val_SI - h0, h1 - o.h.val_SI)
        else:
            msg = f"variable: {property_.label} is not a valid property in DTU_repair_matrix of {self.__class__.__name__}: {self.label}"
            raise ValueError(msg)

    def DTU_tensor(self, increment_filter, k):
        f = self.DTU_func
        numeric_variables_list = ([('p', self.is_variable(c.p, increment_filter), c, c.p.J_col) for c in [self.inl[0], self.outl[1]]] +
                                  [('h', self.is_variable(c.h, increment_filter), c, c.h.J_col) for c in [self.inl[0], self.outl[1]]])
        self.generate_numerical_tensor(f, k, numeric_variables_list)

    def DTU_sh_func(self):
        """
        Calculate upper terminal temperature difference based on saturated temperature of hot side.

        :return:
        """
        i = self.inl[0]
        o = self.outl[1]
        T_i1 = i.calc_T_sat()
        T_o2 = o.calc_T()
        return self.DTU_sh.val_SI - T_i1 + T_o2

    def DTU_sh_variables_columns(self):
        variables_colmns1 = [data.J_col for c in [self.outl[1]] for data in [c.h] if data.is_var]  # [c.p, c.h]
        variables_colmns1.sort()
        return [variables_colmns1]

    def DTU_sh_take_effect(self):
        pass

    def DTU_sh_solve_isolated(self):
        i = self.inl[0]
        o = self.outl[1]
        if i.fluid.is_var or o.fluid.is_var:
            return False
        if sum([1 if data.is_var else 0 for data in [i.p, o.p, o.h]]) > 1:  # [i.p, i.h, o.p, o.h]
            return False
        if i.p.is_var:
            T_i1 = self.DTU_sh.val_SI + o.calc_T()
            i.p.val_SI = p_sat_T(T_i1, i.fluid_data)
            i.p.is_set = True
            i.p.is_var = False
            self.DTU_sh.is_set = False
            return True
        elif o.p.is_var:
            T_o2 = i.calc_T_sat() - self.DTU_sh.val_SI
            o.p.val_SI = p_mix_hT(o.h.val_SI, T_o2, o.fluid_data, o.mixing_rule)
            o.p.is_set = True
            o.p.is_var = False
            self.DTU_sh.is_set = False
            return True
        elif o.h.is_var:
            T_o2 = i.calc_T_sat() - self.DTU_sh.val_SI
            o.h.val_SI = h_mix_pT(o.p.val_SI, T_o2, o.fluid_data, o.mixing_rule)
            o.h.is_set = True
            o.h.is_var = False
            self.DTU_sh.is_set = False
            return True
        else:
            self.DTU_sh.is_set = False
            return True

    def DTU_sh_deriv(self, increment_filter, k):
        i = self.inl[0]
        o = self.outl[1]
        if self.is_variable(o.h, increment_filter):
            self.network.jacobian[k, o.h.J_col] = dT_mix_pdh(o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule, T0=o.T.val_SI)

    def DTU_sh_repair_matrix(self, property_):
        i = self.inl[0]
        o = self.outl[1]
        if property_ == o.h:
            h0 = h_mix_pQ(o.p.val_SI, 0, o.fluid_data)
            h1 = h_mix_pQ(o.p.val_SI, 1, o.fluid_data)
            return abs(self.DTU_sh.val_SI - i.calc_T_sat() + o.calc_T()) / max(o.h.val_SI - h0, h1 - o.h.val_SI)
        else:
            msg = f"variable: {property_.label} is not a valid property in DTU_sh_repair_matrix of {self.__class__.__name__}: {self.label}"
            raise ValueError(msg)

    def DTU_sh_tensor(self, increment_filter, k):
        pass

    def DTL_func(self):
        r"""
        Equation for lower terminal temperature difference.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = DT_{L} - T_{out,1} + T_{in,2}
        """
        i = self.inl[1]
        o = self.outl[0]
        T_i2 = i.calc_T()
        T_o1 = o.calc_T()
        return self.DTL.val_SI - T_o1 + T_i2

    def DTL_variables_columns(self):
        variables_colmns1 = [data.J_col for c in [self.inl[1], self.outl[0]] for data in [c.h] if data.is_var]  # [c.p, c.h]
        variables_colmns1.sort()
        return [variables_colmns1]

    def DTL_take_effect(self):
        pass

    def DTL_solve_isolated(self):
        i = self.inl[1]
        o = self.outl[0]
        if i.fluid.is_var or o.fluid.is_var:
            return False
        if sum([1 if data.is_var else 0 for data in [i.p, i.h, o.p, o.h]]) > 1:  # [i.p, i.h, o.p, o.h]
            return False
        if i.h.is_var:
            T_i2 = o.calc_T() - self.DTL.val_SI
            i.h.val_SI = h_mix_pT(i.p.val_SI, T_i2, i.fluid_data, i.mixing_rule)
            i.h.is_set = True
            i.h.is_var = False
            self.DTL.is_set = False
            return True
        elif i.p.is_var:
            T_i2 = o.calc_T() - self.DTL.val_SI
            i.p.val_SI = p_mix_hT(i.h.val_SI, T_i2, i.fluid_data, i.mixing_rule)
            i.p.is_set = True
            i.p.is_var = False
            self.DTL.is_set = False
            return True
        elif o.p.is_var:
            T_o1 = i.calc_T() + self.DTL.val_SI
            o.p.val_SI = p_mix_hT(o.h.val_SI, T_o1, o.fluid_data, o.mixing_rule)
            o.p.is_set = True
            o.p.is_var = False
            self.DTL.is_set = False
            return True
        elif o.h.is_var:
            T_o1 = i.calc_T() + self.DTL.val_SI
            o.h.val_SI = h_mix_pT(o.p.val_SI, T_o1, o.fluid_data, o.mixing_rule)
            o.h.is_set = True
            o.h.is_var = False
            self.DTL.is_set = False
            return True
        else:
            self.DTL.is_set = False
            return True

    def DTL_func_doc(self, label):
        r"""
        Equation for lower terminal temperature difference.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        latex = r'0 = DT_\mathrm{L} - T_\mathrm{out,1} + T_\mathrm{in,2}'
        return generate_latex_eq(self, latex, label)

    def DTL_deriv(self, increment_filter, k):
        """
        Calculate partial derivates of lower terminal temperature function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        i = self.inl[1]
        o = self.outl[0]
        # if self.is_variable(i.p, increment_filter):
        #     self.network.jacobian[k, i.p.J_col] = dT_mix_dph(i.p.val_SI, i.h.val_SI, i.fluid_data, i.mixing_rule, T0=i.T.val_SI)
        # if self.is_variable(o.p, increment_filter):
        #     self.network.jacobian[k, o.p.J_col] = -dT_mix_dph(o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule, T0=o.T.val_SI)
        if self.is_variable(i.h, increment_filter):
            self.network.jacobian[k, i.h.J_col] = dT_mix_pdh(i.p.val_SI, i.h.val_SI, i.fluid_data, i.mixing_rule, T0=i.T.val_SI)
        if self.is_variable(o.h, increment_filter):
            self.network.jacobian[k, o.h.J_col] = -dT_mix_pdh(o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule, T0=o.T.val_SI)

    def DTL_repair_matrix(self, property_):
        i = self.inl[1]
        o = self.outl[0]
        if property_ == i.h:
            h0 = h_mix_pQ(i.p.val_SI, 0, i.fluid_data)
            h1 = h_mix_pQ(i.p.val_SI, 1, i.fluid_data)
            return abs(self.DTL.val_SI - o.calc_T() + i.calc_T()) / max(i.h.val_SI - h0, h1 - i.h.val_SI)
        elif property_ == o.h:
            h0 = h_mix_pQ(o.p.val_SI, 0, o.fluid_data)
            h1 = h_mix_pQ(o.p.val_SI, 1, o.fluid_data)
            return -abs(self.DTL.val_SI - o.calc_T() + i.calc_T()) / max(o.h.val_SI - h0, h1 - o.h.val_SI)
        else:
            msg = f"variable: {property_.label} is not a valid property in DTL_repair_matrix of {self.__class__.__name__}: {self.label}"
            raise ValueError(msg)

    def DTL_tensor(self, increment_filter, k):
        f = self.DTL_func
        numeric_variables_list = ([('p', self.is_variable(c.p, increment_filter), c, c.p.J_col) for c in [self.inl[1], self.outl[0]]] +
                                  [('h', self.is_variable(c.h, increment_filter), c, c.h.J_col) for c in [self.inl[1], self.outl[0]]])
        self.generate_numerical_tensor(f, k, numeric_variables_list)

    def DTN_min_func(self):
        r"""
        Equation for upper terminal temperature difference.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                ttd_{l} = T_{out,1} - T_{in,2}
                ttd_{u} = T_{in,1} - T_{out,2}
                0 = \text{min}\left(ttd_{u}, ttd_{l}\right)

        """
        i = self.inl[1]
        o = self.outl[0]
        T_i2 = i.calc_T()
        T_o1 = o.calc_T()

        i = self.inl[0]
        o = self.outl[1]
        T_i1 = i.calc_T()
        T_o2 = o.calc_T()

        DTL = T_o1 - T_i2
        DTU = T_i1 - T_o2

        return self.DTN_min.val_SI - min(DTL, DTU)

    def DTN_min_variables_columns(self):
        variabels_colmns1 = [data.J_col for c in self.inl + self.outl for data in [c.h] if data.is_var]  # [c.p, c.h]
        variabels_colmns1.sort()
        return [variabels_colmns1]

    def DTN_min_take_effect(self):
        pass

    def DTN_min_solve_isolated(self):
        return False

    def DTN_min_deriv(self, increment_filter, k):
        """
        Calculate partial derivates of minimum terminal temperature function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        f = self.DTN_min_func
        for c in self.inl + self.outl:
            # if self.is_variable(c.p, increment_filter):
            #     self.jacobian[k, c.p.J_col] = self.numeric_deriv(f, 'p', c)
            if self.is_variable(c.h, increment_filter):
                self.jacobian[k, c.h.J_col] = self.numeric_deriv(f, 'h', c)

    def DTN_min_repair_matrix(self, property):
        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0]
        o2 = self.outl[1]
        if property == i1.h:
            pass
        elif property == o1.h:
            pass
        elif property == i2.h:
            pass
        elif property == o2.h:
            pass
        else:
            msg = f"variable: {property.label} is not a valid property in ttd_min_repair_matrix of {self.__class__.__name__}: {self.label}"
            raise ValueError(msg)

    def DTN_min_tensor(self, increment_filter, k):
        pass

    def calc_dh_max_cold(self):
        r"""Calculate the theoretical maximum enthalpy increase on the cold side

        Returns
        -------
        float
            Maxmium cold side enthalpy increase.

            .. math::

                h\left(p_{out,2}, T_{in,1}\right) - h_{in,2}
        """
        o2 = self.outl[1]
        T_in_hot = self.inl[0].calc_T()
        h_at_T_in_hot = h_mix_pT(
            o2.p.val_SI, T_in_hot, o2.fluid_data, o2.mixing_rule
        )
        return h_at_T_in_hot - self.inl[1].h.val_SI

    def eff_cold_func(self):
        r"""Equation for cold side heat exchanger effectiveness.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = \text{eff}_\text{cold} \cdot
                \left(h\left(p_{out,2}, T_{in,1} \right) - h_{in,2}\right)
                - \left( h_{out,2} - h_{in,2} \right)
        """
        return (
            self.eff_cold.val_SI * self.calc_dh_max_cold()
            - (self.outl[1].h.val_SI - self.inl[1].h.val_SI)
        )

    def eff_cold_variables_columns(self):
        i1 = self.inl[0]
        i2 = self.inl[1]
        o2 = self.outl[1]
        variables_colmns1 = [data.J_col for c in [i1, o2] for data in [c.h] if data.is_var]  # [c.p, c.h]
        if i2.h.is_var:
            variables_colmns1.append(i2.h.J_col)
        variables_colmns1.sort()
        return [variables_colmns1]

    def eff_cold_take_effect(self):
        pass

    def eff_cold_solve_isolated(self):
        return False

    def eff_cold_deriv(self, increment_filter, k):
        """
        Calculate partial derivates of hot side effectiveness function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        f = self.eff_cold_func
        i1 = self.inl[0]
        i2 = self.inl[1]
        o2 = self.outl[1]
        for c in [i1, o2]:
            # if self.is_variable(c.p, increment_filter):
            #     self.network.jacobian[k, c.p.J_col] = self.numeric_deriv(f, 'p', c)
            if self.is_variable(c.h, increment_filter):
                self.network.jacobian[k, c.h.J_col] = self.numeric_deriv(f, 'h', c)
        if self.is_variable(i2.h):
            self.network.jacobian[k, i2.h.J_col] = 1 - self.eff_cold.val_SI

    def eff_cold_tensor(self, increment_filter, k):
        f = self.eff_cold_func
        i1 = self.inl[0]
        i2 = self.inl[1]
        o2 = self.outl[1]
        numeric_variables_list = ([('p', self.is_variable(c.p, increment_filter), c, c.p.J_col) for c in [i1, o2]] +
                                  [('h', self.is_variable(c.h, increment_filter), c, c.h.J_col) for c in [i1, o2]])
        self.generate_numerical_tensor(f, k, numeric_variables_list)

    def calc_dh_max_hot(self):
        r"""Calculate the theoretical maximum enthalpy decrease on the hot side

        Returns
        -------
        float
            Maxmium hot side enthalpy decrease.

            .. math::

                h\left(p_{out,1}, T_{in,2}\right) - h_{in,1}
        """
        o1 = self.outl[0]
        T_in_cold = self.inl[1].calc_T()
        h_at_T_in_cold = h_mix_pT(
            o1.p.val_SI, T_in_cold, o1.fluid_data, o1.mixing_rule
        )
        return h_at_T_in_cold - self.inl[0].h.val_SI

    def eff_hot_func(self):
        r"""Equation for hot side heat exchanger effectiveness.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = \text{eff}_\text{hot} \cdot
                \left(h\left(p_{out,1}, T_{in,2}\right) - h_{in,1}\right)
                - \left( h_{out,1} - h_{in,1}\right)
        """
        return (
            self.eff_hot.val_SI * self.calc_dh_max_hot()
            - (self.outl[0].h.val_SI - self.inl[0].h.val_SI)
        )

    def eff_hot_variables_columns(self):
        i1 = self.inl[0]
        o1 = self.outl[0]
        i2 = self.inl[1]
        variables_colmns1 = [data.J_col for c in [o1, i2] for data in [c.h] if data.is_var]  # [c.p, c.h]
        if i1.h.is_var:
            variables_colmns1.append(i1.h.J_col)
        variables_colmns1.sort()
        return [variables_colmns1]

    def eff_hot_take_effect(self):
        pass

    def eff_hot_solve_isolated(self):
        return False

    def eff_hot_deriv(self, increment_filter, k):
        """
        Calculate partial derivates of hot side effectiveness function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        f = self.eff_hot_func
        i1 = self.inl[0]
        o1 = self.outl[0]
        i2 = self.inl[1]
        if self.is_variable(i1.h):
            self.network.jacobian[k, i1.h.J_col] = 1 - self.eff_hot.val_SI
        for c in [o1, i2]:
            # if self.is_variable(c.p, increment_filter):
            #     self.network.jacobian[k, c.p.J_col] = self.numeric_deriv(f, 'p', c)
            if self.is_variable(c.h, increment_filter):
                self.network.jacobian[k, c.h.J_col] = self.numeric_deriv(f, 'h', c)

    def eff_hot_tensor(self, increment_filter, k):
        f = self.eff_hot_func
        i1 = self.inl[0]
        o1 = self.outl[0]
        i2 = self.inl[1]
        numeric_variables_list = ([('p', self.is_variable(c.p, increment_filter), c, c.p.J_col) for c in [o1, i2]] +
                                  [('h', self.is_variable(c.h, increment_filter), c, c.h.J_col) for c in [o1, i2]])
        self.generate_numerical_tensor(f, k, numeric_variables_list)

    def eff_max_func(self):
        r"""Equation for maximum heat exchanger effectiveness.

        .. note::

            This functions works on what is larger: hot side or cold side
            effectiveness. It may cause numerical issues, if applied, when one
            of both sides' effectiveness is already predetermined, e.g. by
            temperature specifications.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = \text{eff}_\text{max} - \text{max}
                \left(\text{eff}_\text{hot},\text{eff}_\text{cold}\right)
        """
        eff_hot = (
            (self.outl[0].h.val_SI - self.inl[0].h.val_SI)
            / self.calc_dh_max_hot()
        )
        eff_cold = (
            (self.outl[1].h.val_SI - self.inl[1].h.val_SI)
            / self.calc_dh_max_cold()
        )
        return self.eff_max.val_SI - max(eff_hot, eff_cold)

    def eff_max_variables_columns(self):
        variables_colmns1 = [data.J_col for c in self.inl + self.outl for data in [c.h] if data.is_var]  # [c.p, c.h]
        variables_colmns1.sort()
        return [variables_colmns1]

    def eff_max_take_effect(self):
        pass

    def eff_max_solve_isolated(self):
        return False

    def eff_max_deriv(self, increment_filter, k):
        """
        Calculate partial derivates of max effectiveness function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        f = self.eff_max_func
        for c in self.inl + self.outl:
            # if self.is_variable(c.p, increment_filter):
            #     self.network.jacobian[k, c.p.J_col] = self.numeric_deriv(f, 'p', c)
            if self.is_variable(c.h, increment_filter):
                self.network.jacobian[k, c.h.J_col] = self.numeric_deriv(f, 'h', c)

    def eff_max_tensor(self, increment_filter, k):
        f = self.eff_max_func
        numeric_variables_list = ([('p', self.is_variable(c.p, increment_filter), c, c.p.J_col) for c in self.inl + self.outl] +
                                  [('h', self.is_variable(c.h, increment_filter), c, c.h.J_col) for c in self.inl + self.outl])
        self.generate_numerical_tensor(f, k, numeric_variables_list)

    def bus_func(self, bus):
        r"""
        Calculate the value of the bus function.

        Parameters
        ----------
        bus : tespy.connections.bus.Bus
            TESPy bus object.

        Returns
        -------
        val : float
            Value of energy transfer :math:`\dot{E}`. This value is passed to
            :py:meth:`Aurora.components.component.Component.calc_bus_value`
            for value manipulation according to the specified characteristic
            line of the bus.

            .. math::

                \dot{E} = \dot{m}_{in,1} \cdot \left(
                h_{out,1} - h_{in,1} \right)
        """
        return self.inl[0].m.val_SI * (
            self.outl[0].h.val_SI - self.inl[0].h.val_SI
        )

    def bus_variables_columns(self, bus):
        variables_colmns1 = [data.J_col for c in [self.inl[0]] for data in [c.m, c.h] if data.is_var]
        if self.outl[0].h.is_var:
            variables_colmns1.append(self.outl[0].h.J_col)
        variables_colmns1.sort()
        return [variables_colmns1]

    def bus_func_doc(self, bus):
        r"""
        Return LaTeX string of the bus function.

        Parameters
        ----------
        bus : tespy.connections.bus.Bus
            TESPy bus object.

        Returns
        -------
        latex : str
            LaTeX string of bus function.
        """
        return (
            r'\dot{m}_\mathrm{in,1} \cdot \left(h_\mathrm{out,1} - '
            r'h_\mathrm{in,1} \right)')

    def bus_deriv(self, bus, increment_filter, k):
        r"""
        Calculate partial derivatives of the bus function.

        Parameters
        ----------
        bus : tespy.connections.bus.Bus
            TESPy bus object.

        Returns
        -------
        deriv : ndarray
            Matrix of partial derivatives.
        """
        f = self.calc_bus_value
        if self.inl[0].m.is_var:
            self.network.jacobian[k, self.inl[0].m.J_col] -= self.numeric_deriv(f, 'm', self.inl[0], bus=bus)

        if self.inl[0].h.is_var:
            self.network.jacobian[k, self.inl[0].h.J_col] -= self.numeric_deriv(f, 'h', self.inl[0], bus=bus)

        if self.outl[0].h.is_var:
            self.network.jacobian[k, self.outl[0].h.J_col] -= self.numeric_deriv(f, 'h', self.outl[0], bus=bus)

    def bus_tensor(self, bus, increment_filter, k):
        f = self.calc_bus_value
        numeric_variables_list = (
                    [('m', self.is_variable(c.m, increment_filter), c, c.m.J_col) for c in [self.inl[0]]] +
                    [('h', self.is_variable(c.h, increment_filter), c, c.h.J_col) for c in [self.inl[0], self.outl[0]]])
        self.generate_numerical_bus_tensor(f, k, numeric_variables_list, bus=bus)

    def initialise_source(self, c, key):
        r"""
        Return a starting value for pressure and enthalpy at outlet.

        Parameters
        ----------
        c : tespy.connections.connection.Connection
            Connection to perform initialisation on.

        key : str
            Fluid property to retrieve.

        Returns
        -------
        val : float
            Starting value for pressure/enthalpy in SI units.

            .. math::

                val = \begin{cases}
                4 \cdot 10^5 & \text{key = 'p'}\\
                h\left(p, 200 \text{K} \right) & \text{key = 'h' at outlet 1}\\
                h\left(p, 250 \text{K} \right) & \text{key = 'h' at outlet 2}
                \end{cases}
        """
        if key == 'p':
            return 50e5
        elif key == 'h':
            if c.source_id == 'out1':
                T = 200 + 273.15
            else:
                T = 250 + 273.15
            return h_mix_pT(c.p.val_SI, T, c.fluid_data, c.mixing_rule)

    def initialise_target(self, c, key):
        r"""
        Return a starting value for pressure and enthalpy at inlet.

        Parameters
        ----------
        c : tespy.connections.connection.Connection
            Connection to perform initialisation on.

        key : str
            Fluid property to retrieve.

        Returns
        -------
        val : float
            Starting value for pressure/enthalpy in SI units.

            .. math::

                val = \begin{cases}
                4 \cdot 10^5 & \text{key = 'p'}\\
                h\left(p, 300 \text{K} \right) & \text{key = 'h' at inlet 1}\\
                h\left(p, 220 \text{K} \right) & \text{key = 'h' at outlet 2}
                \end{cases}
        """
        if key == 'p':
            return 50e5
        elif key == 'h':
            if c.target_id == 'in1':
                T = 300 + 273.15
            else:
                T = 220 + 273.15
            return h_mix_pT(c.p.val_SI, T, c.fluid_data, c.mixing_rule)
    
    def boundary_check(self):
        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0]
        o2 = self.outl[1]

    def correct_massflow_enthalpy(self):
        """
        Correct mass flow and enthalpy in calculation iteration.
        :return:
        """
        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0]
        o2 = self.outl[1]
        if i1.h.is_var:
            enthalpy_i1 = self.network.variables_vector[i1.h.J_col]
        else:
            enthalpy_i1 = i1.h.val_SI / ps['h']['scale']
        if i2.h.is_var:
            enthalpy_i2 = self.network.variables_vector[i2.h.J_col]
        else:
            enthalpy_i2 = i2.h.val_SI / ps['h']['scale']
        if o1.h.is_var:
            enthalpy_o1 = self.network.variables_vector[o1.h.J_col]
        else:
            enthalpy_o1 = o1.h.val_SI / ps['h']['scale']
        if o2.h.is_var:
            enthalpy_o2 = self.network.variables_vector[o2.h.J_col]
        else:
            enthalpy_o2 = o2.h.val_SI / ps['h']['scale']
        # inequation constraints of massflow、enthalpy in hot side
        if i1.m.is_var:
            massflow_i1 = self.network.variables_vector[i1.m.J_col]
            if massflow_i1 < (enthalpy_o1 - enthalpy_i1):
                msg = (f"The hot side mass flow: {i1.m.label}: {massflow_i1 * self.network.variables_scale[i1.m.J_col]} {i1.m.SI_unit} "
                       f"adjust to {(enthalpy_o1 - enthalpy_i1 + 1e0) * self.network.variables_scale[i1.m.J_col]} "
                       f"antique enthalpy difference: < {enthalpy_o1 * ps['h']['scale']} - {enthalpy_i1 * ps['h']['scale']} > ({fpd['h']['SI_unit']})"
                       f"outlet: {o1.label} -- inlet: {i1.label} of hot side in {self.__class__.__name__}: {self.label}")
                logger.debug(msg)
                self.network.variables_vector[i1.m.J_col] = (enthalpy_o1 - enthalpy_i1) + 1e0
        if i2.m.is_var:
            massflow_i2 = self.network.variables_vector[i2.m.J_col]
            if massflow_i2 < (enthalpy_i2 - enthalpy_o2):
                msg = (f"The cold side mass flow: {i2.m.label}: {massflow_i2 * self.network.variables_scale[i2.m.J_col]} {i2.m.SI_unit} "
                       f"adjust to {(enthalpy_i2 - enthalpy_o2 + 1e0) * self.network.variables_scale[i2.m.J_col]} "
                       f"antique enthalpy difference: < {enthalpy_i2 * ps['h']['scale']} - {enthalpy_o2 * ps['h']['scale']} > ({fpd['h']['SI_unit']})"
                       f"inlet: {i2.label} -- outlet: {o2.label} of cold side in {self.__class__.__name__}: {self.label}")
                logger.debug(msg)
                self.network.variables_vector[i2.m.J_col] = (enthalpy_i2 - enthalpy_o2) + 1e0
        pass

    def constraints_h_generate(self):
        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0]
        o2 = self.outl[1]
        constraints = []
        hot_h_index = []
        cold_h_index = []
        hot_T_index = []
        cold_T_index = []
        # generate index of constraints
        if i1.h.is_var:
            hot_h_index.append(i1.h.J_col)
            hot_T_index.append(i1.h.J_col)
        if o1.h.is_var:
            hot_h_index.append(o1.h.J_col)
            cold_T_index.append(o1.h.J_col)
        if i2.h.is_var:
            cold_h_index.append(i2.h.J_col)
            cold_T_index.append(i2.h.J_col)
        if o2.h.is_var:
            cold_h_index.append(o2.h.J_col)
            hot_T_index.append(o2.h.J_col)
        if i1.p.is_var:
            hot_T_index.append(i1.p.J_col)
        if o1.p.is_var:
            cold_T_index.append(o1.p.J_col)
        if i2.p.is_var:
            cold_T_index.append(i2.p.J_col)
        if o2.p.is_var:
            hot_T_index.append(o2.p.J_col)

        # hot side enthalpy constraints
        def hot_enthalpy_constraints(variables):
            if i1.h.is_var and o1.h.is_var:
                return variables[i1.h.J_col] - variables[o1.h.J_col]
            elif i1.h.is_var and not o1.h.is_var:
                return variables[i1.h.J_col] - o1.h.val_SI / self.network.variables_scale[i1.h.J_col]
            elif not i1.h.is_var and o1.h.is_var:
                return i1.h.val_SI / self.network.variables_scale[o1.h.J_col] - variables[o1.h.J_col]
        def hot_enthalpy_constraints_jacobin(variables):
            jac = np.zeros_like(variables)
            if i1.h.is_var:
                jac[i1.h.J_col] = 1
            if o1.h.is_var:
                jac[o1.h.J_col] = -1

        # add hot enthalpy constraints
        if i1.h.is_var or o1.h.is_var:
            constraints += [
                {'type': 'ineq', 'fun': hot_enthalpy_constraints, 'jac': hot_enthalpy_constraints_jacobin,
                 'jac_indices': hot_h_index,
                 'k': 10, 'epsilon': 1, 'bounds': (-ERR, np.inf), 'info': f"hot_enthalpy_constraints of {self.label}"}]

        # cold side enthalpy constraints
        def cold_enthalpy_constraints(variables):
            if i2.h.is_var and o2.h.is_var:
                return variables[o2.h.J_col] - variables[i2.h.J_col]
            elif not i2.h.is_var and o2.h.is_var:
                return variables[o2.h.J_col] - i2.h.val_SI / self.network.variables_scale[o2.h.J_col]
            elif i2.h.is_var and not o2.h.is_var:
                return o2.h.val_SI / self.network.variables_scale[i2.h.J_col] - variables[i2.h.J_col]
        def cold_enthalpy_constraints_jacobin(variables):
            jac = np.zeros_like(variables)
            if i2.h.is_var:
                jac[i2.h.J_col] = -1
            if o2.h.is_var:
                jac[o2.h.J_col] = 1

        # add cold enthalpy constraints
        if i2.h.is_var or o2.h.is_var:
            constraints += [
                {'type': 'ineq', 'fun': cold_enthalpy_constraints, 'jac': cold_enthalpy_constraints_jacobin,
                 'jac_indices': cold_h_index,
                 'k': 10, 'epsilon': 1, 'bounds': (-ERR, np.inf), 'info': f"cold_enthalpy_constraints of {self.label}"}]

        # hot temperature side constraints
        def hot_temperature_constraints(variables):
            if i1.p.is_var:
                p1_val_SI = variables[i1.p.J_col] * self.network.variables_scale[i1.p.J_col]
            else:
                p1_val_SI = i1.p.val_SI
            if i1.h.is_var:
                h1_val_SI = variables[i1.h.J_col] * self.network.variables_scale[i1.h.J_col]
            else:
                h1_val_SI = i1.h.val_SI
            # cold side
            if o2.p.is_var:
                p2_val_SI = variables[o2.p.J_col] * self.network.variables_scale[o2.p.J_col]
            else:
                p2_val_SI = o2.p.val_SI
            if o2.h.is_var:
                h2_val_SI = variables[o2.h.J_col] * self.network.variables_scale[o2.h.J_col]
            else:
                h2_val_SI = o2.h.val_SI

            return ((T_mix_ph(p1_val_SI, h1_val_SI, i1.fluid_data, i1.mixing_rule)
                    - T_mix_ph(p2_val_SI, h2_val_SI, o2.fluid_data, o2.mixing_rule))
                    / ps['T']['scale'])
        def hot_temperature_constraints_jacobin(variables):
            jac = np.zeros_like(variables)
            if i1.p.is_var:
                jac[i1.p.J_col] = -1
            if i1.h.is_var:
                jac[i1.h.J_col] = 1
            if o2.p.is_var:
                jac[o2.p.J_col] = 1
            if o2.h.is_var:
                jac[o2.h.J_col] = 1

        # add hot temperature constraints
        if i1.p.is_var or i1.h.is_var or o2.p.is_var or o2.h.is_var:
            constraints += [
                {'type': 'ineq', 'fun': hot_temperature_constraints, 'jac': hot_temperature_constraints_jacobin,
                 'jac_indices': hot_T_index,
                 'k': 10, 'epsilon': 1, 'bounds': (-ERR, np.inf),
                 'info': f"hot_temperature_constraints of {self.label}"}
            ]
        # cold temperature side constraints
        def cold_temperature_constraints(variables):
            if o1.p.is_var:
                p1_val_SI = variables[o1.p.J_col] * self.network.variables_scale[o1.p.J_col]
            else:
                p1_val_SI = o1.p.val_SI
            if o1.h.is_var:
                h1_val_SI = variables[o1.h.J_col] * self.network.variables_scale[o1.h.J_col]
            else:
                h1_val_SI = o1.h.val_SI
            # cold side
            if i2.p.is_var:
                p2_val_SI = variables[i2.p.J_col] * self.network.variables_scale[i2.p.J_col]
            else:
                p2_val_SI = i2.p.val_SI
            if i2.h.is_var:
                h2_val_SI = variables[i2.h.J_col] * self.network.variables_scale[i2.h.J_col]
            else:
                h2_val_SI = i2.h.val_SI

            return ((T_mix_ph(p1_val_SI, h1_val_SI, o1.fluid_data, o1.mixing_rule)
                    - T_mix_ph(p2_val_SI, h2_val_SI, i2.fluid_data, i2.mixing_rule))
                    / ps['T']['scale'])
        def cold_temperature_constraints_jacobin(variables):
            jac = np.zeros_like(variables)
            if o1.p.is_var:
                jac[o1.p.J_col] = -1
            if o1.h.is_var:
                jac[o1.h.J_col] = 1
            if i2.p.is_var:
                jac[i2.p.J_col] = 1
            if i2.h.is_var:
                jac[i2.h.J_col] = 1

        # add cold temperature constraints
        if o1.p.is_var or o1.h.is_var or i2.p.is_var or i2.h.is_var:
            constraints += [
                {'type': 'ineq', 'fun': cold_temperature_constraints, 'jac': cold_temperature_constraints_jacobin,
                 'jac_indices': cold_T_index,
                 'k': 10, 'epsilon': 1, 'bounds': (-ERR, np.inf),
                 'info': f"cold_temperature_constraints of {self.label}"}
            ]
        # add all constraints of component to constraints of solver
        self.network.constraints.extend(constraints)

    def correct_attracting_basin_path_(self):
        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0]
        o2 = self.outl[1]
        enthalpy_value_list_ = []
        for conn in [i1, i2, o1, o2]:
            if conn.h.is_var:
                enthalpy_value_list_.append(self.network.variables_vector[conn.h.J_col] * self.network.variables_scale[conn.h.J_col])
            else:
                enthalpy_value_list_.append(conn.h.val_SI)
        # check
        if enthalpy_value_list_[0] < enthalpy_value_list_[2]:
            hot_delta_enthalpy = enthalpy_value_list_[2] - enthalpy_value_list_[0]
            if i1.h.is_var:
                self.network.variables_vector[i1.h.J_col] += hot_delta_enthalpy / self.network.variables_scale[i1.h.J_col]
            if o1.h.is_var:
                self.network.variables_vector[o1.h.J_col] -= hot_delta_enthalpy / self.network.variables_scale[o1.h.J_col]
        if enthalpy_value_list_[3] < enthalpy_value_list_[1]:
            cold_delta_enthalpy = enthalpy_value_list_[1] - enthalpy_value_list_[3]
            if i2.h.is_var:
                self.network.variables_vector[i2.h.J_col] -= cold_delta_enthalpy / self.network.variables_scale[i2.h.J_col]
            if o2.h.is_var:
                self.network.variables_vector[o2.h.J_col] += cold_delta_enthalpy / self.network.variables_scale[o2.h.J_col]

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        # component parameters
        self.Q.val_SI = self.inl[0].m.val_SI * (
            self.outl[0].h.val_SI - self.inl[0].h.val_SI
        )
        self.DTU.val_SI = self.inl[0].T.val_SI - self.outl[1].T.val_SI
        self.DTL.val_SI = self.outl[0].T.val_SI - self.inl[1].T.val_SI
        self.DTN_min.val_SI = min(self.DTU.val_SI, self.DTL.val_SI)
        self.DTM.val_SI = self.calc_DTM()
        # pr and zeta
        for i in range(2):
            self.get_attr(f'pr{i + 1}').val_SI = (
                    self.outl[i].p.val_SI / self.inl[i].p.val_SI
            )
            self.get_attr(f'zeta{i + 1}').val_SI = self.calc_zeta(
                self.inl[i], self.outl[i]
            )
            self.get_attr(f'dp{i + 1}').val_SI = (
                    self.inl[i].p.val_SI - self.outl[i].p.val_SI
            )
        # kA and logarithmic temperature difference
        if self.DTU.val_SI < 0 or self.DTL.val_SI < 0:
            self.DT_log.val_SI = np.nan
        elif self.DTL.val_SI == self.DTU.val_SI:
            self.DT_log.val_SI = self.DTL.val_SI
        else:
            self.DT_log.val_SI = (
                (self.DTL.val_SI - self.DTU.val_SI)
                / math.log(self.DTL.val_SI / self.DTU.val_SI)
            )
        self.kA.val_SI = -self.Q.val_SI / self.DT_log.val_SI
        # differ kA
        if self.network.converged:
            self.KDTA.val_SI = self.calc_KDTA()
            self.fA.val_SI = self.KDTA.val_SI * (1 / self.hf1.val_SI + 1 / self.hf2.val_SI)
        else:
            self.KDTA.val_SI = np.nan
            self.fA.val_SI = np.nan
            # heat exchanger efficiencies
        try:
            self.eff_hot.val_SI = (
                (self.outl[0].h.val_SI - self.inl[0].h.val_SI)
                / self.calc_dh_max_hot()
            )
        except ValueError:
            self.eff_hot.val_SI = np.nan
            msg = (
                "Cannot calculate heat exchanger hot side effectiveness "
                "because cold side inlet temperature is out of bounds for hot "
                "side fluid."
            )
            logger.warning(msg)
        try:
            self.eff_cold.val_SI = (
                (self.outl[1].h.val_SI - self.inl[1].h.val_SI)
                / self.calc_dh_max_cold()
            )
        except ValueError:
            self.eff_cold.val_SI = np.nan
            msg = (
                "Cannot calculate heat exchanger cold side effectiveness "
                "because hot side inlet temperature is out of bounds for cold "
                "side fluid."
            )
            logger.warning(msg)
        self.eff_max.val_SI = max(self.eff_hot.val_SI, self.eff_cold.val_SI)

    def calc_nodes_properties(self):
        for side in self.nodes.index:
            for col in range(self.nodes_num):
                node = self.nodes.loc[side, col]
                node.n.val = (self.KDTA_fit() * dT_mix_pdh(node.p.val_SI,
                                                            node.h.val_SI,
                                                            node.fluid_reference.fluid_data,
                                                            node.fluid_reference.mixing_rule,
                                                            T0=node.T.val_SI)) / node.m.val_SI + 2
        # convert main properties
        super().calc_nodes_properties()

    def entropy_balance(self):
        r"""
        Calculate entropy balance of a heat exchanger.

        The allocation of the entropy streams due to heat exchanged and due to
        irreversibility is performed by solving for T on both sides of the heat
        exchanger:

        .. math::

            h_\mathrm{out} - h_\mathrm{in} = \int_\mathrm{in}^\mathrm{out} v
            \cdot dp - \int_\mathrm{in}^\mathrm{out} T \cdot ds

        As solving :math:`\int_\mathrm{in}^\mathrm{out} v \cdot dp` for non
        isobaric processes would require perfect process knowledge (the path)
        on how specific volume and pressure change throught the component, the
        heat transfer is splitted into three separate virtual processes for
        both sides:

        - in->in*: decrease pressure to
          :math:`p_\mathrm{in*}=p_\mathrm{in}\cdot\sqrt{\frac{p_\mathrm{out}}{p_\mathrm{in}}}`
          without changing enthalpy.
        - in*->out* transfer heat without changing pressure.
          :math:`h_\mathrm{out*}-h_\mathrm{in*}=h_\mathrm{out}-h_\mathrm{in}`
        - out*->out decrease pressure to outlet pressure :math:`p_\mathrm{out}`
          without changing enthalpy.

        Note
        ----
        The entropy balance makes the follwing parameter available:

        .. math::

            \text{S\_Q1}=\dot{m} \cdot \left(s_\mathrm{out*,1}-s_\mathrm{in*,1}
            \right)\\
            \text{S\_Q2}=\dot{m} \cdot \left(s_\mathrm{out*,2}-s_\mathrm{in*,2}
            \right)\\
            \text{S\_Qirr}=\text{S\_Q2} - \text{S\_Q1}\\
            \text{S\_irr1}=\dot{m} \cdot \left(s_\mathrm{out,1}-s_\mathrm{in,1}
            \right) - \text{S\_Q1}\\
            \text{S\_irr2}=\dot{m} \cdot \left(s_\mathrm{out,2}-s_\mathrm{in,2}
            \right) - \text{S\_Q2}\\
            \text{S\_irr}=\sum \dot{S}_\mathrm{irr}\\
            \text{T\_mQ1}=\frac{\dot{Q}}{\text{S\_Q1}}\\
            \text{T\_mQ2}=\frac{\dot{Q}}{\text{S\_Q2}}
        """
        self.S_irr = 0
        for i in range(2):
            inl = self.inl[i]
            out = self.outl[i]
            p_star = inl.p.val_SI * (
                self.get_attr('pr' + str(i + 1)).val_SI) ** 0.5
            s_i_star = s_mix_ph(
                p_star, inl.h.val_SI, inl.fluid_data, inl.mixing_rule,
                T0=inl.T.val_SI
            )
            s_o_star = s_mix_ph(
                p_star, out.h.val_SI, out.fluid_data, out.mixing_rule,
                T0=out.T.val_SI
            )

            setattr(
                self, 'S_Q' + str(i + 1),
                inl.m.val_SI * (s_o_star - s_i_star)
            )
            S_Q = self.get_attr('S_Q' + str(i + 1))
            setattr(
                self, 'S_irr' + str(i + 1),
                inl.m.val_SI * (out.s.val_SI - inl.s.val_SI) - S_Q
            )
            setattr(
                self, 'T_mQ' + str(i + 1),
                inl.m.val_SI * (out.h.val_SI - inl.h.val_SI) / S_Q
            )

            self.S_irr += self.get_attr('S_irr' + str(i + 1))

        self.S_irr += self.S_Q1 + self.S_Q2

    def exergy_balance(self, T0):
        r"""
        Calculate exergy balance of a heat exchanger.

        Parameters
        ----------
        T0 : float
            Ambient temperature T0 / K.

        Note
        ----
        .. math::

            \dot{E}_\mathrm{P} =
            \begin{cases}
            \dot{E}_\mathrm{out,2}^\mathrm{T} -
            \dot{E}_\mathrm{in,2}^\mathrm{T}
            & T_\mathrm{in,1}, T_\mathrm{in,2}, T_\mathrm{out,1},
            T_\mathrm{out,2} > T_0\\
            \dot{E}_\mathrm{out,1}^\mathrm{T} -
            \dot{E}_\mathrm{in,1}^\mathrm{T}
            & T_0 \geq  T_\mathrm{in,1}, T_\mathrm{in,2}, T_\mathrm{out,1},
            T_\mathrm{out,2}\\
            \dot{E}_\mathrm{out,1}^\mathrm{T} +
            \dot{E}_\mathrm{out,2}^\mathrm{T}
            & T_\mathrm{in,1}, T_\mathrm{out,2} > T_0 \geq
            T_\mathrm{in,2}, T_\mathrm{out,1}\\
            \dot{E}_\mathrm{out,1}^\mathrm{T}
            & T_\mathrm{in,1} > T_0 \geq
            T_\mathrm{in,2}, T_\mathrm{out,1}, T_\mathrm{out,2}\\
            \text{not defined (nan)}
            & T_\mathrm{in,1}, T_\mathrm{out,1} > T_0 \geq
            T_\mathrm{in,2}, T_\mathrm{out,2}\\
            \dot{E}_\mathrm{out,2}^\mathrm{T}
            & T_\mathrm{in,1}, T_\mathrm{out,1},
            T_\mathrm{out,2} \geq T_0 > T_\mathrm{in,2}\\
            \end{cases}

            \dot{E}_\mathrm{F} =
            \begin{cases}
            \dot{E}_\mathrm{in,1}^\mathrm{PH} -
            \dot{E}_\mathrm{out,1}^\mathrm{PH} +
            \dot{E}_\mathrm{in,2}^\mathrm{M} -
            \dot{E}_\mathrm{out,2}^\mathrm{M}
            & T_\mathrm{in,1}, T_\mathrm{in,2}, T_\mathrm{out,1},
            T_\mathrm{out,2} > T_0\\
            \dot{E}_\mathrm{in,2}^\mathrm{PH} -
            \dot{E}_\mathrm{out,2}^\mathrm{PH} +
            \dot{E}_\mathrm{in,1}^\mathrm{M} -
            \dot{E}_\mathrm{out,1}^\mathrm{M}
            & T_0 \geq T_\mathrm{in,1}, T_\mathrm{in,2}, T_\mathrm{out,1},
            T_\mathrm{out,2}\\
            \dot{E}_\mathrm{in,1}^\mathrm{PH} +
            \dot{E}_\mathrm{in,2}^\mathrm{PH} -
            \dot{E}_\mathrm{out,1}^\mathrm{M} -
            \dot{E}_\mathrm{out,2}^\mathrm{M}
            & T_\mathrm{in,1}, T_\mathrm{out,2} > T_0 \geq
            T_\mathrm{in,2}, T_\mathrm{out,1}\\
            \dot{E}_\mathrm{in,1}^\mathrm{PH} +
            \dot{E}_\mathrm{in,2}^\mathrm{PH} -
            \dot{E}_\mathrm{out,2}^\mathrm{PH} -
            \dot{E}_\mathrm{out,1}^\mathrm{M}
            & T_\mathrm{in,1} > T_0 \geq
            T_\mathrm{in,2}, T_\mathrm{out,1}, T_\mathrm{out,2}\\
            \dot{E}_\mathrm{in,1}^\mathrm{PH} -
            \dot{E}_\mathrm{out,1}^\mathrm{PH} +
            \dot{E}_\mathrm{in,2}^\mathrm{PH} -
            \dot{E}_\mathrm{out,2}^\mathrm{PH}
            & T_\mathrm{in,1}, T_\mathrm{out,1} > T_0 \geq
            T_\mathrm{in,2}, T_\mathrm{out,2}\\
            \dot{E}_\mathrm{in,1}^\mathrm{PH} -
            \dot{E}_\mathrm{out,1}^\mathrm{PH} +
            \dot{E}_\mathrm{in,2}^\mathrm{PH} -
            \dot{E}_\mathrm{out,2}^\mathrm{M}
            & T_\mathrm{in,1}, T_\mathrm{out,1},
            T_\mathrm{out,2} \geq T_0 > T_\mathrm{in,2}\\
            \end{cases}
        """
        if all([c.T.val_SI > T0 for c in self.inl + self.outl]):
            self.E_P = self.outl[1].Ex_therm - self.inl[1].Ex_therm  # cold side
            self.E_F = self.inl[0].Ex_physical - self.outl[0].Ex_physical + (
                self.inl[1].Ex_mech - self.outl[1].Ex_mech)
        elif all([c.T.val_SI <= T0 for c in self.inl + self.outl]):
            self.E_P = self.outl[0].Ex_therm - self.inl[0].Ex_therm  # hot side
            self.E_F = self.inl[1].Ex_physical - self.outl[1].Ex_physical + (
                self.inl[0].Ex_mech - self.outl[0].Ex_mech)
        elif (self.inl[0].T.val_SI > T0 and self.outl[1].T.val_SI > T0 and
              self.outl[0].T.val_SI <= T0 and self.inl[1].T.val_SI <= T0):
            self.E_P = self.outl[0].Ex_therm + self.outl[1].Ex_therm
            self.E_F = self.inl[0].Ex_physical + self.inl[1].Ex_physical - (
                self.outl[0].Ex_mech + self.outl[1].Ex_mech)
        elif (self.inl[0].T.val_SI > T0 and self.inl[1].T.val_SI <= T0 and
              self.outl[0].T.val_SI <= T0 and self.outl[1].T.val_SI <= T0):
            self.E_P = self.outl[0].Ex_therm
            self.E_F = self.inl[0].Ex_physical + self.inl[1].Ex_physical - (
                self.outl[1].Ex_physical + self.outl[0].Ex_mech)
        elif (self.inl[0].T.val_SI > T0 and self.outl[0].T.val_SI > T0 and
              self.inl[1].T.val_SI <= T0 and self.outl[1].T.val_SI <= T0):
            self.E_P = np.nan
            self.E_F = self.inl[0].Ex_physical - self.outl[0].Ex_physical + (
                self.inl[1].Ex_physical - self.outl[1].Ex_physical)
        else:  # only self.inl[1].T.val_SI <= T0
            self.E_P = self.outl[1].Ex_therm
            self.E_F = self.inl[0].Ex_physical - self.outl[0].Ex_physical + (
                self.inl[1].Ex_physical - self.outl[1].Ex_mech)

        self.E_bus = {"chemical": np.nan, "physical": np.nan, "massless": np.nan}
        if np.isnan(self.E_P):
            self.E_D = self.E_F
        else:
            self.E_D = self.E_F - self.E_P
        self.epsilon = self._calc_epsilon()

    def get_plotting_data(self):
        """Generate a dictionary containing FluProDia plotting information.

        Returns
        -------
        data : dict
            A nested dictionary containing the keywords required by the
            :code:`calc_individual_isoline` method of the
            :code:`FluidPropertyDiagram` class. First level keys are the
            connection index ('in1' -> 'out1', therefore :code:`1` etc.).
        """
        return {
            i + 1: {
                'isoline_property': 'p',
                'isoline_value': self.inl[i].p.val,
                'isoline_value_end': self.outl[i].p.val,
                'starting_point_property': 'v',
                'starting_point_value': self.inl[i].vol.val,
                'ending_point_property': 'v',
                'ending_point_value': self.outl[i].vol.val
            } for i in range(2)}
