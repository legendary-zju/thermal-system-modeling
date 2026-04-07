# -*- coding: utf-8

"""Module of class BoilerSimple.
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
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
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
class BoilerSimple(FluidComponent):
    r"""
    Class for virtual heat source.

    **Optional Equations**

    - :py:meth:`AURORA.components.heat_exchangers.boiler_simple.BoilerSimple.Q_func`
    - :py:meth:`AURORA.components.heat_exchangers.boiler_simple.BoilerSimple.T_out_func`
    - any side :py:meth:`AURORA.components.component.Component.pr_func`
    - any side :py:meth:`AURORA.components.component.Component.dp_func`

    Inlets/Outlets

    - ini (index i: any parallel side)
    - outi (index i: any parallel side)

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

    pri : float, dict, :code:`"var"`
        Outlet to inlet pressure ratio at ith side, :math:`pr/i`.

    dpi : float, dict, :code:`"var"`
        Inlet to outlet pressure delta at ith side.

    zetai : float, dict, :code:`"var"`
        Geometry independent friction coefficient at ith side,
        :math:`\frac{\zeta}{D^4}/\frac{1}{\text{m}^4}`.

    T_out : float, dict, :code:`"var"`
        Temperature of outlet at all side.

    """

    @staticmethod
    def component():
        return 'boiler simple'

    def set_properties(self):
        return {
            'num_side': dc_simple(
                val=1,
            ),
        }

    def get_parameters(self):
        data = {
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
                var_scale=ps['m']['scale'] * ps['h']['scale']
            ),
            'auto_distribute': dc_simple(
                val=False,
                func=self.auto_distribute_func,
                variables_columns=self.auto_distribute_variables_columns,
                solve_isolated=self.auto_distribute_solve_isolated,
                deriv=self.auto_distribute_deriv,
                tensor=self.auto_distribute_tensor,
                latex=self.auto_distribute_func_doc,
                num_eq=self.num_side.val,
                scale=ps['m']['scale'] * ps['h']['scale'],
            ),
        }
        for i in range(self.num_side.val):
            data.update({
                f'T_out{i + 1}': dc_cp(
                    min_val=0,
                    func=self.T_out_func,
                    variables_columns=self.T_out_variables_columns,
                    solve_isolated=self.T_out_solve_isolated,
                    deriv=self.T_out_deriv,
                    tensor=self.T_out_tensor,
                    latex=self.T_out_func_doc,
                    num_eq=1,
                    func_params={"outconn": i},
                    property_data=fpd['T'],
                    SI_unit=fpd['T']['SI_unit'],
                    scale=ps['DT']['scale'],
                    var_scale=ps['T']['scale']
                ),
                f'pr{i + 1}': dc_cp(
                    min_val=0,
                    func=self.pr_func,
                    variables_columns=self.pr_variables_columns,
                    solve_isolated=self.pr_solve_isolated,
                    deriv=self.pr_deriv,
                    tensor=self.pr_tensor,
                    latex=self.pr_func_doc,
                    num_eq=1,
                    func_params={"inconn": i, "outconn": i, "pr": f"pr{i + 1}"},
                    property_data=cpd['ratio'],
                    SI_unit=cpd['ratio']['SI_unit'],
                    scale=ps['p']['scale'],
                    var_scale=ps['pr']['scale']
                ),
                f'dp{i + 1}': dc_cp(
                    min_val=0,
                    deriv=self.dp_deriv,
                    variables_columns=self.dp_variables_columns,
                    solve_isolated=self.dp_solve_isolated,
                    func=self.dp_func,
                    tensor=self.dp_tensor,
                    num_eq=1,
                    func_params={"inconn": i, "outconn": i, "dp": f"dp{i + 1}"},
                    property_data=cpd['dp'],
                    SI_unit=cpd['dp']['SI_unit'],
                    scale=ps['p']['scale'],
                    var_scale=ps['p']['scale']
                ),
                f'pr{i + 1}_fit': dc_fit(
                    rule='constant',
                    constant=self.pr_constant_func_,
                    default=self.pr_default_func_,
                ),
                f'dp{i + 1}_fit': dc_fit(
                    rule='constant',
                    constant=self.dp_constant_func_,
                    default=self.dp_default_func_,
                ),
            })
        return data

    def inlets(self):
        return ['in' + str(i + 1) for i in range(self.num_side.val)]

    def outlets(self):
        return ['out' + str(i + 1) for i in range(self.num_side.val)]

    def spread_forward_pressure_values(self, inconn):
        conn_idx = self.inl.index(inconn)  # the index of branch
        outconn = self.outl[conn_idx]  # outconn: object of outlet connection.
        if inconn.p.is_set and not outconn.p.is_set and (
                (getattr(self, f'pr{conn_idx + 1}').is_set and getattr(self, f'pr{conn_idx + 1}_fit').rule in ['constant', 'static'])
                or (getattr(self, f'dp{conn_idx + 1}').is_set and getattr(self, f'dp{conn_idx + 1}_fit').rule in ['constant', 'static'])):
            if getattr(self, f'pr{conn_idx + 1}').is_set:
                outconn.p.val_SI = inconn.p.val_SI * getattr(self, f'pr{conn_idx + 1}').val_SI
                outconn.p.is_set = True
                outconn.p.is_var = False
                getattr(self, f'pr{conn_idx + 1}').is_set = False
            elif getattr(self, f'dp{conn_idx + 1}').is_set:
                outconn.p.val_SI = inconn.p.val_SI - getattr(self, f'dp{conn_idx + 1}').val_SI
                outconn.p.is_set = True
                outconn.p.is_var = False
                getattr(self, f'dp{conn_idx + 1}').is_set = False
            if outconn not in self.network.connections_spread_pressure_container:
                self.network.connections_spread_pressure_container.append(outconn)
                outconn.target.spread_forward_pressure_values(outconn)
                outconn.spread_pressure_reference_check()
        return

    def spread_backward_pressure_values(self, outconn):
        conn_idx = self.outl.index(outconn)
        inconn = self.inl[conn_idx]
        if not inconn.p.is_set and outconn.p.is_set and (
                (getattr(self, f'pr{conn_idx + 1}').is_set and getattr(self, f'pr{conn_idx + 1}_fit').rule in ['constant', 'static'])
                or (getattr(self, f'dp{conn_idx + 1}').is_set and getattr(self, f'dp{conn_idx + 1}_fit').rule in ['constant', 'static'])):
            if getattr(self, f'pr{conn_idx + 1}').is_set:
                inconn.p.val_SI = outconn.p.val_SI / getattr(self, f'pr{conn_idx + 1}').val_SI
                inconn.p.is_set = True
                inconn.p.is_var = False
                getattr(self, f'pr{conn_idx + 1}').is_set = False
            elif getattr(self, f'dp{conn_idx + 1}').is_set:
                inconn.p.val_SI = outconn.p.val_SI + getattr(self, f'dp{conn_idx + 1}').val_SI
                inconn.p.is_set = True
                inconn.p.is_var = False
                getattr(self, f'dp{conn_idx + 1}').is_set = False
            if inconn not in self.network.connections_spread_pressure_container:
                self.network.connections_spread_pressure_container.append(inconn)
                inconn.source.spread_backward_pressure_values(inconn)
                inconn.spread_pressure_reference_check()
        return

    def set_pressure_initial_factor(self, branch_index=0):
        inconn = self.inl[branch_index]
        outconn = self.outl[branch_index]
        if getattr(self, f'pr{branch_index + 1}').is_set:
            return getattr(self, f'pr{branch_index + 1}').val_SI
        elif getattr(self, f'dp{branch_index + 1}').is_set:
            if inconn.p.is_set:
                return (inconn.p.val_SI - getattr(self, f'dp{branch_index + 1}').val_SI) / inconn.p.val_SI
            elif outconn.p.is_set:
                return outconn.p.val_SI / (outconn.p.val_SI + getattr(self, f'dp{branch_index + 1}').val_SI)
        return 0.98

    def energy_balance_hot_func(self):
        q = 0
        for i in range(self.num_side.val):
            q += self.inl[i].m.val_SI * (self.outl[i].h.val_SI - self.inl[i].h.val_SI)
        return -q - self.Q.val_SI

    def energy_balance_hot_variables_columns(self):
        variables_columns1 = [data.J_col for i in range(self.num_side.val)
                              for data in [self.inl[i].m, self.inl[i].h, self.outl[i].h]
                              if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def energy_balance_hot_solve_isolated(self):
        return False

    def energy_balance_hot_deriv(self, increment_filter, k):
        for i in range(self.num_side.val):
            if self.is_variable(self.inl[i].m, increment_filter):
                self.network.jacobian[k, self.inl[i].m.J_col] = - self.outl[i].h.val_SI + self.inl[i].h.val_SI
            if self.is_variable(self.inl[i].h, increment_filter):
                self.network.jacobian[k, self.inl[i].h.J_col] = self.inl[i].m.val_SI
            if self.is_variable(self.outl[i].h, increment_filter):
                self.network.jacobian[k, self.outl[i].h.J_col] = - self.inl[i].m.val_SI

    def energy_balance_hot_tensor(self, increment_filter, k):
        pass

    def energy_balance_hot_func_doc(self, label):
        pass

    def auto_distribute_func(self):
        residual = []
        m_all = 0
        for i in range(self.num_side.val):
            m_all += self.inl[i].m.val_SI
        for i in range(self.num_side.val):
            residual += [self.inl[i].m.val_SI * (self.outl[i].h.val_SI - self.inl[i].h.val_SI) + self.Q.design * self.inl[i].m.val_SI / m_all]
        return residual

    def auto_distribute_variables_columns(self):
        variables_columns = []
        for i in range(self.num_side.val):
            variables_columnsi = []
            variables_columnsi += [data.J_col for data in [self.inl[i].m, self.inl[i].h, self.outl[i].h] if data.is_var]
            variables_columnsi.sort()
            variables_columns.append(variables_columnsi)
        return variables_columns

    def auto_distribute_solve_isolated(self):
        return False

    def auto_distribute_deriv(self, increment_filter, k):
        m_all = 0
        for i in range(self.num_side.val):
            m_all += self.inl[i].m.val_SI
        for i in range(self.num_side.val):
            if self.inl[i].m.is_var:
                self.network.jacobian[k + i, self.inl[i].m.J_col] = ((self.outl[i].h.val_SI - self.inl[i].h.val_SI) +
                                                                     self.Q.design * (m_all - self.inl[i].m.val_SI) / m_all ** 2)
            if self.inl[i].h.is_var:
                self.network.jacobian[k + i, self.inl[i].h.J_col] = - self.inl[i].m.val_SI
            if self.outl[i].h.is_var:
                self.network.jacobian[k + i, self.outl[i].h.J_col] = self.inl[i].m.val_SI

    def auto_distribute_tensor(self, increment_filter, k):
        pass

    def auto_distribute_func_doc(self, label):
        pass

    def T_out_func(self, outconn):
        i = outconn
        return self.outl[i].calc_T() - self.get_attr(f'T_out{i + 1}').val_SI

    def T_out_variables_columns(self, outconn):
        i = outconn
        variables_columns1 = [data.J_col
                              for data in [self.outl[i].h]
                              if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def T_out_solve_isolated(self, outconn):
        i = outconn
        if self.outl[i].fluid.is_var:
            return False
        if self.outl[i].p.is_set and not self.outl[i].h.is_set:
            self.outl[i].h.val_SI = h_mix_pT(self.outl[i].p.val_SI,
                                             self.get_attr(f'T_out{i + 1}').val_SI,
                                             self.outl[i].fluid_data,
                                             self.outl[i].mixing_rule)
            self.outl[i].h.is_set = True
            self.outl[i].h.is_var = False
            self.get_attr(f'T_out{i + 1}').is_set = False
            return True
        elif not self.outl[i].p.is_set and self.outl[i].h.is_set:
            self.outl[i].p.val_SI = p_mix_hT(self.outl[i].h.val_SI,
                                             self.get_attr(f'T_out{i + 1}').val_SI,
                                             self.outl[i].fluid_data,
                                             self.outl[i].mixing_rule
                                             )
            self.outl[i].p.is_set = True
            self.outl[i].p.is_var = False
            self.get_attr(f'T_out{i + 1}').is_set = False
            return True
        elif self.outl[i].p.is_set and self.outl[i].h.is_set:
            self.get_attr(f'T_out{i + 1}').is_set = False
            return True
        return False

    def T_out_deriv(self, increment_filter, k, outconn):
        i = outconn
        if self.outl[i].h.is_var:
            self.network.jacobian[k, self.outl[i].h.J_col] = dT_mix_pdh(self.outl[i].p.val_SI,
                                                                        self.outl[i].h.val_SI,
                                                                        self.outl[i].fluid_data,
                                                                        self.outl[i].mixing_rule)

    def T_out_tensor(self, increment_filter, k, outconn):
        pass

    def T_out_func_doc(self, label):
        pass

    def bus_func(self, bus):
        r"""
        Calculate the value of the bus function.

        Parameters
        ----------
        bus : aurora.connections.bus.Bus
            AURORA bus object.

        Returns
        -------
        val : float
        """
        q = 0
        for i in range(self.num_side.val):
            q += self.inl[i].m.val_SI * (self.outl[i].h.val_SI - self.inl[i].h.val_SI)
        return -q

    def bus_variables_columns(self, bus):
        variables_colmns1 = [data.J_col for i in range(self.num_side.val)
                             for data in [self.inl[i].m, self.inl[i].h, self.outl[i].h]
                             if data.is_var]
        variables_colmns1.sort()
        return [variables_colmns1]

    def bus_func_doc(self, bus):
        r"""
        Return LaTeX string of the bus function.

        Parameters
        ----------
        bus : aurora.connections.bus.Bus
            AURORA bus object.

        Returns
        -------
        latex : str
            LaTeX string of bus function.
        """
        return

    def bus_deriv(self, bus, increment_filter, k):
        r"""
        Calculate partial derivatives of the bus function.

        Parameters
        ----------
        bus : aurora.connections.bus.Bus
            AURORA bus object.

        Returns
        -------
        deriv : ndarray
            Matrix of partial derivatives.
        """
        f = self.calc_bus_value
        for i in range(self.num_side.val):
            if self.inl[i].m.is_var:
                self.network.jacobian[k, self.inl[i].m.J_col] -= self.numeric_deriv(f, 'm', self.inl[i], bus=bus)
            if self.inl[i].h.is_var:
                self.network.jacobian[k, self.inl[i].h.J_col] -= self.numeric_deriv(f, 'h', self.inl[i], bus=bus)
            if self.outl[i].h.is_var:
                self.network.jacobian[k, self.outl[i].h.J_col] -= self.numeric_deriv(f, 'h', self.outl[i], bus=bus)

    def bus_tensor(self, bus, increment_filter, k):
        pass

    def initialise_source(self, c, key):
        r"""
        Return a starting value for pressure and enthalpy at outlet.

        Parameters
        ----------
        c : aurora.connections.connection.Connection
            Connection to perform initialisation on.

        key : str
            Fluid property to retrieve.

        Returns
        -------
        val : float
            Starting value for pressure/enthalpy in SI units.
        """
        if key == 'p':
            return 50e5
        elif key == 'h':
            T = 600 + 273.15
            return h_mix_pT(c.p.val_SI, T, c.fluid_data, c.mixing_rule)

    def initialise_target(self, c, key):
        r"""
        Return a starting value for pressure and enthalpy at inlet.

        Parameters
        ----------
        c : aurora.connections.connection.Connection
            Connection to perform initialisation on.

        key : str
            Fluid property to retrieve.

        Returns
        -------
        val : float
            Starting value for pressure/enthalpy in SI units.
        """
        if key == 'p':
            return 50e5
        elif key == 'h':
            if c.target_id == 'in1':
                T = 300 + 273.15
            else:
                T = 220 + 273.15
            return h_mix_pT(c.p.val_SI, T, c.fluid_data, c.mixing_rule)

    def calc_Q_(self):
        q = 0
        for i in range(self.num_side.val):
            q += self.inl[i].m.val_SI * (self.outl[i].h.val_SI - self.inl[i].h.val_SI)
        return -q

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        # component parameters
        self.Q.val_SI = self.calc_Q_()
        for i in range(self.num_side.val):
            self.get_attr(f'T_out{i + 1}').val_SI = self.outl[i].calc_T()
            self.get_attr(f'pr{i + 1}').val_SI = self.outl[i].p.val_SI / self.inl[i].p.val_SI
            self.get_attr(f'dp{i + 1}').val_SI = self.inl[i].p.val_SI - self.outl[i].p.val_SI




