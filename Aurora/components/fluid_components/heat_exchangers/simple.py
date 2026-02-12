# -*- coding: utf-8

"""Module of class SimpleHeatExchanger.
"""

import math
import warnings

import numpy as np

from Aurora.components.component import Component
from Aurora.components.fluid_components.fluid_component import FluidComponent
from Aurora.components.component import component_registry
from Aurora.tools import logger
from Aurora.tools.data_containers import ComponentCharacteristics as dc_cc
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import GroupedComponentProperties as dc_gcp
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.data_containers import FitCoefficient as dc_fit
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools.fluid_properties import s_mix_ph
from Aurora.tools.fluid_properties.helpers import darcy_friction_factor as dff
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import fluid_property_data as fpd
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.global_vars import space_time_property_data as stpd


@component_registry
class SimpleHeatExchanger(FluidComponent):
    r"""
    A basic heat exchanger representing a heat source or heat sink.

    The component SimpleHeatExchanger is the parent class for the components:

    - :py:class:`AURORA.components.heat_exchangers.solar_collector.SolarCollector`
    - :py:class:`AURORA.components.heat_exchangers.parabolic_trough.ParabolicTrough`
    - :py:class:`AURORA.components.piping.pipe.Pipe`

    **Mandatory Equations**

    - :py:meth:`AURORA.components.component.Component.fluid_func`
    - :py:meth:`AURORA.components.component.Component.mass_flow_func`

    **Optional Equations**

    - :py:meth:`AURORA.components.component.Component.pr_func`
    - :py:meth:`AURORA.components.component.Component.zeta_func`
    - :py:meth:`AURORA.components.heat_exchangers.simple.SimpleHeatExchanger.energy_balance_func`
    - :py:meth:`AURORA.components.heat_exchangers.simple.SimpleHeatExchanger.darcy_group_func`
    - :py:meth:`AURORA.components.heat_exchangers.simple.SimpleHeatExchanger.hw_group_func`
    - :py:meth:`AURORA.components.heat_exchangers.simple.SimpleHeatExchanger.kA_group_func`
    - :py:meth:`AURORA.components.heat_exchangers.simple.SimpleHeatExchanger.kA_char_group_func`

    Inlets/Outlets

    - in1
    - out1

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

    Q : float, dict, :code:`"var"`
        Heat transfer, :math:`Q/\text{W}`.

    pr : float, dict, :code:`"var"`
        Outlet to inlet pressure ratio, :math:`pr/1`.

    zeta : float, dict, :code:`"var"`
        Geometry independent friction coefficient,
        :math:`\frac{\zeta}{D^4}/\frac{1}{\text{m}^4}`.

    D : float, dict, :code:`"var"`
        Diameter of the pipes, :math:`D/\text{m}`.

    L : float, dict, :code:`"var"`
        Length of the pipes, :math:`L/\text{m}`.

    ks : float, dict, :code:`"var"`
        Pipe's roughness, :math:`ks/\text{m}`.

    darcy_group : str, dict
        Parametergroup for pressure drop calculation based on pipes dimensions
        using darcy weissbach equation.

    ks_HW : float, dict, :code:`"var"`
        Pipe's roughness, :math:`ks/\text{1}`.

    hw_group : str, dict
        Parametergroup for pressure drop calculation based on pipes dimensions
        using hazen williams equation.

    kA : float, dict, :code:`"var"`
        Area independent heat transfer coefficient,
        :math:`kA/\frac{\text{W}}{\text{K}}`.

    kA_char : AURORA.tools.characteristics.CharLine, dict
        Characteristic line for heat transfer coefficient.

    Tamb : float, dict
        Ambient temperature, provide parameter in network's temperature unit.

    kA_group : str, dict
        Parametergroup for heat transfer calculation from ambient temperature
        and area independent heat transfer coefficient kA.

    Example
    -------
    The SimpleHeatExchanger can be used as a sink or source of heat. This
    component does not simulate the secondary side of the heat exchanger. It
    is possible to calculate the pressure ratio with the Darcy-Weisbach
    equation or in case of liquid water use the Hazen-Williams equation.
    Also, given ambient temperature and the heat transfer coeffiecient, it is
    possible to predict heat transfer.
    """

    @staticmethod
    def component():
        return 'heat exchanger simple'

    def spread_forward_pressure_values(self, inconn):
        outconn = self.outl[0]
        if inconn.p.is_set and not outconn.p.is_set and ((self.pr.is_set and self.pr_fit.rule in ['constant', 'static'])
                                                         or (self.dp.is_set and self.dp_fit.rule in ['constant', 'static'])):
            if self.pr.is_set:
                outconn.p.val_SI = inconn.p.val_SI * self.pr.val_SI
                outconn.p.is_set = True
                outconn.p.is_var = False
                self.pr.is_set = False
            elif self.dp.is_set:
                outconn.p.val_SI = inconn.p.val_SI - self.dp.val_SI
                outconn.p.is_set = True
                outconn.p.is_var = False
                self.dp.is_set = False
            if outconn not in self.network.connections_spread_pressure_container:
                self.network.connections_spread_pressure_container.append(outconn)
                outconn.target.spread_forward_pressure_values(outconn)
                outconn.spread_pressure_reference_check()
        return

    def spread_backward_pressure_values(self, outconn):
        inconn = self.inl[0]
        if not inconn.p.is_set and outconn.p.is_set and ((self.pr.is_set and self.pr_fit.rule in ['constant', 'static'])
                                                         or (self.dp.is_set and self.dp_fit.rule in ['constant', 'static'])):
            if self.pr.is_set:
                inconn.p.val_SI = outconn.p.val_SI / self.pr.val_SI
                inconn.p.is_set = True
                inconn.p.is_var = False
                self.pr.is_set = False
            elif self.dp.is_set:
                inconn.p.val_SI = outconn.p.val_SI + self.dp.val_SI
                inconn.p.is_set = True
                inconn.p.is_var = False
                self.dp.is_set = False
            if inconn not in self.network.connections_spread_pressure_container:
                self.network.connections_spread_pressure_container.append(inconn)
                inconn.source.spread_backward_pressure_values(inconn)
                inconn.spread_pressure_reference_check()
        return

    def set_pressure_initial_factor(self, branch_index=0):
        inconn = self.inl[0]
        outconn = self.outl[0]
        if self.pr.is_set:
            return self.pr.val_SI
        elif self.dp.is_set:
            if inconn.p.is_set:
                return (inconn.p.val_SI - self.dp.val_SI) / inconn.p.val_SI
            elif outconn.p.is_set:
                return outconn.p.val_SI / (outconn.p.val_SI + self.dp.val_SI)
        return 0.98

    def get_parameters(self):
        return {
            'Q': dc_cp(
                func=self.energy_balance_func,
                variables_columns=self.energy_balance_variables_columns,
                deriv=self.energy_balance_deriv,
                tensor=self.energy_balance_tensor,
                latex=self.energy_balance_func_doc,
                num_eq=1,
                property_data=cpd['Q'],
                SI_unit=cpd['Q']['SI_unit'],
                scale=ps['m']['scale'] * ps['h']['scale'],
                var_scale=ps['m']['scale'] * ps['h']['scale']
            ),
            'pr': dc_cp(
                min_val=1e-4,
                max_val=1,
                func=self.pr_func,
                variables_columns=self.pr_variables_columns,
                deriv=self.pr_deriv,
                tensor=self.pr_tensor,
                latex=self.pr_func_doc,
                num_eq=1,
                func_params={'pr': 'pr'},
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
                scale=ps['p']['scale'],
                var_scale=ps['pr']['scale']
            ),
            'dp': dc_cp(
                min_val=0,
                max_val=1e15,
                num_eq=1,
                deriv=self.dp_deriv,
                variables_columns=self.dp_variables_columns,
                func=self.dp_func,
                tensor=self.dp_tensor,
                func_params={'dp': 'dp', 'inconn': 0, 'outconn': 0},
                property_data=cpd['dp'],
                SI_unit=cpd['dp']['SI_unit'],
                scale=ps['p']['scale'],
                var_scale=ps['p']['scale']
            ),
            'pr_fit': dc_fit(
                rule='constant',
                constant=self.pr_constant_func_,
                default=self.pr_default_func_,
            ),
            'dp_fit': dc_fit(
                choice=['darcy', 'hazen'],
                rule='constant',
                constant=self.dp_constant_func_,
                default=self.dp_default_func_,
                darcy=self.dp_darcy_func_,
                hazen=self.dp_hazen_func_,
            ),
            'zeta': dc_cp(
                min_val=0,
                max_val=1e15,
                is_property=True,
                is_result=True,
                property_data=cpd['zeta'],
                SI_unit=cpd['zeta']['SI_unit'],
            ),
            'D': dc_cp(
                min_val=1e-2,
                max_val=2,
                d=1e-4,
                is_property=True,
                is_result=True,
                property_data=stpd['l'],
                SI_unit=stpd['l']['SI_unit'],
            ),
            'L': dc_cp(
                min_val=1e-1,
                d=1e-3,
                is_property=True,
                is_result=True,
                property_data=stpd['l'],
                SI_unit=stpd['l']['SI_unit'],
            ),
            'ks': dc_cp(
                val=1e-4,
                min_val=1e-7,
                max_val=1e-3,
                d=1e-8,
                is_property=True,
                is_result=True,
                property_data=cpd['ks'],
                SI_unit=cpd['ks']['SI_unit'],
            ),
            'ks_HW': dc_cp(
                val=10,
                min_val=1e-1,
                max_val=1e3,
                d=1e-2,
                is_property=True,
                is_result=True,
                property_data=cpd['ks'],
                SI_unit=cpd['ks']['SI_unit'],
            ),
            'kA': dc_cp(
                min_val=0,
                d=1,
                latex=self.kA_func_doc,
                func=self.kA_func,
                variables_columns=self.kA_variables_columns,
                deriv=self.kA_deriv,
                tensor=self.kA_tensor,
                num_eq=1,
                scale=ps['m']['scale'] * ps['h']['scale'],
                property_data=cpd['kA'],
                SI_unit=cpd['kA']['SI_unit'],
            ),
            'kA_fit': dc_fit(
                rule='constant',
                static=self.kA_static_func_,
                constant=self.kA_constant_func_,
                default=self.kA_default_func_,
                charline=self.kA_charline_func_,
            ),
            'kA_char': dc_cc(
                param='m',
                char_params={'type': 'rel', 'inconn': 0, 'outconn': 0}
            ),
            'Tamb': dc_cp(
                is_property=True,
                is_result=True,
                property_data=fpd['T'],
                SI_unit=fpd['T']['SI_unit'],
            ),
            'dissipative': dc_simple(val=None),
            'hf': dc_cp(
                val=0,
                val_SI=500,
                min_val=0,
                is_property=True,
                is_result=True,
                property_data=cpd['hf'],
                SI_unit=cpd['hf']['SI_unit'],
            ),
            'exm': dc_cp(
                val=0,
                val_SI=0.61,
                min_val=0,
                is_property=True,
                is_result=True,
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
            ),
            'fA': dc_cp(
                val=0,
                min_val=0,
                is_property=True,
                is_result=True,
                property_data=cpd['fA'],
                SI_unit=cpd['fA']['SI_unit'],
            ),
        }

    def get_bypass_constraints(self):
        return {
            'pressure_equality_constraints': dc_cons(
                func=self.pressure_equality_func,
                variables_columns=self.pressure_equality_variables_columns,
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
                deriv=self.enthalpy_equality_deriv,
                tensor=self.enthalpy_equality_tensor,
                constant_deriv=False,
                latex=self.enthalpy_equality_func_doc,
                num_eq=self.num_i,
                scale=ps['h']['scale']
            )
        }

    @staticmethod
    def inlets():
        return ['in1']

    @staticmethod
    def outlets():
        return ['out1']

    def energy_balance_func(self):
        r"""
        Equation for pressure drop calculation.

        Returns
        -------
        residual : float
            Residual value of equation:

            .. math::

                0 =\dot{m}_{in}\cdot\left( h_{out}-h_{in}\right) -\dot{Q}
        """
        return self.inl[0].m.val_SI * (
            self.outl[0].h.val_SI - self.inl[0].h.val_SI
        ) - self.Q.val_SI

    def energy_balance_variables_columns(self):
        i = self.inl[0]
        o = self.outl[0]
        variables_columns = [data.J_col for data in [i.m, i.h, o.h, self.Q] if data.is_var]
        variables_columns.sort()
        return [variables_columns]

    def energy_balance_func_doc(self, label):
        r"""
        Equation for pressure drop calculation.

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
            r'0 = \dot{m}_\mathrm{in} \cdot \left(h_\mathrm{out} - '
            r'h_\mathrm{in} \right) -\dot{Q}'
        )
        return generate_latex_eq(self, latex, label)

    def energy_balance_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of energy balance.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        i = self.inl[0]
        o = self.outl[0]
        if i.m.is_var:
            self.network.jacobian[k, i.m.J_col] = o.h.val_SI - i.h.val_SI
        if i.h.is_var:
            self.network.jacobian[k, i.h.J_col] = -i.m.val_SI
        if o.h.is_var:
            self.network.jacobian[k, o.h.J_col] = i.m.val_SI
        # custom variable Q
        if self.Q.is_var:
            self.network.jacobian[k, self.Q.J_col] = -1

    def energy_balance_tensor(self, increment_filter, k):
        pass

    def dp_darcy_func_(self, dp='', inconn=0, outconn=0):
        for element in ['L', 'ks', 'D']:
            if not self.get_attr(element).is_set:
                msg = f'{element} is not set in darcy fit of {self.__class__.__name__}: {self.label}'
                logger.error(msg)
                raise AttributeError(msg)
        i = self.inl[inconn]
        o = self.outl[outconn]
        if abs(i.m.val_SI) < 1e-4:
            return 0
        visc_i = i.calc_viscosity(T0=i.T.val_SI)
        visc_o = o.calc_viscosity(T0=o.T.val_SI)
        v_i = i.calc_vol(T0=i.T.val_SI)
        v_o = o.calc_vol(T0=o.T.val_SI)
        Re = 4 * abs(i.m.val_SI) / (math.pi * self.D.val_SI * (visc_i + visc_o) / 2)
        return (
                 8 * abs(i.m.val_SI) * i.m.val_SI * (v_i + v_o)
                / 2 * self.L.val_SI * dff(Re, self.ks.val_SI, self.D.val_SI)
                / (math.pi ** 2 * self.D.val_SI ** 5)
        )

    def calc_darcy_variables_(self):
        # custom variables of hydro group
        elements = ['L', 'ks', 'D']
        func = self.dp_func
        for variable_name in self.darcy_group.elements:
            parameter = self.get_attr(variable_name)
            if parameter.is_var:
                deriv = (
                    self.numeric_deriv(func, variable_name, None)
                )

    def dp_hazen_func_(self, dp='', inconn=0, outconn=0):
        for element in ['L', 'ks_HW', 'D']:
            if not self.get_attr(element).is_set:
                msg = f'{element} is not set in hazen fit of {self.__class__.__name__}: {self.label}'
                logger.error(msg)
                raise AttributeError(msg)
        i = self.inl[inconn]
        o = self.outl[outconn]
        if abs(i.m.val_SI) < 1e-4:
            return 0
        v_i = i.calc_vol(T0=i.T.val_SI)
        v_o = o.calc_vol(T0=o.T.val_SI)
        return  (10.67 * abs(i.m.val_SI) ** 1.852 * self.L.val_SI /
                    (self.ks_HW.val_SI ** 1.852 * self.D.val_SI ** 4.871)
                ) * (9.81 * ((v_i + v_o) / 2) ** 0.852)

    def calc_hazen_variables_(self):
        elements = ['L', 'ks_HW', 'D']
        pass

    def kA_func(self):
        r"""
        Calculate heat transfer from heat transfer coefficient.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = \dot{m}_{in} \cdot \left( h_{out} - h_{in}\right) +
                kA \cdot \Delta T_{log}

                \Delta T_{log} = \begin{cases}
                \frac{T_{in}-T_{out}}{\ln{\frac{T_{in}-T_{amb}}
                {T_{out}-T_{amb}}}} & T_{in} > T_{out} \\
                \frac{T_{out}-T_{in}}{\ln{\frac{T_{out}-T_{amb}}
                {T_{in}-T_{amb}}}} & T_{in} < T_{out}\\
                0 & T_{in} = T_{out}
                \end{cases}

                T_{amb}: \text{ambient temperature}
        """
        if not self.Tamb.is_set:
            msg = f'Tamb is not set in kA calculation of {self.__class__.__name__}: {self.label}'
            logger.error(msg)
            raise AttributeError(msg)
        i = self.inl[0]
        o = self.outl[0]
        ttd_1 = i.calc_T() - self.Tamb.val_SI
        ttd_2 = o.calc_T() - self.Tamb.val_SI
        # For numerical stability: If temperature differences have
        # different sign use mean difference to avoid negative logarithm.
        if (ttd_1 / ttd_2) < 0:
            td_log = (ttd_2 + ttd_1) / 2
        elif ttd_1 > ttd_2:
            td_log = (ttd_1 - ttd_2) / math.log(ttd_1 / ttd_2)
        elif ttd_1 < ttd_2:
            td_log = (ttd_2 - ttd_1) / math.log(ttd_2 / ttd_1)
        else:
            # both values are equal
            td_log = ttd_2
        return i.m.val_SI * (o.h.val_SI - i.h.val_SI) + self.kA_fit() * td_log

    def kA_variables_columns(self):
        i = self.inl[0]
        o = self.outl[0]
        variables_columns = [data.J_col for data in [i.h, o.h] if data.is_var]  # [i.m, i.p, i.h, o.p, o.h, self.kA]
        variables_columns.sort()
        return [variables_columns]

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
            r'\begin{split}' + '\n'
            r'0=&\dot{m}_\mathrm{in}\cdot\left(h_\mathrm{out}-'
            r'h_\mathrm{in}\right)+kA \cdot \Delta T_\mathrm{log}\\' + '\n'
            r'\Delta T_\mathrm{log} = &\begin{cases}' + '\n'
            r'\frac{T_\mathrm{in}-T_\mathrm{out}}{\ln{\frac{T_\mathrm{in}-'
            r'T_\mathrm{amb}}{T_\mathrm{out}-T_\mathrm{amb}}}} &'
            r' T_\mathrm{in} > T_\mathrm{out} \\' + '\n'
            r'\frac{T_\mathrm{out}-T_\mathrm{in}}{\ln{\frac{'
            r'T_\mathrm{out}-T_\mathrm{amb}}{T_\mathrm{in}-'
            r'T_\mathrm{amb}}}} & T_\mathrm{in} < T_\mathrm{out}\\' + '\n'
            r'0 & T_\mathrm{in} = T_\mathrm{out}' + '\n'
            r'\end{cases}\\' + '\n'
            r'T_\mathrm{amb} =& \text{ambient temperature}' + '\n'
            r'\end{split}'
        )
        return generate_latex_eq(self, latex, label)

    def kA_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of kA group.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        f = self.kA_group_func
        i = self.inl[0]
        o = self.outl[0]
        # if self.is_variable(i.m, increment_filter):
        #     self.network.jacobian[k, i.m.J_col] = o.h.val_SI - i.h.val_SI
        # if self.is_variable(i.p, increment_filter):
        #     self.network.jacobian[k, i.p.J_col] = self.numeric_deriv(f, 'p', i)
        if self.is_variable(i.h, increment_filter):
            self.network.jacobian[k, i.h.J_col] = self.numeric_deriv(f, 'h', i)
        # if self.is_variable(o.p, increment_filter):
        #     self.network.jacobian[k, o.p.J_col] = self.numeric_deriv(f, 'p', o)
        if self.is_variable(o.h, increment_filter):
            self.network.jacobian[k, o.h.J_col] = self.numeric_deriv(f, 'h', o)

    def kA_tensor(self, increment_filter, k):
        pass

    def kA_static_func_(self, **kwargs):
        return self.kA.val_SI

    def kA_constant_func_(self, **kwargs):
        if self.hf.is_set and self.fA.is_set:
            return self.hf.val_SI * self.fA.val_SI
        else:
            return self.kA.val_SI

    def kA_default_func_(self, **kwargs):
        alfa1 = (self.inl[0].m.val_SI / self.inl[0].m.design) ** self.exm.val_SI
        fkA = alfa1
        return self.kA.design * fkA

    def kA_charline_func_(self, **kwargs):
        p = self.kA_char.param
        expr = self.get_char_expr(p, **self.kA_char.char_params)
        fkA = 2 / (1 + 1 / self.kA_char.char_func.evaluate(expr))
        return self.kA.design * fkA

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

                \dot{E} = \dot{m}_{in} \cdot \left( h_{out} - h_{in} \right)
        """
        return self.inl[0].m.val_SI * (
            self.outl[0].h.val_SI - self.inl[0].h.val_SI)

    def bus_variables_columns(self, bus):
        i = self.inl[0]
        o = self.outl[0]
        variables_columns = [data.J_col for data in [i.m, i.h, o.h] if data.is_var]
        variables_columns.sort()
        return [variables_columns]

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
            r'\dot{m}_\mathrm{in} \cdot \left(h_\mathrm{out} - '
            r'h_\mathrm{in} \right)')

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
        pass

    def initialise_source(self, c, key):
        r"""
        Return a starting value for pressure and enthalpy the outlets.

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
                \begin{cases}
                1 \cdot 10^5 \; \frac{\text{J}}{\text{kg}} & \dot{Q} < 0\\
                3 \cdot 10^5 \; \frac{\text{J}}{\text{kg}} & \dot{Q} = 0\\
                5 \cdot 10^5 \; \frac{\text{J}}{\text{kg}} & \dot{Q} > 0
                \end{cases} & \text{key = 'h'}\\
                \; \; \; \; 10^5 \text{Pa} & \text{key = 'p'}
                \end{cases}

        """
        if key == 'p':
            return 1e5
        elif key == 'h':
            if self.Q.val < 0 and self.Q.is_set:
                return 1e5
            elif self.Q.val > 0 and self.Q.is_set:
                return 5e5
            else:
                return 3e5

    def initialise_target(self, c, key):
        r"""
        Return a starting value for pressure and enthalpy the inlets.

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
                1 \cdot 10^5 & \text{key = 'p'}\\
                \begin{cases}
                5 \cdot 10^5 & \dot{Q} < 0\\
                3 \cdot 10^5 & \dot{Q} = 0\\
                1 \cdot 10^5 & \dot{Q} > 0
                \end{cases} & \text{key = 'h'}\\
                \end{cases}
        """
        if key == 'p':
            return 1e5
        elif key == 'h':
            if self.Q.val < 0 and self.Q.is_set:
                return 5e5
            elif self.Q.val > 0 and self.Q.is_set:
                return 1e5
            else:
                return 3e5

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        i = self.inl[0]
        o = self.outl[0]

        self.Q.val_SI = i.m.val_SI * (o.h.val_SI - i.h.val_SI)
        self.pr.val_SI = o.p.val_SI / i.p.val_SI
        self.dp.val_SI = i.p.val_SI - o.p.val_SI
        self.zeta.val_SI = self.calc_zeta(i, o)

        if self.Tamb.is_set:
            ttd_1 = i.T.val_SI - self.Tamb.val_SI
            ttd_2 = o.T.val_SI - self.Tamb.val_SI

            if (ttd_1 / ttd_2) < 0:
                td_log = np.nan
            if ttd_1 > ttd_2:
                td_log = (ttd_1 - ttd_2) / math.log(ttd_1 / ttd_2)
            elif ttd_1 < ttd_2:
                td_log = (ttd_2 - ttd_1) / math.log(ttd_2 / ttd_1)
            else:
                # both values are equal
                td_log = ttd_1

            self.kA.val_SI = abs(self.Q.val_SI / td_log)
            self.kA.is_result = True
        else:
            self.kA.is_result = False

    def entropy_balance(self):
        r"""
        Calculate entropy balance of a simple heat exchanger.

        The allocation of the entropy streams due to heat exchanged and due to
        irreversibility is performed by solving for T:

        .. math::

            h_\mathrm{out} - h_\mathrm{in} = \int_\mathrm{out}^\mathrm{in}
            v \cdot dp - \int_\mathrm{out}^\mathrm{in} T \cdot ds

        As solving :math:`\int_\mathrm{out}^\mathrm{in} v \cdot dp` for non
        isobaric processes would require perfect process knowledge (the path)
        on how specific volume and pressure change throught the component, the
        heat transfer is splitted into three separate virtual processes:

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

            \text{S\_Q}=\dot{m} \cdot \left(s_\mathrm{out*}-s_\mathrm{in*}
            \right)\\
            \text{S\_irr}=\dot{m} \cdot \left(s_\mathrm{out}-s_\mathrm{in}
            \right) - \text{S\_Q}\\
            \text{T\_mQ}=\frac{\dot{Q}}{\text{S\_Q}}
        """
        i = self.inl[0]
        o = self.outl[0]

        p1_star = i.p.val_SI * (o.p.val_SI / i.p.val_SI) ** 0.5
        s1_star = s_mix_ph(
            p1_star, i.h.val_SI, i.fluid_data, i.mixing_rule, T0=i.T.val_SI
        )
        s2_star = s_mix_ph(
            p1_star, o.h.val_SI, o.fluid_data, o.mixing_rule, T0=o.T.val_SI
        )
        self.S_Q = i.m.val_SI * (s2_star - s1_star)
        self.S_irr = i.m.val_SI * (o.s.val_SI - i.s.val_SI) - self.S_Q
        self.T_mQ = (o.h.val_SI - i.h.val_SI) / (s2_star - s1_star)

    def exergy_balance(self, T0):
        r"""
        Calculate exergy balance of a simple heat exchanger.

        The exergy of heat is calculated by allocation of thermal and
        mechanical share of exergy in the physical exergy. Depending on the
        temperature levels at the inlet and outlet of the heat exchanger as
        well as the direction of heat transfer (input or output) fuel and
        product exergy are calculated as follows.

        Parameters
        ----------
        T0 : float
            Ambient temperature T0 / K.

        Note
        ----
        If the fluid transfers heat to the ambient, you can specify
        :code:`mysimpleheatexchanger.set_attr(dissipative=False)` if you do
        NOT want the exergy production nan (only applicable in case
        :math:`\dot{Q}<0`).

        .. math ::

            \dot{E}_\mathrm{P} =
            \begin{cases}
            \begin{cases}
            \begin{cases}
            \text{not defined (nan)} & \text{if dissipative}\\
            \dot{E}_\mathrm{in}^\mathrm{T} - \dot{E}_\mathrm{out}^\mathrm{T} &
            \text{else}\\
            \end{cases}
            & T_\mathrm{in}, T_\mathrm{out} \geq T_0\\
            \dot{E}_\mathrm{out}^\mathrm{T}
            & T_\mathrm{in} \geq T_0 > T_\mathrm{out}\\
            \dot{E}_\mathrm{out}^\mathrm{T} - \dot{E}_\mathrm{in}^\mathrm{T}
            & T_0 \geq T_\mathrm{in}, T_\mathrm{out}\\
            \end{cases} & \dot{Q} < 0\\

            \begin{cases}
            \dot{E}_\mathrm{out}^\mathrm{PH} - \dot{E}_\mathrm{in}^\mathrm{PH}
            & T_\mathrm{in}, T_\mathrm{out} \geq T_0\\
            \dot{E}_\mathrm{in}^\mathrm{T} + \dot{E}_\mathrm{out}^\mathrm{T}
            & T_\mathrm{out} > T_0 \geq T_\mathrm{in}\\
            \dot{E}_\mathrm{in}^\mathrm{T} - \dot{E}_\mathrm{out}^\mathrm{T} +
            \dot{E}_\mathrm{out}^\mathrm{M} - \dot{E}_\mathrm{in}^\mathrm{M} +
            & T_0 \geq T_\mathrm{in}, T_\mathrm{out}\\
            \end{cases} & \dot{Q} > 0\\
            \end{cases}

            \dot{E}_\mathrm{F} =
            \begin{cases}
            \begin{cases}
            \dot{E}_\mathrm{in}^\mathrm{PH} - \dot{E}_\mathrm{out}^\mathrm{PH}
            & T_\mathrm{in}, T_\mathrm{out} \geq T_0\\
            \dot{E}_\mathrm{in}^\mathrm{T} + \dot{E}_\mathrm{in}^\mathrm{M} +
            \dot{E}_\mathrm{out}^\mathrm{T} - \dot{E}_\mathrm{out}^\mathrm{M}
            & T_\mathrm{in} \geq T_0 > T_\mathrm{out}\\
            \dot{E}_\mathrm{out}^\mathrm{T} - \dot{E}_\mathrm{in}^\mathrm{T} +
            \dot{E}_\mathrm{in}^\mathrm{M} - \dot{E}_\mathrm{out}^\mathrm{M} +
            & T_0 \geq T_\mathrm{in}, T_\mathrm{out}\\
            \end{cases} & \dot{Q} < 0\\

            \begin{cases}
            \dot{E}_\mathrm{out}^\mathrm{T} - \dot{E}_\mathrm{in}^\mathrm{T}
            & T_\mathrm{in}, T_\mathrm{out} \geq T_0\\
            \dot{E}_\mathrm{in}^\mathrm{T} + \dot{E}_\mathrm{in}^\mathrm{M} -
            \dot{E}_\mathrm{out}^\mathrm{M}
            & T_\mathrm{out} > T_0 \geq T_\mathrm{in}\\
            \dot{E}_\mathrm{in}^\mathrm{T}-\dot{E}_\mathrm{out}^\mathrm{T}
            & T_0 \geq T_\mathrm{in}, T_\mathrm{out}\\
            \end{cases} & \dot{Q} > 0\\
            \end{cases}

            \dot{E}_\mathrm{bus} =
            \begin{cases}
            \begin{cases}
            \dot{E}_\mathrm{P} & \text{other cases}\\
            \dot{E}_\mathrm{in}^\mathrm{T}
            & T_\mathrm{in} \geq T_0 > T_\mathrm{out}\\
            \end{cases} & \dot{Q} < 0\\
            \dot{E}_\mathrm{F} & \dot{Q} > 0\\
            \end{cases}
        """
        if self.dissipative.val is None:
            self.dissipative.val = True
            msg = (
                "In a future version of TESPy, the dissipative property must "
                "explicitly be set to True or False in the context of the "
                f"exergy analysis for component {self.label}."
            )
            logger.warning(msg)
        if self.Q.val < 0:
            if self.inl[0].T.val_SI >= T0 and self.outl[0].T.val_SI >= T0:
                if self.dissipative.val:
                    self.E_P = np.nan
                else:
                    self.E_P = self.inl[0].Ex_therm - self.outl[0].Ex_therm
                self.E_F = self.inl[0].Ex_physical - self.outl[0].Ex_physical
                self.E_bus = {
                    "chemical": 0, "physical": 0, "massless": self.E_P
                }
            elif self.inl[0].T.val_SI >= T0 and self.outl[0].T.val_SI < T0:
                self.E_P = self.outl[0].Ex_therm
                self.E_F = self.inl[0].Ex_therm + self.outl[0].Ex_therm + (
                    self.inl[0].Ex_mech - self.outl[0].Ex_mech)
                self.E_bus = {
                    "chemical": 0, "physical": 0,
                    "massless": self.inl[0].Ex_therm + self.outl[0].Ex_therm
                }
            elif self.inl[0].T.val_SI <= T0 and self.outl[0].T.val_SI <= T0:
                self.E_P = self.outl[0].Ex_therm - self.inl[0].Ex_therm
                self.E_F = self.outl[0].Ex_therm - self.outl[0].Ex_therm + (
                    self.inl[0].Ex_mech - self.outl[0].Ex_mech)
                self.E_bus = {
                    "chemical": 0, "physical": 0, "massless": self.E_P
                }
            else:
                msg = ('Exergy balance of simple heat exchangers, where '
                       'outlet temperature is higher than inlet temperature '
                       'with heat extracted is not implmented.')
                logger.warning(msg)
                self.E_P = np.nan
                self.E_F = np.nan
                self.E_bus = {
                    "chemical": np.nan, "physical": np.nan, "massless": np.nan
                }
        elif self.Q.val > 0:
            if self.inl[0].T.val_SI >= T0 - 1e-6 and self.outl[0].T.val_SI >= T0 - 1e-6:
                self.E_P = self.outl[0].Ex_physical - self.inl[0].Ex_physical
                self.E_F = self.outl[0].Ex_therm - self.inl[0].Ex_therm
                self.E_bus = {
                    "chemical": 0, "physical": 0, "massless": self.E_F
                }
            elif self.inl[0].T.val_SI <= T0 and self.outl[0].T.val_SI > T0:
                self.E_P = self.outl[0].Ex_therm + self.inl[0].Ex_therm
                self.E_F = self.inl[0].Ex_therm + (
                    self.inl[0].Ex_mech - self.outl[0].Ex_mech)
                self.E_bus = {
                    "chemical": 0, "physical": 0,
                    "massless": self.inl[0].Ex_therm
                }
            elif self.inl[0].T.val_SI < T0 and self.outl[0].T.val_SI < T0:
                if self.dissipative.val:
                    self.E_P = np.nan
                else:
                    self.E_P = self.inl[0].Ex_therm - self.outl[0].Ex_therm + (
                        self.outl[0].Ex_mech - self.inl[0].Ex_mech
                    )
                self.E_F = self.inl[0].Ex_therm - self.outl[0].Ex_therm
                self.E_bus = {
                    "chemical": 0, "physical": 0, "massless": self.E_F
                }
            else:
                msg = ('Exergy balance of simple heat exchangers, where '
                       'inlet temperature is higher than outlet temperature '
                       'with heat injected is not implmented.')
                logger.warning(msg)
                self.E_P = np.nan
                self.E_F = np.nan
                self.E_bus = {
                    "chemical": np.nan, "physical": np.nan, "massless": self.E_F
                }
        else:
            # fully dissipative
            self.E_P = np.nan
            self.E_F = self.inl[0].Ex_physical - self.outl[0].Ex_physical
            self.E_bus = {
                "chemical": np.nan, "physical": np.nan, "massless": np.nan
            }

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
            1: {
                'isoline_property': 'p',
                'isoline_value': self.inl[0].p.val,
                'isoline_value_end': self.outl[0].p.val,
                'starting_point_property': 's',
                'starting_point_value': self.inl[0].s.val,
                'ending_point_property': 's',
                'ending_point_value': self.outl[0].s.val
            }
        }
