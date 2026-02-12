# -*- coding: utf-8

"""Module of class Turbomachine.
"""

from Aurora.components.component import Component
from Aurora.components.fluid_components.fluid_component import FluidComponent
from Aurora.components.component import component_registry
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import ComponentCharacteristics as dc_cc
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.data_containers import FitCoefficient as dc_fit
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.document_models import generate_latex_eq


@component_registry
class Turbomachine(FluidComponent):
    r"""
    Parent class for compressor, pump and turbine.

    **Mandatory Equations**

    - :py:meth:`Aurora.components.component.Component.fluid_func`
    - :py:meth:`Aurora.components.component.Component.mass_flow_func`

    **Optional Equations**

    - :py:meth:`Aurora.components.component.Component.pr_func`
    - :py:meth:`Aurora.components.turbomachinery.base.base.energy_balance_func`

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

    P : float, dict
        Power, :math:`P/\text{W}`

    pr : float, dict, :code:`"var"`
        Outlet to inlet pressure ratio, :math:`pr/1`

    Example
    -------
    For an example please refer to:

    - :class:`Aurora.components.turbomachinery.compressor.Compressor`
    - :class:`Aurora.components.turbomachinery.pump.Pump`
    - :class:`Aurora.components.turbomachinery.turbine.Turbine`
    """

    @staticmethod
    def component():
        return 'turbomachine'

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
        if self.__class__.__name__ in ['Turbine', 'SteamTurbine']:
            factor = 0.2
        elif self.__class__.__name__ in ['Compressor', 'Pump']:
            factor = 5
        else:
            factor = 1
        return factor

    def get_parameters(self):
        return {
            'P': dc_cp(
                func=self.energy_balance_func,
                variables_columns=self.energy_balance_variables_columns,
                solve_isolated=self.energy_balance_solve_isolated,
                deriv=self.energy_balance_deriv,
                tensor=self.energy_balance_tensor,
                num_eq=1,
                latex=self.energy_balance_func_doc,
                property_data=cpd['P'],
                SI_unit=cpd['P']['SI_unit'],
                scale=ps['m']['scale'] * ps['h']['scale'],
                var_scale=ps['m']['scale'] * ps['h']['scale']
            ),
            'pr': dc_cp(
                func=self.pr_func,
                variables_columns=self.pr_variables_columns,
                solve_isolated=self.pr_solve_isolated,
                deriv=self.pr_deriv,
                tensor=self.pr_tensor,
                num_eq=1,
                func_params={'pr': 'pr', 'inconn': 0, 'outconn': 0},
                latex=self.pr_func_doc,
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
                scale=ps['p']['scale'],
                var_scale=ps['pr']['scale']
            ),
            'pr_char': dc_cc(
                param='v',
                char_params={'type': 'rel', 'inconn': 0, 'outconn': 0}
            ),
            'pr_fit': dc_fit(
                rule='constant',
                constant=self.pr_constant_func_,
            ),
            'dp': dc_cp(
                min_val=-1e15,
                max_val=1e15,
                num_eq=1,
                deriv=self.dp_deriv,
                variables_columns=self.dp_variables_columns,
                solve_isolated=self.dp_solve_isolated,
                func=self.dp_func,
                tensor=self.dp_tensor,
                func_params={'dp': 'dp', 'inconn': 0, 'outconn': 0},
                property_data=cpd['dp'],
                SI_unit=cpd['dp']['SI_unit'],
                scale=ps['p']['scale'],
                var_scale=ps['p']['scale']
            ),
            'dp_char': dc_cc(
                param='v',
                char_params={'type': 'rel', 'inconn': 0, 'outconn': 0}
            ),
            'dp_fit': dc_fit(
                rule='constant',
                constant=self.dp_constant_func_,
            )
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

    @staticmethod
    def inlets():
        return ['in1']

    @staticmethod
    def outlets():
        return ['out1']

    def energy_balance_func(self):
        r"""
        Calculate energy balance of a turbomachine.

        Returns
        -------
        residual : float
            Residual value of turbomachine energy balance

            .. math::

                0=\dot{m}_{in}\cdot\left(h_{out}-h_{in}\right)-P
        """
        return self.inl[0].m.val_SI * (
            self.outl[0].h.val_SI - self.inl[0].h.val_SI) - self.P.val_SI

    def energy_balance_variables_columns(self):
        i = self.inl[0]
        o = self.outl[0]
        variables_columns1 = [data.J_col for data in [i.m, i.h, o.h, self.P] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def energy_balance_take_effect(self):
        pass

    def energy_balance_solve_isolated(self):
        i = self.inl[0]
        o = self.outl[0]
        if sum([1 if data.is_var else 0 for data in [i.m, i.h, o.h, self.P]]) > 1:
            return False
        if i.m.is_var:
            i.m.val_SI = self.P.val_SI / (o.h.val_SI - i.h.val_SI)
            i.m.is_set = True
            i.m.is_var = False
            return True
        elif i.h.is_var:
            i.h.val_SI = o.h.val_SI - self.P.val_SI / i.m.val_SI
            i.h.is_set = True
            i.h.is_var = False
            return True
        elif o.h.is_var:
            o.h.val_SI = self.P.val_SI / i.m.val_SI + i.h.val_SI
            o.h.is_set = True
            o.h.is_var = False
            return True
        else:
            return True

    def energy_balance_func_doc(self, label):
        r"""
        Calculate energy balance of a turbomachine.

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
            r'0=\dot{m}_\mathrm{in}\cdot\left(h_\mathrm{out}-h_\mathrm{in}'
            r'\right)-P')
        return generate_latex_eq(self, latex, label)

    def energy_balance_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of energy balance of a turbomachine.

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
        # custom variable P
        if self.P.is_var:
            self.network.jacobian[k, self.P.J_col] = -1

    def energy_balance_tensor(self, increment_filter, k):
        i = self.inl[0]
        o = self.outl[0]
        if i.m.is_var and i.h.is_var:
            self.network.tensor[i.m.J_col, i.h.J_col, k] = -1
            self.network.tensor[i.h.J_col, i.m.J_col, k] = -1
        if i.m.is_var and o.h.is_var:
            self.network.tensor[i.m.J_col, o.h.J_col, k] = 1
            self.network.tensor[o.h.J_col, i.m.J_col, k] = 1

    def pr_default_func_(self, pr='', inconn=0, outconn=0):
        pass

    def dp_default_func_(self, dp='', inconn=0, outconn=0):
        pass

    def bus_func(self, bus):
        r"""
        Calculate the value of the bus function.

        Parameters
        ----------
        bus : tespy.connections.bus.Bus
            TESPy bus object.

        Returns
        -------
        residual : float
            Value of energy transfer :math:`\dot{E}`. This value is passed to
            :py:meth:`Aurora.components.component.Component.calc_bus_value`
            for value manipulation according to the specified characteristic
            line of the bus.

            .. math::

                \dot{E} = \dot{m}_{in} \cdot \left(h_{out} - h_{in} \right)
        """
        return self.inl[0].m.val_SI * (
            self.outl[0].h.val_SI - self.inl[0].h.val_SI
        )

    def bus_variables_columns(self, bus):
        variables_columns1 = [data.J_col for data in [self.inl[0].m, self.inl[0].h, self.outl[0].h] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

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
        f = self.calc_bus_value
        numeric_variables_list = (
                [('m', self.is_variable(c.m, increment_filter), c, c.m.J_col) for c in [self.inl[0]]] +
                [('h', self.is_variable(c.h, increment_filter), c, c.h.J_col) for c in [self.inl[0], self.outl[0]]])
        self.generate_numerical_bus_tensor(f, k, numeric_variables_list, bus=bus)

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        self.P.val_SI = self.inl[0].m.val_SI * (
                self.outl[0].h.val_SI - self.inl[0].h.val_SI)
        self.pr.val_SI = self.outl[0].p.val_SI / self.inl[0].p.val_SI
        self.dp.val_SI = self.inl[0].p.val_SI - self.outl[0].p.val_SI

    def entropy_balance(self):
        r"""
        Calculate entropy balance of turbomachine.

        Note
        ----
        The entropy balance makes the follwing parameter available:

        .. math::

            \text{S\_irr}=\dot{m} \cdot \left(s_\mathrm{out}-s_\mathrm{in}
            \right)\\
        """
        self.S_irr = self.inl[0].m.val_SI * (
            self.outl[0].s.val_SI - self.inl[0].s.val_SI
        )

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
                'isoline_property': 's',
                'isoline_value': self.inl[0].s.val,
                'isoline_value_end': self.outl[0].s.val,
                'starting_point_property': 'v',
                'starting_point_value': self.inl[0].vol.val,
                'ending_point_property': 'v',
                'ending_point_value': self.outl[0].vol.val
            }
        }
