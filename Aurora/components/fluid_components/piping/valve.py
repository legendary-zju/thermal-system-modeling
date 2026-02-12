# -*- coding: utf-8

"""Module of class Valve.
"""

import numpy as np

from Aurora.components.component import Component
from Aurora.components.fluid_components.fluid_component import FluidComponent
from Aurora.components.component import component_registry
from Aurora.tools import logger
from Aurora.tools.data_containers import ComponentCharacteristics as dc_cc
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.data_containers import FitCoefficient as dc_fit
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools.helpers import AURORANetworkError


@component_registry
class Valve(FluidComponent):
    r"""
    The Valve throttles a fluid without changing enthalpy.

    **Mandatory Equations**

    - :py:meth:`AURORA.components.component.Component.fluid_func`
    - :py:meth:`AURORA.components.component.Component.mass_flow_func`

    **Optional Equations**

    - :py:meth:`AURORA.components.component.Component.pr_func`
    - :py:meth:`AURORA.components.component.Component.zeta_func`
    - :py:meth:`AURORA.components.piping.valve.Valve.dp_char_func`

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

    pr : float, dict, :code:`"var"`
        Outlet to inlet pressure ratio, :math:`pr/1`

    zeta : float, dict, :code:`"var"`
        Geometry independent friction coefficient,
        :math:`\frac{\zeta}{D^4}/\frac{1}{\text{m}^4}`.

    dp_char : AURORA.tools.characteristics.CharLine, dict
        Characteristic line for difference pressure to mass flow.

    """

    @staticmethod
    def component():
        return 'valve'

    def simplify_pressure_enthalpy_mass_topology(self, inconn):
        if self.simplify_pressure_enthalpy_mass_topology_check():
            self.network.branches_components.append(self)
            outconn = self.outl[0]
            # enthalpy
            conn_h_set_container = []
            h_value_set_container = []
            conn_h_shared_container = []
            for conn in self.inl + self.outl:
                if conn.h.is_set:
                    conn_h_set_container.append(conn)
                    h_value_set_container.append(conn.h.val)
                if conn.h.is_shared:
                    conn_h_shared_container.append(conn)
            # simplify enthalpy objective
            if conn_h_shared_container:
                for conn in set([c for c_shared in conn_h_shared_container for c in c_shared.h.shared_connection]
                                + self.inl + self.outl):
                    if not hasattr(conn, "_h_tmp"):
                        conn._h_tmp = conn.h
                    conn.h = inconn.h
            else:
                outconn._h_tmp = outconn.h
                outconn.h = inconn.h
            # set enthalpy value
            if conn_h_set_container:
                if len(set(h_value_set_container)) > 1:
                    msg = f"Has not set sole enthalpy value of branches of cycle closer component: {self.label}"
                    raise AURORANetworkError(msg)
                else:
                    # set h value
                    inconn.h.val = h_value_set_container[0]
                    inconn.h.is_set = True
            #
            for conn in self.inl + self.outl:
                conn.h.is_shared = True
                if conn not in conn.h.shared_connection:
                    conn.h.shared_connection.append(conn)
            outconn.target.simplify_pressure_enthalpy_mass_topology(outconn)

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
            'pr': dc_cp(
                min_val=1e-4,
                max_val=1,
                num_eq=1,
                func=self.pr_func,
                variables_columns=self.pr_variables_columns,
                solve_isolated=self.pr_solve_isolated,
                deriv=self.pr_deriv,
                tensor=self.pr_tensor,
                func_params={'pr': 'pr'},
                latex=self.pr_func_doc,
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
                solve_isolated=self.dp_solve_isolated,
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
                choice=[],
                rule='constant',
                constant=self.dp_constant_func_,
                default=self.dp_default_func_,
                charline=self.dp_charline_func_,
            ),
            'zeta': dc_cp(
                min_val=0,
                max_val=1e15,
                is_property=True,
                is_result=True,
                property_data=cpd['zeta'],
                SI_unit=cpd['zeta']['SI_unit'],
            ),
            'dp_char': dc_cc(
                param='m',
                char_params={'type': 'rel', 'inconn': 0, 'outconn': 0},
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

    def dp_charline_func_(self):
        r"""
        Equation for characteristic line of difference pressure to mass flow.

        Returns
        -------
        residual : ndarray
            Residual value of equation.

            .. math::

                0=p_\mathrm{in}-p_\mathrm{out}-f\left( expr \right)
        """
        p = self.dp_char.param
        expr = self.get_char_expr(p, **self.dp_char.char_params)
        if not expr:
            msg = ('Please choose a valid parameter, you want to link the '
                   'pressure drop to at component ' + self.label + '.')
            logger.error(msg)
            raise ValueError(msg)
        return self.dp_char.char_func.evaluate(expr) * self.dp.design

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
                5 \cdot 10^5 & \text{key = 'h'}
                \end{cases}
        """
        if key == 'p':
            return 4e5
        elif key == 'h':
            return 5e5

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
                5 \cdot 10^5 & \text{key = 'p'}\\
                5 \cdot 10^5 & \text{key = 'h'}
                \end{cases}
        """
        if key == 'p':
            return 5e5
        elif key == 'h':
            return 5e5

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        i = self.inl[0]
        o = self.outl[0]
        self.pr.val_SI = o.p.val_SI / i.p.val_SI
        self.dp.val_SI = i.p.val_SI - o.p.val_SI
        self.zeta.val_SI = self.calc_zeta(i, o)

    def entropy_balance(self):
        r"""
        Calculate entropy balance of a valve.

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

    def exergy_balance(self, T0):
        r"""
        Calculate exergy balance of a valve.

        Parameters
        ----------
        T0 : float
            Ambient temperature T0 / K.

        Note
        ----
        .. math::

            \dot{E}_\mathrm{P} =
            \begin{cases}
            \text{not defined (nan)} & T_\mathrm{in}, T_\mathrm{out} \geq T_0\\
            \dot{E}_\mathrm{out}^\mathrm{T}
            & T_\mathrm{in} > T_0 \geq T_\mathrm{out}\\
            \dot{E}_\mathrm{out}^\mathrm{T} - \dot{E}_\mathrm{in}^\mathrm{T}
            & T_0 \geq T_\mathrm{in}, T_\mathrm{out}\\
            \end{cases}

            \dot{E}_\mathrm{F} =
            \begin{cases}
            \dot{E}_\mathrm{in}^\mathrm{PH} - \dot{E}_\mathrm{out}^\mathrm{PH}
            & T_\mathrm{in}, T_\mathrm{out} \geq T_0\\
            \dot{E}_\mathrm{in}^\mathrm{T} + \dot{E}_\mathrm{in}^\mathrm{M}-
            \dot{E}_\mathrm{out}^\mathrm{M}
            & T_\mathrm{in} > T_0 \geq T_\mathrm{out}\\
            \dot{E}_\mathrm{in}^\mathrm{M} - \dot{E}_\mathrm{out}^\mathrm{M}
            & T_0 \geq T_\mathrm{in}, T_\mathrm{out}\\
            \end{cases}
        """
        if self.inl[0].T.val_SI > T0 and self.outl[0].T.val_SI > T0:
            self.E_P = np.nan
            self.E_F = self.inl[0].Ex_physical - self.outl[0].Ex_physical
        elif self.outl[0].T.val_SI <= T0 and self.inl[0].T.val_SI > T0:
            self.E_P = self.outl[0].Ex_therm
            self.E_F = self.inl[0].Ex_therm + (
                self.inl[0].Ex_mech - self.outl[0].Ex_mech)
        elif self.inl[0].T.val_SI <= T0 and self.outl[0].T.val_SI <= T0:
            self.E_P = self.outl[0].Ex_therm - self.inl[0].Ex_therm
            self.E_F = self.inl[0].Ex_mech - self.outl[0].Ex_mech
        else:
            msg = ('Exergy balance of a valve, where outlet temperature is '
                   'larger than inlet temperature is not implmented.')
            logger.warning(msg)
            self.E_P = np.nan
            self.E_F = np.nan

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
                'isoline_property': 'h',
                'isoline_value': self.inl[0].h.val,
                'isoline_value_end': self.outl[0].h.val,
                'starting_point_property': 'v',
                'starting_point_value': self.inl[0].vol.val,
                'ending_point_property': 'v',
                'ending_point_value': self.outl[0].vol.val
            }
        }
