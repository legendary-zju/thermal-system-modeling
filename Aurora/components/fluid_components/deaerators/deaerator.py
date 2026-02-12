# -*- coding: utf-8

"""Module of class Deaerator.
"""
import math

import numpy as np

from Aurora.components.component import component_registry
from Aurora.components.fluid_components.fluid_component import FluidComponent
from Aurora.components.fluid_components.distributors.base import NodeBase
from Aurora.tools import logger
from Aurora.tools.data_containers import ComponentCharacteristics as dc_cc
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import GroupedComponentCharacteristics as dc_gcc
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.data_containers import FitCoefficient as dc_fit
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools.fluid_properties import s_mix_ph
from Aurora.tools.fluid_properties import dh_mix_dpQ
from Aurora.tools.fluid_properties import h_mix_pQ, p_mix_hQ
from Aurora.tools.helpers import AURORANetworkError


@component_registry
class Deaerator(NodeBase):
    r"""
    A Deaerator extracts oxygen from vapour.
    The saturated liquid is obtained on the outlet of the deaerator.
    **Mandatory Equations**

    - :py:meth:`Aurora.components.component.Component.fluid_func`
    - :py:meth:`Aurora.components.component.Component.mass_flow_func`
    - :py:meth:`Aurora.components.deaerator.Deaerator.energy_balance_func`
    - :py:meth:`Aurora.components.deaerator.Deaerator.saturated_liquid_func

    **Optional Equations**

    - liquid side :py:meth:`Aurora.components.component.Component.pr_func`
    - liquid side :py:meth:`Aurora.components.component.Component.zeta_func`

    Inlets/Outlets

    - in1, in2 (index 1: cooled liquid, index 2: extract vapour)
    - out1，out2 (index 1: saturated liquid, index 2: drain away saturated liquid)

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
        Outlet to inlet pressure ratio at liquid side, :math:`pr/1`.

    pr2 : float, dict, :code:`"var"`
        Outlet to inlet pressure ratio at gas side, :math:`pr/1`.

    zeta1 : float, dict, :code:`"var"`
        Geometry independent friction coefficient at liquid side,
        :math:`\frac{\zeta}{D^4}/\frac{1}{\text{m}^4}`.

    zeta2 : float, dict, :code:`"var"`
        Geometry independent friction coefficient at gas side,
        :math:`\frac{\zeta}{D^4}/\frac{1}{\text{m}^4}`.

    DTN : float, dict
        Upper gas terminal temperature difference.

    kA : float, dict
        Area independent heat transfer coefficient,
        :math:`kA/\frac{\text{W}}{\text{K}}`.

    kA_char : dict
        Area independent heat transfer coefficient characteristic.

    kA_char1 : Aurora.tools.characteristics.CharLine, dict
        Characteristic line for hot side heat transfer coefficient.

    kA_char2 : Aurora.tools.characteristics.CharLine, dict
        Characteristic line for cold side heat transfer coefficient.
    """

    @staticmethod
    def component():
        return 'deaerator'

    @staticmethod
    def inlets():
        return ['in1', 'in2']

    @staticmethod
    def outlets():
        return ['out1', 'out2']

    @staticmethod
    def is_branch_source():
        return True

    def start_branch(self):
        # main liquid
        outconn = self.outl[0]
        branch = {
            "connections": [outconn],
            "components": [self, outconn.target],
            "subbranches": {}
        }
        outconn.target.propagate_to_target(branch)
        # drain liquid
        outconn_d = self.outl[1]
        subbranch = {
            "connections": [outconn_d],
            "components": [self, outconn_d.target],
            "subbranches": {}
        }
        outconn_d.target.propagate_to_target(subbranch)
        branch["subbranches"][outconn_d.label] = subbranch

        return {outconn.label: branch}

    def propagate_to_target(self, branch):
        return

    def propagate_wrapper_to_target(self, branch):
        if self in branch["components"]:
            return
        for outconn in self.outl:
            branch["connections"] += [outconn]
            branch["components"] += [self]
            outconn.target.propagate_wrapper_to_target(branch)

    def simplify_pressure_enthalpy_mass_topology(self, inconni):
        """
        The pressure objective and enthalpy objective of outlet have been shared.

        :param inconni:
        :return:
        """
        if self.simplify_pressure_enthalpy_mass_topology_check():
            self.network.branches_components.append(self)
            outconn_main = self.outl[0]
            # pressure
            conn_p_set_container = []
            p_value_set_container = []
            conn_p_shared_container = []
            # properties information
            for conn in self.outl:
                if conn.p.is_set:
                    conn_p_set_container.append(conn)
                    p_value_set_container.append(conn.p.val)
                if conn.p.is_shared:
                    conn_p_shared_container.append(conn)
            # simplify pressure objective
            if conn_p_shared_container:
                for conn in set([c for c_shared in conn_p_shared_container for c in c_shared.p.shared_connection]
                                + self.outl):
                    if not hasattr(conn, "_p_tmp"):
                        conn._p_tmp = conn.p
                    conn.p = outconn_main.p
            else:
                for outconn in self.outl:
                    outconn._p_tmp = outconn.p
                    outconn.p = outconn_main.p
            # set pressure value
            if conn_p_set_container:
                if len(set(p_value_set_container)) > 1:
                    msg = f"Has not set sole pressure value of branches of deaerator component: {self.label}"
                    raise AURORANetworkError(msg)
                else:
                    # set p value
                    outconn_main.p.val = p_value_set_container[0]
                    outconn_main.p.is_set = True
                    outconn_main.p.is_var = False
            # posterior
            for conn in self.outl:
                conn.p.is_shared = True
                if conn not in conn.p.shared_connection:
                    conn.p.shared_connection.append(conn)
            for outconn in self.outl:
                outconn.target.simplify_pressure_enthalpy_mass_topology(outconn)

    def looking_forward_pressure_values(self, inconn):
        """
        Looking forward for original pressure set point along branch, in order to spread pressure value.

        :param inconn:
        :return:
        """
        conn_idx = self.inl.index(inconn)  # the index of branch
        outconn = self.outl[0]  # outconn: object of outlet connection.
        outconn_d = self.outl[1]
        p_ef = self.get_p_ef_obj(conn_idx)
        #
        if inconn not in self.network.connections_looking_pressure_container:
            self.network.connections_looking_pressure_container.append(inconn)
            # start spreading pressure value
            if inconn.p.is_set and not outconn.p.is_set and p_ef.is_set:
                self.spread_forward_pressure_values(inconn)
                inconn.spread_pressure_reference_check()
            elif not inconn.p.is_set and outconn.p.is_set and p_ef.is_set:
                self.spread_backward_pressure_values(outconn)
                outconn.spread_pressure_reference_check()
            # looking for p.set never looked
            outconn.target.looking_forward_pressure_values(outconn)
            outconn_d.target.looking_forward_pressure_values(outconn_d)
            for inconni in self.inl:
                if inconni != inconn:
                    inconni.source.looking_backward_pressure_values(inconni)
        else:
            return

    def looking_backward_pressure_values(self, outconn):
        """
        Looking backward for original pressure set point along branch, in order to spread pressure value.
        The single branch component could inherit this method directly.

        :param outconn:
        :return:
        """
        if outconn not in self.network.connections_looking_pressure_container:
            self.network.connections_looking_pressure_container.append(outconn)
            for conn_idx, inconn in enumerate(self.inl):
                p_ef = self.get_p_ef_obj(conn_idx)
                # start spreading pressure value
                if inconn.p.is_set and not outconn.p.is_set and p_ef.is_set:
                    self.spread_forward_pressure_values(inconn)
                    inconn.spread_pressure_reference_check()
                elif not inconn.p.is_set and outconn.p.is_set and p_ef.is_set:
                    self.spread_backward_pressure_values(outconn)
                    outconn.spread_pressure_reference_check()
                # looking for p.set never looked
                inconn.source.looking_backward_pressure_values(inconn)
            for outconni in self.outl:
                if outconni != outconn:
                    outconni.target.looking_forward_pressure_values(outconni)
        else:
            return

    def spread_forward_pressure_values(self, inconn):
        """
        Spread forward pressure value set along branch.

        :param inconn:
        :return:
        """
        conn_idx = self.inl.index(inconn)  # the index of branch
        for outconn in self.outl:
            if not outconn in self.network.connections_spread_pressure_container:
                self.network.connections_spread_pressure_container.append(outconn)
                if conn_idx == 0:
                    if inconn.p.is_set and not outconn.p.is_set and (
                            (self.pr1.is_set and self.pr1_fit.rule in ['constant', 'static'])
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
                        outconn.target.spread_forward_pressure_values(outconn)
                        outconn.spread_pressure_reference_check()
                        if ((self.pr2.is_set and self.pr2_fit.rule in ['constant', 'static'])
                            or (self.dp2.is_set and self.dp2_fit.rule in ['constant', 'static'])):
                            self.spread_backward_pressure_values(outconn)
                else:  # 2
                    if inconn.p.is_set and not outconn.p.is_set and (
                            (self.pr2.is_set and self.pr2_fit.rule in ['constant', 'static'])
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
                        outconn.target.spread_forward_pressure_values(outconn)
                        outconn.spread_pressure_reference_check()
                        if ((self.pr1.is_set and self.pr1_fit.rule in ['constant', 'static'])
                                or (self.dp1.is_set and self.dp1_fit.rule in ['constant', 'static'])):
                            self.spread_backward_pressure_values(outconn)
        return

    def spread_backward_pressure_values(self, outconn):
        """
        Spread backward pressure value set along branch.

        :param outconn:
        :return:
        """
        for conn_idx, inconn in enumerate(self.inl):
            if not inconn in self.network.connections_spread_pressure_container:
                self.network.connections_spread_pressure_container.append(inconn)
                if conn_idx == 0:
                    if not inconn.p.is_set and outconn.p.is_set and (
                            (self.pr1.is_set and self.pr1_fit.rule in ['constant', 'static'])
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
                        inconn.source.spread_backward_pressure_values(inconn)
                        inconn.spread_pressure_reference_check()
                else: # 2
                    if not inconn.p.is_set and outconn.p.is_set and (
                            (self.pr2.is_set and self.pr2_fit.rule in ['constant', 'static'])
                            or (self.dp2.is_set and self.dp2_fit.rule in ['constant', 'static'])):
                        if self.pr2.is_set:
                            inconn.p.val_SI = outconn.p.val_SI / self.pr2.val_SI
                            inconn.p.is_set = True
                            inconn.p.is_var = False
                            self.pr1.is_set = False
                        elif self.dp2.is_set:
                            inconn.p.val_SI = outconn.p.val_SI + self.dp2.val_SI
                            inconn.p.is_set = True
                            inconn.p.is_var = False
                            self.dp2.is_set = False
                        inconn.source.spread_backward_pressure_values(inconn)
                        inconn.spread_pressure_reference_check()
        return

    def looking_for_pressure_set_boundary(self, inconn):
        """
        Looking for pressure set boundary.

        :param inconn:
        :return:
        """
        if inconn.p.is_set:
            self.spread_forward_pressure_initial(inconn)
            inconn.source.spread_backward_pressure_initial(inconn)
            return
        for outconn in self.outl:
            if outconn.p.is_set:
                outconn.target.spread_forward_pressure_initial(outconn)
                self.spread_backward_pressure_initial(outconn)
                return
        if inconn not in self.network.connections_pressure_boundary_container:
            self.network.connections_pressure_boundary_container.append(inconn)
            for outconn in self.outl:
                outconn.target.looking_for_pressure_set_boundary(outconn)
        return

    def spread_forward_pressure_initial(self, inconn):
        """
        Set initial pressure value based on pressure value set without pressure ratio or pressure drop.

        :param inconn:
        :return:
        """
        for _, outconn in enumerate(self.outl):
            if outconn not in self.network.connections_pressure_initial_container:
                self.network.connections_pressure_initial_container.append(outconn)
                conn_idx = self.inl.index(inconn)
                if inconn.p.val_SI and not outconn.p.val_SI:
                    outconn.p.val_SI = inconn.p.val_SI * self.set_pressure_initial_factor(conn_idx)
                outconn.target.spread_forward_pressure_initial(outconn)
        return

    def spread_backward_pressure_initial(self, outconn):
        """
        Set initial pressure value based on pressure value set without pressure ratio or pressure drop.

        :param outconn:
        :return:
        """
        for conn_idx, inconn in enumerate(self.inl):
            if inconn not in self.network.connections_pressure_initial_container:
                self.network.connections_pressure_initial_container.append(inconn)
                if outconn.p.val_SI and not inconn.p.val_SI:
                    inconn.p.val_SI = outconn.p.val_SI / self.set_pressure_initial_factor(conn_idx)
                inconn.source.spread_backward_pressure_initial(inconn)
        return

    def set_pressure_initial_factor(self, branch_index=0):
        """
        Set initial pressure ratio factor for pressure initial value generation.

        :param branch_index:
        :return:
        """
        return 0.99

    def manage_fluid_equations(self):
        self.num_fluid_eqs = 0
        variable_fluids = set(
            [fluid for c in self.inl + self.outl for fluid in c.fluid.is_var]
        )
        num_fluid_eq = len(variable_fluids)
        fluid_equations = {
            'fluid_constraints': dc_cons(
                func=self.fluid_func,
                variables_columns=self.fluid_variables_columns,
                deriv=self.fluid_deriv,
                tensor=self.fluid_tensor,
                constant_deriv=False,
                latex=self.fluid_func_doc,
                num_eq=num_fluid_eq,
                fluid_composition_list=list(variable_fluids),
                scale=ps['fluid']['scale'] * ps['m']['scale']),
        }
        for key, equation in fluid_equations.items():
            if equation.num_eq > 0:
                equation.label = f"fluid composition equation: {key} of {self.__class__.__name__}: {self.label}"
                self.network.fluid_equations_module_container.append(equation)
                self.num_fluid_eqs += equation.num_eq

    @staticmethod
    def initialise_source(c, key):
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
                10^6 & \text{key = 'p'}\\
                h\left(p, x=1 \right) & \text{key = 'h' at outlet 1}\\
                \end{cases}
        """
        if key == 'p':  # 10e5
            return 10e5 * 1.5
        elif key == 'h':
            if c.source_id == 'out1':
                return h_mix_pQ(c.p.val_SI, 0, c.fluid_data)
            elif c.source_id == 'out2':
                return h_mix_pQ(c.p.val_SI, 1, c.fluid_data)

    def get_parameters(self):
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
                var_scale=ps['m']['scale'] * ps['h']['scale'],
            ),
            'pr1': dc_cp(
                min_val=1e-4,
                max_val=1,
                num_eq=1,
                func=self.pr_func,
                variables_columns=self.pr_variables_columns,
                solve_isolated=self.pr_solve_isolated,
                deriv=self.pr_deriv,
                tensor=self.pr_tensor,
                latex=self.pr_func_doc,
                func_params={'pr': 'pr1', 'inconn': 0, 'outconn': 0},
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
                scale=ps['p']['scale'],
                var_scale=ps['pr']['scale']
            ),
            'pr2': dc_cp(
                min_val=1e-4,
                max_val=1,
                num_eq=1,
                func=self.pr_func,
                variables_columns=self.pr_variables_columns,
                solve_isolated=self.pr_solve_isolated,
                latex=self.pr_func_doc,
                deriv=self.pr_deriv,
                tensor=self.pr_tensor,
                func_params={'pr': 'pr2', 'inconn': 1, 'outconn': 0},
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
                scale=ps['p']['scale'],
                var_scale=ps['pr']['scale']
            ),
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
        }

    def get_mandatory_constraints(self):
        return {
            'mass_flow_constraints': dc_cons(
                func=self.mass_flow_func,
                variables_columns=self.mass_flow_variables_columns,
                solve_isolated=self.mass_flow_solve_isolated,
                deriv=self.mass_flow_deriv,
                tensor=self.mass_flow_tensor,
                constant_deriv=True,
                latex=self.mass_flow_func_doc,
                num_eq=1,
                scale=ps['m']['scale']
            ),
            'energy_balance_constraints': dc_cons(
                func=self.energy_balance_func,
                variables_columns=self.energy_balance_variables_columns,
                solve_isolated=self.energy_balance_solve_isolated,
                deriv=self.energy_balance_deriv,
                tensor=self.energy_balance_tensor,
                constant_deriv=False,
                latex=self.energy_balance_func_doc,
                num_eq=1,
                scale=ps['m']['scale'] * ps['h']['scale']
            ),
            'saturated_liquid_constraints': dc_cons(
                func=self.saturated_liquid_func,
                variables_columns=self.saturated_liquid_variables_columns,
                solve_isolated=self.saturated_liquid_solve_isolated,
                deriv=self.saturated_liquid_deriv,
                tensor=self.saturated_liquid_tensor,
                constant_deriv=False,
                latex=self.saturated_liquid_func_doc,
                num_eq=1,
                scale=ps['h']['scale']
            ),
            'saturated_vapour_constraints': dc_cons(
                func=self.saturated_vapour_func,
                variables_columns=self.saturated_vapour_variables_columns,
                solve_isolated=self.saturated_vapour_solve_isolated,
                deriv=self.saturated_vapour_deriv,
                tensor=self.saturated_vapour_tensor,
                constant_deriv=False,
                latex=self.saturated_vapour_func_doc,
                num_eq=1,
                scale=ps['h']['scale']
            ),
        }

    def fluid_func(self):
        r"""
        Calculate the vector of residual values for fluid balance equations.

        Returns
        -------
        residual : list
            Vector of residual values for component's fluid balance.

            .. math::

                0 = \sum_i \dot{m}_{in,i} \cdot x_{fl,in,i} -
                \dot {m}_{out} \cdot x_{fl,out}\\
                \forall fl \in \text{network fluids},
                \; \forall i \in \text{inlets}
        """
        residual = []
        for fluid, x in self.outl[0].fluid.val.items():
            res = -x * self.outl[0].m.val_SI
            for i in self.inl:
                res += i.fluid.val[fluid] * i.m.val_SI
            residual += [res]
        return residual

    def fluid_variables_columns(self):
        pass

    def fluid_func_doc(self, label):
        r"""
        Calculate the vector of residual values for fluid balance equations.

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
            r'0=\sum_i \dot{m}_{\mathrm{in,}i} \cdot x_{fl\mathrm{,in,}i}'
            r'- \dot {m}_\mathrm{out} \cdot x_{fl\mathrm{,out}}'
            r'\; \forall fl \in \text{network fluids,} \; \forall i \in'
            r'\text{inlets}'
        )
        return generate_latex_eq(self, latex, label)

    def fluid_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of fluid balance.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        o = self.outl[0]
        for fluid, x in self.outl[0].fluid.val.items():
            for i in self.inl:
                if fluid in i.fluid.is_var:
                    self.network.fluid_jacobian[k, i.fluid.J_col[fluid]] = i.m.val_SI
            if fluid in o.fluid.is_var:
                self.network.fluid_jacobian[k, o.fluid.J_col[fluid]] = -o.m.val_SI
            k += 1  # k-th fluid equation

    def fluid_tensor(self, increment_filter, k):
        pass

    def energy_balance_func(self):
        r"""
        Equation for deaerator energy balance.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = \dot{m}_{in,1} \cdot \left(h_{out,1} - h_{in,1} \right) +
                \dot{m}_{in,2} \cdot \left(h_{out,1} - h_{in,2} \right)
        """
        res = 0
        for i in self.inl:
            res += i.m.val_SI * i.h.val_SI
        for o in self.outl:
            res -= o.m.val_SI * o.h.val_SI
        return res

    def energy_balance_variables_columns(self):
        variables_columns1 = []
        variables_columns1 += [data.J_col for c in self.inl + self.outl for data in [c.m, c.h] if
                               data.is_var]  # [c.m, c.h]
        variables_columns1.sort()
        return [variables_columns1]

    def energy_balance_solve_isolated(self):
        if sum([1 if data.is_var else 0 for conn in self.inl + self.outl for data in [conn.m, conn.h]]) > 1:
            return False
        for inconn in self.inl:
            if inconn.m.is_var:
                inconn.m.val_SI = ((sum([
                                            outconn.m.val_SI * outconn.h.val_SI if not outconn.m.is_var and not outconn.h.is_var else 0
                                            for outconn in self.outl]) -
                                    sum([
                                            inconn.m.val_SI * inconn.h.val_SI if not inconn.m.is_var and not inconn.h.is_var else 0
                                            for inconn in self.inl])) /
                                   inconn.h.val_SI)
                inconn.m.is_set = True
                inconn.m.is_var = False
                return True
            if inconn.h.is_var:
                inconn.h.val_SI = ((sum([
                                            outconn.m.val_SI * outconn.h.val_SI if not outconn.m.is_var and not outconn.h.is_var else 0
                                            for outconn in self.outl]) -
                                    sum([
                                            inconn.m.val_SI * inconn.h.val_SI if not inconn.m.is_var and not inconn.h.is_var else 0
                                            for inconn in self.inl])) /
                                   inconn.m.val_SI)
                inconn.h.is_set = True
                inconn.h.is_var = False
                return True
        for outconn in self.outl:
            if outconn.m.is_var:
                outconn.m.val_SI = ((sum([
                                             inconn.m.val_SI * inconn.h.val_SI if not inconn.m.is_var and not inconn.h.is_var else 0
                                             for inconn in self.inl]) -
                                     sum([
                                             outconn.m.val_SI * outconn.h.val_SI if not outconn.m.is_var and not outconn.h.is_var else 0
                                             for outconn in self.outl])) /
                                    outconn.h.val_SI)
                outconn.m.is_set = True
                outconn.m.is_var = False
                return True
            if outconn.h.is_var:
                outconn.h.val_SI = ((sum([
                                             inconn.m.val_SI * inconn.h.val_SI if not inconn.m.is_var and not inconn.h.is_var else 0
                                             for inconn in self.inl]) -
                                     sum([
                                             outconn.m.val_SI * outconn.h.val_SI if not outconn.m.is_var and not outconn.h.is_var else 0
                                             for outconn in self.outl])) /
                                    outconn.m.val_SI)
                outconn.h.is_set = True
                outconn.h.is_var = False
                return True
        return True

    def energy_balance_func_doc(self, label):
        r"""
        Equation for deaerator energy balance.

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
            r'\left(h_\mathrm{out,1} - h_\mathrm{in,2} \right)')
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

        for i in self.inl:
            if self.is_variable(i.m):  #
                self.network.jacobian[k, i.m.J_col] = i.h.val_SI
            if self.is_variable(i.h):
                self.network.jacobian[k, i.h.J_col] = i.m.val_SI
        for o in self.outl:
            if self.is_variable(o.m):
                self.network.jacobian[k, o.m.J_col] = - o.h.val_SI
            if self.is_variable(o.h):
                self.network.jacobian[k, o.h.J_col] = - o.m.val_SI

    def energy_balance_tensor(self, increment_filter, k):
        pass

    def saturated_liquid_func(self):
        r"""
        Calculate outlet liquid state.

        Returns
        -------
        residual : float
            Residual value of equation

            .. math::

                0 = h_{out,1} - h\left(p_{out,1}, x=0 \right)
        """
        o = self.outl[0]
        return o.h.val_SI - h_mix_pQ(o.p.val_SI, 0, o.fluid_data)

    def saturated_liquid_variables_columns(self):
        o = self.outl[0]
        variables_columns1 = [data.J_col for data in [o.h] if data.is_var]  # [o.p, o.h]
        variables_columns1.sort()
        return [variables_columns1]

    def saturated_liquid_solve_isolated(self):
        o = self.outl[0]
        if not o.p.is_var and not o.h.is_var:
            return True
        elif not o.p.is_var and o.h.is_var:
            o.h.val_SI = h_mix_pQ(o.p.val_SI, 0, o.fluid_data)
            o.h.is_set = True
            o.h.is_var = False
            return True
        elif o.p.is_var and not o.h.is_var:
            o.p.val_SI = p_mix_hQ(o.h.val_SI, 0, o.fluid_data)
            o.p.is_set = True
            o.p.is_var = False
            return True
        return False

    def saturated_liquid_func_doc(self, label):
        r"""
        Calculate outlet liquid state.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        latex = r'0=h_\mathrm{out,1}-h\left(p_\mathrm{out,1}, x=0 \right)'
        return generate_latex_eq(self, latex, label)

    def saturated_liquid_deriv(self, increment_filter, k):
        r"""
        Partial derivatives of saturated liquid at mere outlet function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        o = self.outl[0]
        # if self.is_variable(o.p):
        #     self.network.jacobian[k, o.p.J_col] = -dh_mix_dpQ(o.p.val_SI, 0, o.fluid_data)
        if self.is_variable(o.h):
            self.network.jacobian[k, o.h.J_col] = 1

    def saturated_liquid_tensor(self, increment_filter, k):
        pass

    def saturated_vapour_func(self):
        r"""
        Calculate outlet liquid state.

        Returns
        -------
        residual : float
            Residual value of equation

            .. math::

                0 = h_{out,2} - h\left(p_{out,2}, x=1 \right)
        """
        o = self.outl[1]
        return o.h.val_SI - h_mix_pQ(o.p.val_SI, 1, o.fluid_data)

    def saturated_vapour_variables_columns(self):
        o = self.outl[1]
        variables_columns1 = [data.J_col for data in [o.h] if data.is_var]  # [o.p, o.h]
        variables_columns1.sort()
        return [variables_columns1]

    def saturated_vapour_solve_isolated(self):
        o = self.outl[1]
        if not o.p.is_var and not o.h.is_var:
            return True
        elif not o.p.is_var and o.h.is_var:
            o.h.val_SI = h_mix_pQ(o.p.val_SI, 1, o.fluid_data)
            o.h.is_set = True
            o.h.is_var = False
            return True
        elif o.p.is_var and not o.h.is_var:
            o.p.val_SI = p_mix_hQ(o.h.val_SI, 1, o.fluid_data)
            o.p.is_set = True
            o.p.is_var = False
            return True
        return False

    def saturated_vapour_func_doc(self, label):
        return None

    def saturated_vapour_deriv(self, increment_filter, k):
        r"""
        Partial derivatives of saturated liquid at mere outlet function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        o = self.outl[1]
        # if self.is_variable(o.p):
        #     self.network.jacobian[k, o.p.J_col] = -dh_mix_dpQ(o.p.val_SI, 1, o.fluid_data)
        if self.is_variable(o.h):
            self.network.jacobian[k, o.h.J_col] = 1

    def saturated_vapour_tensor(self, increment_filter, k):
        pass

    def energy_balance_hot_func(self):
        r"""
        Equation for hot side deaerator energy balance.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 =\dot{m}_{in,2} \cdot \left(h_{out,1}-h_{in,2}\right)-\dot{Q}
        """
        return self.inl[1].m.val_SI * (
            self.outl[0].h.val_SI - self.inl[1].h.val_SI
        ) - self.Q.val_SI

    def energy_balance_hot_variables_columns(self):
        i = self.inl[1]
        o = self.outl[0]
        variables_columns1 = [data.J_col for data in [o.h, i.h, i.m] if data.is_var]  #
        variables_columns1.sort()
        return [variables_columns1]

    def energy_balance_hot_solve_isolated(self):
        i = self.inl[1]
        o = self.outl[0]
        if not i.m.is_var and not i.h.is_var and not o.h.is_var:
            self.Q.is_set = False
            return True
        elif i.m.is_var and not i.h.is_var and not o.h.is_var:
            i.m.val_SI = self.Q.val_SI / (o.h.val_SI - i.h.val_SI)
            i.m.is_set = True
            i.m.is_var = False
            self.Q.is_set = False
            return True
        elif not i.m.is_var and i.h.is_var and not o.h.is_var:
            i.h.val_SI = o.h.val_SI - self.Q.val_SI / i.m.val_SI
            i.h.is_set = True
            i.h.is_var = False
            self.Q.is_set = False
            return True
        elif not i.m.is_var and not i.h.is_var and o.h.is_var:
            o.h.val_SI = self.Q.val_SI / i.m.val_SI + i.h.val_SI
            o.h.is_set = True
            o.h.is_var = False
            self.Q.is_set = False
            return True
        else:
            return False

    def energy_balance_hot_func_doc(self, label):
        r"""
        Equation for hot side deaerator energy balance.

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
            r'0 =\dot{m}_{in,2} \cdot \left(h_{out,1}-'
            r'h_{in,2}\right)-\dot{Q}')
        return generate_latex_eq(self, latex, label)

    def energy_balance_hot_deriv(self, increment_filter, k):
        r"""
        Partial derivatives for hot side deaerator energy balance.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        i = self.inl[1]
        o = self.outl[0]
        if self.is_variable(i.m):
            self.network.jacobian[k, i.m.J_col] = o.h.val_SI - i.h.val_SI
        if self.is_variable(i.h):
            self.network.jacobian[k, i.h.J_col] = -i.m.val_SI
        if self.is_variable(o.h):
            self.network.jacobian[k, o.h.J_col] = i.m.val_SI

    def energy_balance_hot_tensor(self, increment_filter, k):
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
        val : float
            Value of energy transfer :math:`\dot{E}`. This value is passed to
            :py:meth:`Aurora.components.component.Component.calc_bus_value`
            for value manipulation according to the specified characteristic
            line of the bus.

            .. math::

                \dot{E} = \dot{m}_{in,2} \cdot \left(
                h_{out,1} - h_{in,2} \right)
        """
        return self.inl[1].m.val_SI * (
            self.outl[0].h.val_SI - self.inl[1].h.val_SI
        )

    def bus_variables_columns(self, bus):
        pass

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
            r'\dot{m}_\mathrm{in,2} \cdot \left(h_\mathrm{out,1} - '
            r'h_\mathrm{in,2} \right)')

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
        if self.inl[1].m.is_var:
            self.network.jacobian[k, self.inl[1].m.J_col] -= self.numeric_deriv(f, 'm', self.inl[1], bus=bus)
        if self.inl[1].h.is_var:
            self.network.jacobian[k, self.inl[1].h.J_col] -= self.numeric_deriv(f, 'h', self.inl[1], bus=bus)
        if self.outl[0].h.is_var:
            self.network.jacobian[k, self.outl[0].h.J_col] -= self.numeric_deriv(f, 'h', self.outl[0], bus=bus)

    def bus_tensor(self, bus, increment_filter, k):
        pass

    def boundary_check(self):
        o0 = self.outl[0]  # liquid side
        try:
            for c in [o0]:
                if c.p.val_SI > c.calc_p_critical():
                    c.p.val_SI = c.calc_p_critical() * 0.99
                    self.boundary_rectify = True
                    logger.debug(f'The pressure of connection: {c.label} in {self.__class__.__name__}: {self.label} above the critical pressure, '
                                 f'adjusting to {c.p.val_SI}')
        except ValueError as e:
            raise ValueError(f"The boundary check error in {self.__class__.__name__}: {self.label}" + str(e))

    def bounds_p_generate(self):
        o0 = self.outl[0]
        o0.p.max_val = o0.calc_p_critical()

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        # component parameters
        self.Q.val_SI = self.inl[1].m.val_SI * (
            self.outl[0].h.val_SI - self.inl[1].h.val_SI
        )
        # pr and zeta
        for i in range(2):
            self.get_attr('pr' + str(i + 1)).val_SI = (
                    self.outl[0].p.val_SI / self.inl[i].p.val_SI)
            self.get_attr('dp' + str(i + 1)).val_SI = (
                    self.inl[i].p.val_SI - self.outl[0].p.val_SI)
            self.get_attr('zeta' + str(i + 1)).val_SI = self.calc_zeta(
                self.inl[i], self.outl[0]
            )

    def entropy_balance(self):
        r"""
        Calculate entropy balance of a deaerator.

        The allocation of the entropy streams due to heat exchanged and due to
        irreversibility is performed by solving for T on both sides of the deaerator:

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
            \text{S\_Q2}=\dot{m} \cdot \left(s_\mathrm{out*,1}-s_\mathrm{in*,2}
            \right)\\
            \text{S\_Qirr}=\text{S\_Q2} - \text{S\_Q1}\\
            \text{S\_irr1}=\dot{m} \cdot \left(s_\mathrm{out,1}-s_\mathrm{in,1}
            \right) - \text{S\_Q1}\\
            \text{S\_irr2}=\dot{m} \cdot \left(s_\mathrm{out,1}-s_\mathrm{in,2}
            \right) - \text{S\_Q2}\\
            \text{S\_irr}=\sum \dot{S}_\mathrm{irr}\\
            \text{T\_mQ1}=\frac{\dot{Q}}{\text{S\_Q1}}\\
            \text{T\_mQ2}=\frac{\dot{Q}}{\text{S\_Q2}}
        """
        self.S_irr = 0  # all non_reversible entropy transfer
        for i in range(2):
            inl = self.inl[i]
            out = self.outl[0]
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
        Calculate exergy balance of a deaerator.

        Parameters
        ----------
        T0 : float
            Ambient temperature T0 / K.

        Note
        ----
        .. math::

        """
        if all([c.T.val_SI > T0 for c in self.inl + self.outl]):
            self.E_P = self.inl[0].m.val_SI * self.outl[0].ex_therm - self.inl[0].Ex_therm
            self.E_F = self.inl[1].Ex_physical + self.inl[0].Ex_mech - self.outl[0].Ex_physical + self.inl[0].m.val_SI * self.outl[0].ex_therm

        elif all([c.T.val_SI <= T0 for c in self.inl + self.outl]):
            self.E_P = self.inl[1].m.val_SI * self.outl[0].ex_therm - self.inl[1].Ex_therm
            self.E_F = self.inl[1].Ex_mech + self.inl[0].Ex_physical - self.outl[0].Ex_physical + self.inl[1].m.val_SI * self.outl[0].ex_therm

        elif self.inl[1].T.val_SI > T0 >= self.inl[0].T.val_SI and self.outl[0].T.val_SI > T0:
            self.E_P = self.inl[0].m.val_SI * self.outl[0].ex_therm
            self.E_F = self.inl[1].Ex_physical + self.inl[0].Ex_physical - self.outl[0].Ex_physical + self.inl[0].m.val_SI * self.outl[0].ex_therm

        elif self.inl[1].T.val_SI > T0 > self.inl[0].T.val_SI and self.outl[0].T.val_SI < T0:
            self.E_P = self.inl[1].m.val_SI * self.outl[0].ex_therm
            self.E_F = self.inl[1].Ex_physical + self.inl[0].Ex_physical - self.outl[0].Ex_physical + self.inl[1].m.val_SI * self.outl[0].ex_therm

        else:
            self.E_P = np.nan
            self.E_F = self.inl[1].Ex_physical + self.inl[0].Ex_physical - self.outl[0].Ex_physical

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
                'isoline_value': self.inl[i].p.val_SI,
                'isoline_value_end': self.outl[0].p.val_SI,
                'starting_point_property': 'v',
                'starting_point_value': self.inl[i].vol.val_SI,
                'ending_point_property': 'v',
                'ending_point_value': self.outl[0].vol.val_SI
            } for i in range(2)}
