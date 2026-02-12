# -*- coding: utf-8

"""Module of class Merge.
"""

import numpy as np

from Aurora.components.component import component_registry
from Aurora.components.fluid_components.distributors.base import NodeBase
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools.fluid_properties import s_mix_pT
from Aurora.tools import helpers as hlp


@component_registry
class Merge(NodeBase):
    r"""
    Class for merge points with multiple inflows and one outflow.

    **Mandatory Equations**

    - :py:meth:`AURORA.components.distributors.base.NodeBase.mass_flow_func`
    - :py:meth:`AURORA.components.distributors.base.NodeBase.pressure_equality_func`
    - :py:meth:`AURORA.components.distributors.merge.Merge.fluid_func`
    - :py:meth:`AURORA.components.distributors.merge.Merge.energy_balance_func`

    Inlets/Outlets

    - specify number of outlets with :code:`num_in` (default value: 2)
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

    num_in : float, dict
        Number of inlets for this component, default value: 2.

    """

    @staticmethod
    def component():
        return 'merge'

    @staticmethod
    def get_parameters():
        return {'num_in': dc_simple()}

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
                num_eq=1,  # num_m_eq
                scale=ps['m']['scale']),
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

    def inlets(self):
        if self.num_in.is_set:
            return ['in' + str(i + 1) for i in range(self.num_in.val)]
        else:
            self.set_attr(num_in=2)
            return self.inlets()

    @staticmethod
    def outlets():
        return ['out1']

    @staticmethod
    def is_branch_source():
        return True

    def start_branch(self):
        outconn = self.outl[0]
        branch = {
            "connections": [outconn],
            "components": [self, outconn.target],
            "subbranches": {}
        }
        outconn.target.propagate_to_target(branch)
        return {outconn.label: branch}

    def propagate_to_target(self, branch):
        return

    def propagate_wrapper_to_target(self, branch):
        if self in branch["components"]:
            return

        outconn = self.outl[0]
        branch["connections"] += [outconn]
        branch["components"] += [self]
        outconn.target.propagate_wrapper_to_target(branch)

    def simplify_pressure_enthalpy_mass_topology(self, inconni):
        if self.simplify_pressure_enthalpy_mass_topology_check():
            self.network.branches_components.append(self)
            outconn = self.outl[0]
            conn_p_set_container = []
            p_value_set_container = []
            conn_p_shared_container = []
            for conn in self.inl + self.outl:
                if conn.p.is_set:
                    conn_p_set_container.append(conn)
                    p_value_set_container.append(conn.p.val)
                if conn.p.is_shared:
                    conn_p_shared_container.append(conn)
            # simplify pressure objective
            if conn_p_shared_container:
                for conn in set([c for c_shared in conn_p_shared_container for c in c_shared.p.shared_connection]
                                + self.inl + self.outl):
                    if not hasattr(conn, "_p_tmp"):
                        conn._p_tmp = conn.p
                    conn.p = outconn.p
            else:
                for inconn in self.inl:
                    inconn._p_tmp = inconn.p
                    inconn.p = outconn.p
            # set pressure value
            if conn_p_set_container:
                if len(set(p_value_set_container)) > 1:
                    msg = f"Has not set sole pressure value of branches of merge component: {self.label}"
                    raise hlp.AURORANetworkError(msg)
                else:
                    # set p value
                    outconn.p.val = p_value_set_container[0]
                    outconn.p.is_set = True
            # posterior
            for conn in self.inl + self.outl:
                conn.p.is_shared = True
                if conn not in conn.p.shared_connection:
                    conn.p.shared_connection.append(conn)
            outconn.target.simplify_pressure_enthalpy_mass_topology(outconn)

    def spread_forward_pressure_values(self, inconni):
        outconn = self.outl[0]
        for inconn in self.inl:
            if inconn != inconni and inconn not in self.network.connections_spread_pressure_container:
                self.network.connections_spread_pressure_container.append(inconn)
                inconn.source.spread_backward_pressure_values(inconn)
                inconn.spread_pressure_reference_check()
        if outconn not in self.network.connections_spread_pressure_container:
            self.network.connections_spread_pressure_container.append(outconn)
            outconn.target.spread_forward_pressure_values(outconn)
            outconn.spread_pressure_reference_check()
        return

    def spread_backward_pressure_values(self, outconn):
        for inconn in self.inl:
            if inconn not in self.network.connections_spread_pressure_container:
                self.network.connections_spread_pressure_container.append(inconn)
                inconn.source.spread_backward_pressure_values(inconn)
                inconn.spread_pressure_reference_check()
        return

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
                scale=ps['m']['scale'] * ps['fluid']['scale']),
        }
        for key, equation in fluid_equations.items():
            if equation.num_eq > 0:
                equation.label = f"fluid composition equation: {key} of {self.__class__.__name__}: {self.label}"
                self.network.fluid_equations_module_container.append(equation)
                self.num_fluid_eqs += equation.num_eq

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
        num_eq = 0
        o = self.outl[0]
        variables_columns = [[] for _ in range(len(self.outl[0].fluid.val))]
        for fluid, x in self.outl[0].fluid.val.items():
            for i in self.inl:
                # if i.m.is_var:
                #     variables_columns[num_eq].append(i.m.J_col)
                if fluid in i.fluid.is_var:
                    variables_columns[num_eq].append(i.fluid.J_col[fluid])
            # if o.m.is_var:
            #     variables_columns[num_eq].append(o.m.J_col)
            if fluid in o.fluid.is_var:
                variables_columns[num_eq].append(o.fluid.J_col[fluid])
            variables_columns[num_eq].sort()
            num_eq += 1
        return variables_columns

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
        o = self.outl[0]
        for fluid, x in self.outl[0].fluid.val.items():
            for i in self.inl:
                if i.m.is_var and fluid in i.fluid.is_var:
                    self.network.tensor[i.m.J_col, i.fluid.J_col[fluid], k] = 1
                    self.network.tensor[i.fluid.J_col[fluid], i.m.J_col, k] = 1
            if o.m.is_var and fluid in o.fluid.is_var:
                self.network.tensor[o.m.J_col, o.fluid.J_col[fluid], k] = -1
                self.network.tensor[o.fluid.J_col[fluid], o.m.J_col, k] = -1
            k += 1

    def energy_balance_func(self):
        r"""
        Calculate energy balance.

        Returns
        -------
        residual : float
            Residual value of energy balance.

            .. math::

                0 = \sum_i \left(\dot{m}_{in,i} \cdot h_{in,i} \right) -
                \dot{m}_{out} \cdot h_{out}\\
                \forall i \in \text{inlets}
        """
        res = -self.outl[0].m.val_SI * self.outl[0].h.val_SI
        for i in self.inl:
            res += i.m.val_SI * i.h.val_SI
        return res

    def energy_balance_variables_columns(self):
        variables_columns1 = []
        for i in self.inl:
            if i.m.is_var:
                variables_columns1 += [i.m.J_col]
            if i.h.is_var:
                variables_columns1 += [i.h.J_col]
        o = self.outl[0]
        if o.m.is_var:
            variables_columns1 += [o.m.J_col]
        if o.h.is_var:
            variables_columns1 += [o.h.J_col]
        variables_columns1.sort()
        return [variables_columns1]

    def energy_balance_take_effect(self):
        pass

    def energy_balance_solve_isolated(self):
        if sum([1 if data.is_var else 0 for conn in self.inl + self.outl for data in [conn.m, conn.h]]) > 1:
            return False
        for inconn in self.inl:
            if inconn.m.is_var:
                inconn.m.val_SI = ((sum([outconn.m.val_SI * outconn.h.val_SI if not outconn.m.is_var and not outconn.h.is_var else 0 for outconn in self.outl]) -
                                   sum([inconn.m.val_SI * inconn.h.val_SI if not inconn.m.is_var and not inconn.h.is_var else 0 for inconn in self.inl])) /
                                   inconn.h.val_SI)
                inconn.m.is_set = True
                inconn.m.is_var = False
                return True
            if inconn.h.is_var:
                inconn.h.val_SI = ((sum([outconn.m.val_SI * outconn.h.val_SI if not outconn.m.is_var and not outconn.h.is_var else 0 for outconn in self.outl]) -
                                   sum([inconn.m.val_SI * inconn.h.val_SI if not inconn.m.is_var and not inconn.h.is_var else 0 for inconn in self.inl])) /
                                   inconn.m.val_SI)
                inconn.h.is_set = True
                inconn.h.is_var = False
                return True
        for outconn in self.outl:
            if outconn.m.is_var:
                outconn.m.val_SI = ((sum([inconn.m.val_SI * inconn.h.val_SI if not inconn.m.is_var and not inconn.h.is_var else 0 for inconn in self.inl]) -
                                    sum([outconn.m.val_SI * outconn.h.val_SI if not outconn.m.is_var and not outconn.h.is_var else 0 for outconn in self.outl])) /
                                    outconn.h.val_SI)
                outconn.m.is_set = True
                outconn.m.is_var = False
                return True
            if outconn.h.is_var:
                outconn.h.val_SI = ((sum([inconn.m.val_SI * inconn.h.val_SI if not inconn.m.is_var and not inconn.h.is_var else 0 for inconn in self.inl]) -
                                    sum([outconn.m.val_SI * outconn.h.val_SI if not outconn.m.is_var and not outconn.h.is_var else 0 for outconn in self.outl])) /
                                    outconn.m.val_SI)
                outconn.h.is_set = True
                outconn.h.is_var = False
                return True
        return False

    def energy_balance_func_doc(self, label):
        r"""
        Calculate energy balance.

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
            r'0=\sum_i\left(\dot{m}_{\mathrm{in,}i}\cdot h_{\mathrm{in,}i}'
            r'\right) - \dot{m}_\mathrm{out} \cdot h_\mathrm{out} '
            r'\; \forall i \in \text{inlets}'
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
        for i in self.inl:
            if i.m.is_var:
                self.network.jacobian[k, i.m.J_col] = i.h.val_SI
            if i.h.is_var:
                self.network.jacobian[k, i.h.J_col] = i.m.val_SI
        o = self.outl[0]
        if o.m.is_var:
            self.network.jacobian[k, o.m.J_col] = -o.h.val_SI
        if o.h.is_var:
            self.network.jacobian[k, o.h.J_col] = -o.m.val_SI

    def energy_balance_tensor(self, increment_filter, k):
        for i in self.inl:
            if i.m.is_var and i.h.is_var:
                self.network.tensor[i.m.J_col, i.h.J_col, k] = 1
                self.network.tensor[i.h.J_col, i.m.J_col, k] = 1
        o = self.outl[0]
        if o.m.is_var and o.h.is_var:
            self.network.tensor[o.m.J_col, o.h.J_col, k] = -1
            self.network.tensor[o.h.J_col, o.m.J_col, k] = -1

    def entropy_balance(self):
        r"""
        Calculate entropy balance of a merge.

        Note
        ----
        A definition of reference points is included for compensation of
        differences in zero point definitions of different fluid compositions.

        - Reference temperature: 298.15 K.
        - Reference pressure: 1 bar.

        .. math::

            \dot{S}_\mathrm{irr}= \dot{m}_\mathrm{out} \cdot
            \left( s_\mathrm{out} - s_\mathrm{out,ref} \right)
            - \sum_{i} \dot{m}_{\mathrm{in,}i} \cdot
            \left( s_{\mathrm{in,}i} - s_{\mathrm{in,ref,}i} \right)\\
        """
        T_ref = 298.15
        p_ref = 1e5
        o = self.outl[0]
        self.S_irr = o.m.val_SI * (
            o.s.val_SI - s_mix_pT(p_ref, T_ref, o.fluid_data, o.mixing_rule)
        )
        for i in self.inl:
            self.S_irr -= i.m.val_SI * (
                i.s.val_SI - s_mix_pT(p_ref, T_ref, i.fluid_data, i.mixing_rule)
            )

    def exergy_balance(self, T0):
        r"""
        Calculate exergy balance of a merge.

        Parameters
        ----------
        T0 : float
            Ambient temperature T0 / K.

        Note
        ----
        Please note, that the exergy balance accounts for physical exergy only.

        .. math ::

            \dot{E}_\mathrm{P} =
            \begin{cases}
            \begin{cases}
            \sum_i \dot{m}_i \cdot \left(e_\mathrm{out}^\mathrm{PH} -
            e_{\mathrm{in,}i}^\mathrm{PH}\right)
            & T_{\mathrm{in,}i} < T_\mathrm{out} \text{ \& }
            T_{\mathrm{in,}i} \geq T_0 \\
            \sum_i \dot{m}_i \cdot e_\mathrm{out}^\mathrm{PH}
            & T_{\mathrm{in,}i} < T_\mathrm{out} \text{ \& }
            T_{\mathrm{in,}i} < T_0 \\
            \end{cases} & T_\mathrm{out} > T_0\\

            \text{not defined (nan)} & T_\mathrm{out} = T_0\\

            \begin{cases}
            \sum_i \dot{m}_i \cdot e_\mathrm{out}^\mathrm{PH}
            & T_{\mathrm{in,}i} > T_\mathrm{out} \text{ \& }
            T_{\mathrm{in,}i} \geq T_0 \\
            \sum_i \dot{m}_i \cdot \left(e_\mathrm{out}^\mathrm{PH} -
            e_{\mathrm{in,}i}^\mathrm{PH}\right)
            & T_{\mathrm{in,}i} > T_\mathrm{out} \text{ \& }
            T_{\mathrm{in,}i} < T_0 \\
            \end{cases} & T_\mathrm{out} < T_0\\
            \end{cases}

            \dot{E}_\mathrm{F} =
            \begin{cases}
            \begin{cases}
            \sum_i \dot{m}_i \cdot \left(e_{\mathrm{in,}i}^\mathrm{PH} -
            e_\mathrm{out}^\mathrm{PH}\right)
            & T_{\mathrm{in,}i} > T_\mathrm{out} \\
            \sum_i \dot{E}_{\mathrm{in,}i}^\mathrm{PH}
            & T_{\mathrm{in,}i} < T_\mathrm{out} \text{ \& }
            T_{\mathrm{in,}i} < T_0 \\
            \end{cases} & T_\mathrm{out} > T_0\\

            \sum_i \dot{E}_{\mathrm{in,}i}^\mathrm{PH} & T_\mathrm{out} = T_0\\

            \begin{cases}
            \sum_i \dot{E}_{\mathrm{in,}i}^\mathrm{PH}
            & T_{\mathrm{in,}i} > T_\mathrm{out} \text{ \& }
            T_{\mathrm{in,}i} \geq T_0 \\
            \sum_i \dot{m}_i \cdot \left(e_{\mathrm{in,}i}^\mathrm{PH} -
            e_\mathrm{out}^\mathrm{PH}\right)
            & T_{\mathrm{in,}i} < T_\mathrm{out} \\
            \end{cases} & T_\mathrm{out} < T_0\\
            \end{cases}

            \forall i \in \text{merge inlets}

            \dot{E}_\mathrm{bus} = \text{not defined (nan)}
        """
        self.E_P = 0
        self.E_F = 0
        if self.outl[0].T.val_SI > T0:
            for i in self.inl:
                if i.T.val_SI < self.outl[0].T.val_SI:
                    if i.T.val_SI >= T0:
                        self.E_P += i.m.val_SI * (
                            self.outl[0].ex_physical - i.ex_physical)
                    else:
                        self.E_P += i.m.val_SI * self.outl[0].ex_physical
                        self.E_F += i.Ex_physical
                else:
                    self.E_F += i.m.val_SI * (
                        i.ex_physical - self.outl[0].ex_physical)
        elif self.outl[0].T.val_SI == T0:
            self.E_P = np.nan
            for i in self.inl:
                self.E_F += i.Ex_physical
        else:
            for i in self.inl:
                if i.T.val_SI > self.outl[0].T.val_SI:
                    if i.T.val_SI >= T0:
                        self.E_P += i.m.val_SI * self.outl[0].ex_physical
                        self.E_F += i.Ex_physical
                    else:
                        self.E_P += i.m.val_SI * (
                            self.outl[0].ex_physical - i.ex_physical)
                else:
                    self.E_F += i.m.val_SI * (
                        i.ex_physical - self.outl[0].ex_physical)

        self.E_bus = {
            "chemical": np.nan, "physical": np.nan, "massless": np.nan
        }
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
                'isoline_value_end': self.outl[0].p.val,
                'starting_point_property': 'v',
                'starting_point_value': self.inl[i].vol.val,
                'ending_point_property': 'v',
                'ending_point_value': self.outl[0].vol.val
            } for i in range(self.num_i)}
