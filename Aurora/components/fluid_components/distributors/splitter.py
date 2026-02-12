# -*- coding: utf-8

"""Module of class Splitter.
"""

from Aurora.components.component import component_registry
from Aurora.components.fluid_components.distributors.base import NodeBase
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools import helpers as hlp


@component_registry
class Splitter(NodeBase):
    r"""
    Split up a mass flow in parts of identical enthalpy and fluid composition.

    **Mandatory Equations**

    - :py:meth:`AURORA.components.distributors.base.NodeBase.mass_flow_func`
    - :py:meth:`AURORA.components.distributors.base.NodeBase.pressure_equality_func`
    - :py:meth:`AURORA.components.distributors.splitter.Splitter.fluid_func`
    - :py:meth:`AURORA.components.distributors.splitter.Splitter.energy_balance_func`

    Inlets/Outlets

    - in1
    - specify number of outlets with :code:`num_out` (default value: 2)

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

    num_out : float, dict
        Number of outlets for this component, default value: 2.

    """

    @staticmethod
    def component():
        return 'splitter'

    @staticmethod
    def get_parameters():
        return {'num_out': dc_simple()}

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
                scale=ps['m']['scale'])
        }

    @staticmethod
    def inlets():
        return ['in1']

    def outlets(self):
        if self.num_out.is_set:
            return ['out' + str(i + 1) for i in range(self.num_out.val)]
        else:
            self.set_attr(num_out=2)
            return self.outlets()

    def propagate_wrapper_to_target(self, branch):
        branch["components"] += [self]
        for outconn in self.outl:  #
            branch["connections"] += [outconn]
            outconn.target.propagate_wrapper_to_target(branch)

    def simplify_pressure_enthalpy_mass_topology(self, inconn):
        if self.simplify_pressure_enthalpy_mass_topology_check():
            self.network.branches_components.append(self)
            # pressure
            conn_p_set_container = []
            p_value_set_container = []
            conn_p_shared_container = []
            # enthalpy
            conn_h_set_container = []
            h_value_set_container = []
            conn_h_shared_container = []
            for conn in self.inl + self.outl:
                if conn.p.is_set:
                    conn_p_set_container.append(conn)
                    p_value_set_container.append(conn.p.val)
                if conn.p.is_shared:
                    conn_p_shared_container.append(conn)
                if conn.h.is_set:
                    conn_h_set_container.append(conn)
                    h_value_set_container.append(conn.h.val)
                if conn.h.is_shared:
                    conn_h_shared_container.append(conn)
            # simplify pressure objective
            if conn_p_shared_container:
                for conn in set([c for c_shared in conn_p_shared_container for c in c_shared.p.shared_connection]
                                + self.inl + self.outl):
                    if not hasattr(conn, "_p_tmp"):
                        conn._p_tmp = conn.p
                    conn.p = inconn.p
            else:
                for outconn in self.outl:
                    outconn._p_tmp = outconn.p
                    outconn.p = inconn.p
            # set pressure value
            if conn_p_set_container:
                if len(set(p_value_set_container)) > 1:
                    msg = f"Has not set sole pressure value of branches of spliter component: {self.label}"
                    raise hlp.AURORANetworkError(msg)
                else:
                    # set p value
                    inconn.p.val = p_value_set_container[0]
                    inconn.p.is_set = True
            # simplify enthalpy objective
            if conn_h_shared_container:
                for conn in set([c for c_shared in conn_h_shared_container for c in c_shared.h.shared_connection]
                                + self.inl + self.outl):
                    if not hasattr(conn, "_h_tmp"):
                        conn._h_tmp = conn.h
                    conn.h = inconn.h
            else:
                for outconn in self.outl:
                    outconn._h_tmp = outconn.h
                    outconn.h = inconn.h
            # set enthalpy value
            if conn_h_set_container:
                if len(set(h_value_set_container)) > 1:
                    msg = f"Has not set sole enthalpy value of branches of spliter component: {self.label}"
                    raise hlp.AURORANetworkError(msg)
                else:
                    # set h value
                    inconn.h.val = h_value_set_container[0]
                    inconn.h.is_set = True
            #
            for conn in self.inl + self.outl:
                conn.p.is_shared = True
                conn.h.is_shared = True
                if conn not in conn.p.shared_connection:
                    conn.p.shared_connection.append(conn)
                if conn not in conn.h.shared_connection:
                    conn.h.shared_connection.append(conn)
            for outconn in self.outl:
                outconn.target.simplify_pressure_enthalpy_mass_topology(outconn)

    def spread_forward_pressure_values(self, inconn):
        for outconn in self.outl:
            if outconn not in self.network.connections_spread_pressure_container:
                self.network.connections_spread_pressure_container.append(outconn)
                outconn.target.spread_forward_pressure_values(outconn)
                outconn.spread_pressure_reference_check()
        return

    def spread_backward_pressure_values(self, outconni):
        inconn = self.inl[0]
        for outconn in self.outl:
            if outconn != outconni and outconn not in self.network.connections_spread_pressure_container:
                self.network.connections_spread_pressure_container.append(outconn)
                outconn.target.spread_forward_pressure_values(outconn)
                outconn.spread_pressure_reference_check()
        if inconn not in self.network.connections_spread_pressure_container:
            self.network.connections_spread_pressure_container.append(inconn)
            inconn.source.spread_backward_pressure_values(inconn)
            inconn.spread_pressure_reference_check()
        return

    def manage_fluid_equations(self):
        self.num_fluid_eqs = 0
        pass

    def energy_balance_func(self):
        r"""
        Calculate energy balance.

        Returns
        -------
        residual : list
            Residual value of energy balance.

            .. math::

                0 = h_{in} - h_{out,j} \;
                \forall j \in \mathrm{outlets}\\
        """
        residual = []
        for o in self.outl:
            residual += [self.inl[0].h.val_SI - o.h.val_SI]
        return residual

    def energy_balance_variables_columns(self):
        variables_columns = [[] for _ in range(len(self.outl))]
        for eq, o in enumerate(self.outl):  # eq: index
            if self.inl[0].h.is_var:
                variables_columns[eq].append(self.inl[0].h.J_col)
            if o.h.is_var:
                variables_columns[eq].append(o.h.J_col)
            variables_columns[eq].sort()
        return variables_columns

    def energy_balance_func_doc(self, label):
        r"""
        Calculate energy balance.

        Parameters
        ----------
        label : str
            Label for equation.
        """
        latex = r'0=h_{in}-h_{\mathrm{out,}j}\;\forall j \in\text{outlets}'
        return generate_latex_eq(self, latex, label)

    def energy_balance_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives for energy balance equation.

        Returns
        -------
        deriv : list
            Matrix of partial derivatives.
        """
        for eq, o in enumerate(self.outl):  # eq: index
            if self.inl[0].h.is_var:
                self.network.jacobian[k + eq, self.inl[0].h.J_col] = 1
            if o.h.is_var:
                self.network.jacobian[k + eq, o.h.J_col] = -1

    def energy_balance_tensor(self, increment_filter, k):
        pass

