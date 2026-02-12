# -*- coding: utf-8

"""Module of class NodeBase.
"""

from Aurora.components.component import Component
from Aurora.components.fluid_components.fluid_component import FluidComponent
from Aurora.components.component import component_registry
from Aurora.tools.document_models import generate_latex_eq


@component_registry
class NodeBase(FluidComponent):
    """Class NodeBase is parent class for all components of submodule distributors."""

    @staticmethod
    def get_bypass_constraints():
        return {}

    def propagate_to_target(self, branch):
        for outconn in self.outl:
            subbranch = {
                "connections": [outconn],
                "components": [self, outconn.target],
                "subbranches": {}
            }  # subbranch: the lower layer branch
            outconn.target.propagate_to_target(subbranch)  # generate the subordinate branch
            branch["subbranches"][outconn.label] = subbranch

    def simplify_pressure_enthalpy_mass_topology_check(self):
        if self in self.network.branches_components:
            return False
        else:
            return True

    def looking_forward_pressure_values(self, inconni):
        if inconni not in self.network.connections_looking_pressure_container:
            self.network.connections_looking_pressure_container.append(inconni)
            for outconn in self.outl:
                outconn.target.looking_forward_pressure_values(outconn)
                outconn.spread_pressure_reference_check()
            for inconn in self.inl:
                if inconn != inconni:
                    inconn.source.looking_backward_pressure_values(inconn)
                    inconn.spread_pressure_reference_check()
        return

    def looking_backward_pressure_values(self, outconni):
        if outconni not in self.network.connections_looking_pressure_container:
            self.network.connections_looking_pressure_container.append(outconni)
            for inconn in self.inl:
                inconn.source.looking_backward_pressure_values(inconn)
                inconn.spread_pressure_reference_check()
            for outconn in self.outl:
                if outconn != outconni:
                    outconn.target.looking_forward_pressure_values(outconn)
                    outconn.spread_pressure_reference_check()
        return

    def looking_for_pressure_set_boundary(self, inconni):
        if inconni not in self.network.connections_pressure_boundary_container:
            self.network.connections_pressure_boundary_container.append(inconni)
            for outconn in self.outl:
                outconn.target.looking_for_pressure_set_boundary(outconn)
        return

    def spread_forward_pressure_initial(self, inconni):
        for outconn in self.outl:
            if outconn not in self.network.connections_pressure_initial_container:
                self.network.connections_pressure_initial_container.append(outconn)
                outconn.target.spread_forward_pressure_initial(outconn)
        for inconn in self.inl:
            if inconn != inconni and inconn not in self.network.connections_pressure_initial_container:
                self.network.connections_pressure_initial_container.append(inconn)
                inconn.source.spread_backward_pressure_initial(inconn)
        return

    def spread_backward_pressure_initial(self, outconni):
        for inconn in self.inl:
            if inconn not in self.network.connections_pressure_initial_container:
                self.network.connections_pressure_initial_container.append(inconn)
                inconn.source.spread_backward_pressure_initial(inconn)
        for outconn in self.outl:
            if outconn != outconni and outconn not in self.network.connections_pressure_initial_container:
                self.network.connections_pressure_initial_container.append(outconn)
                outconn.target.spread_forward_pressure_initial(outconn)
        return

    @staticmethod
    def is_fluid_composition_component():
        return True

    def mass_flow_func(self):
        r"""
        Calculate the residual value for mass flow balance equation.

        Returns
        -------
        res : float
            Residual value of equation.

            .. math::

                0 = \sum \dot{m}_{in,i} - \sum \dot{m}_{out,j} \;
                \forall i \in inlets, \forall j \in outlets
        """
        res = 0
        for i in self.inl:
            res += i.m.val_SI
        for o in self.outl:
            res -= o.m.val_SI
        return res

    def mass_flow_variables_columns(self):
        variables_columns1 = []
        variables_columns1 += [data.J_col for c in self.inl + self.outl for data in [c.m] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def mass_flow_take_effect(self):
        pass

    def mass_flow_solve_isolated(self):
        if sum([1 if conn.m.is_var else 0 for conn in self.inl + self.outl]) > 1:
            return False
        for inconn in self.inl:
            if inconn.m.is_var:
                inconn.m.val_SI = (sum([outconn.m.val_SI if not outconn.m.is_var else 0 for outconn in self.outl]) -
                                   sum([inconn.m.val_SI if not inconn.m.is_var else 0 for inconn in self.inl]))
                inconn.m.is_set = True
                inconn.m.is_var = False
                return True
        for outconn in self.outl:
            if outconn.m.is_var:
                outconn.m.val_SI = (sum([inconn.m.val_SI if not inconn.m.is_var else 0 for inconn in self.inl]) -
                                    sum([outconn.m.val_SI if not outconn.m.is_var else 0 for outconn in self.outl]))
                outconn.m.is_set = True
                outconn.m.is_var = False
                return True
        return False

    def mass_flow_func_doc(self, label):
        r"""
        Calculate the residual value for mass flow balance equation.

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
            r'0 =\sum\dot{m}_{\mathrm{in},i}-\sum\dot{m}_{\mathrm{out},j}'
            r'\;\forall i \in \text{inlets}, \forall j \in \text{outlets}')
        return generate_latex_eq(self, latex, label)

    def mass_flow_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives for mass flow equation.

        Returns
        -------
        deriv : list
            Matrix with partial derivatives for the fluid equations.
        """
        for i in self.inl:
            if i.m.is_var:
                self.network.jacobian[k, i.m.J_col] = 1
        for o in self.outl:
            if o.m.is_var:
                self.network.jacobian[k, o.m.J_col] = -1

    def mass_flow_tensor(self, increment_filter, k):
        pass

    def pressure_equality_func(self):
        r"""
        Calculate the residual values of pressure equality equations.

        Returns
        -------
        residual : list
            Vector with residual value for pressure equality equations.

            .. math::

                0 = p_{in,1} - p_{in,i}\forall i \in \text{inlets > 1}\\
                0 = p_{in,1} - p_{out,j}\forall j \in \text{outlets}
        """
        residual = []
        inl = []
        if self.num_i > 1:
            inl = self.inl[1:]
        for c in inl + self.outl:
            residual += [self.inl[0].p.val_SI - c.p.val_SI]
        return residual

    def pressure_equality_take_effect(self):
        pass

    def pressure_equality_solve_isolated(self):
        return False

    def pressure_equality_variables_columns(self):
        if self.num_i > 1:
            conns = self.inl[1:] + self.outl
        else:
            conns = self.outl
        variables_columns = [[] for _ in range(len(conns))]
        for eq, o in enumerate(conns):
            if self.inl[0].p.is_var:
                variables_columns[eq].append(self.inl[0].p.J_col)
            if o.p.is_var:
                variables_columns[eq].append(o.p.J_col)
        return variables_columns

    def pressure_equality_func_doc(self, label):
        r"""
        Calculate the residual values of pressure equality equations.

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
            r'0 = p_\mathrm{in,1} - p_{\mathrm{in,}i} '
            r'& \; \forall i \in \text{inlets} \setminus '
            r'\left\lbrace 1\right\rbrace\\' + '\n'
            r'0 = p_\mathrm{in,1} - p_{\mathrm{out,}j} '
            r'& \; \forall j \in \text{outlets}\\' + '\n'
            r'\end{split}'
        )
        return generate_latex_eq(self, latex, label)

    def pressure_equality_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives for all pressure equations.

        Returns
        -------
        deriv : ndarray
            Matrix with partial derivatives for the fluid equations.
        """
        if self.num_i > 1:
            conns = self.inl[1:] + self.outl
        else:
            conns = self.outl

        for eq, o in enumerate(conns):
            if self.inl[0].p.is_var:
                self.network.jacobian[k + eq, self.inl[0].p.J_col] = 1
            if o.p.is_var:
                self.network.jacobian[k + eq, o.p.J_col] = -1

    def pressure_equality_tensor(self, increment_filter, k):
        pass

    @staticmethod
    def initialise_source(c, key):
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

            .. math::

                val = \begin{cases}
                10^5 & \text{key = 'p'}\\
                5 \cdot 10^5 & \text{key = 'h'}
                \end{cases}
        """
        if key == 'p':
            return 1e5
        elif key == 'h':
            return 5e5

    @staticmethod
    def initialise_target(c, key):
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

            .. math::

                val = \begin{cases}
                10^5 & \text{key = 'p'}\\
                5 \cdot 10^5 & \text{key = 'h'}
                \end{cases}
        """
        if key == 'p':
            return 1e5
        elif key == 'h':
            return 5e5

