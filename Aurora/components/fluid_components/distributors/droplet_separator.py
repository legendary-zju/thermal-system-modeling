# -*- coding: utf-8

"""Module of class DropletSeparator.
"""

from Aurora.components.component import component_registry
from Aurora.components.fluid_components.distributors.base import NodeBase
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.fluid_properties import p_mix_hQ
from Aurora.tools.fluid_properties import d2h_mix_d2p_Q
from Aurora.tools.fluid_properties import h_mix_pQ
from Aurora.tools import helpers as hlp
from Aurora.tools import logger


@component_registry
class DropletSeparator(NodeBase):
    r"""
    Separate liquid phase from gas phase of a single fluid.

    This component is the parent component of the Drum.

    **Mandatory Equations**

    - :py:meth:`AURORA.components.distributors.base.NodeBase.mass_flow_func`
    - :py:meth:`AURORA.components.distributors.droplet_separator.DropletSeparator.fluid_func`
    - :py:meth:`AURORA.components.distributors.droplet_separator.DropletSeparator.energy_balance_func`
    - :py:meth:`AURORA.components.distributors.droplet_separator.DropletSeparator.outlet_states_func`

    Inlets/Outlets

    - in1
    - out1, out2 (index 1: saturated liquid, index 2: saturated gas)

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

    """

    @staticmethod
    def component():
        return 'droplet separator'

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
                scale=ps['m']['scale'] * ps['h']['scale']),
            'outlet_constraints1': dc_cons(
                func=self.outlet_states_func1,
                variables_columns=self.outlet_states_variables_columns1,
                solve_isolated=self.outlet_states_solve_isolated1,
                deriv=self.outlet_states_deriv1,
                constant_deriv=False,
                latex=self.outlet_states_func_doc,
                num_eq=1,
                scale=ps['h']['scale']),
            'outlet_constraints2': dc_cons(
                func=self.outlet_states_func2,
                variables_columns=self.outlet_states_variables_columns2,
                solve_isolated=self.outlet_states_solve_isolated2,
                deriv=self.outlet_states_deriv2,
                constant_deriv=False,
                latex=self.outlet_states_func_doc,
                num_eq=1,
                scale=ps['h']['scale'])
        }

    @staticmethod
    def inlets():
        return ['in1']

    @staticmethod
    def outlets():
        return ['out1', 'out2']

    def propagate_wrapper_to_target(self, branch):
        if self in branch["components"]:
            return

        for outconn in self.outl:
            branch["connections"] += [outconn]
            branch["components"] += [self]
            outconn.target.propagate_wrapper_to_target(branch)

    def simplify_pressure_enthalpy_mass_topology(self, inconn):
        if self.simplify_pressure_enthalpy_mass_topology_check():
            self.network.branches_components.append(self)
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
                    conn.p = inconn.p
            else:
                for outconn in self.outl:
                    outconn._p_tmp = outconn.p
                    outconn.p = inconn.p
            # set pressure value
            if conn_p_set_container:
                if len(set(p_value_set_container)) > 1:
                    msg = f"Has not set sole pressure value of branches of droplet separator component: {self.label}"
                    raise hlp.AURORANetworkError(msg)
                else:
                    # set p value
                    inconn.p.val = p_value_set_container[0]
                    inconn.p.is_set = True
                    inconn.p.is_var = False
                    for outconn in self.outl:
                        outconn.p.val = p_value_set_container[0]
                        outconn.p.is_set = True
                        outconn.p.is_var = False

            for conn in self.inl + self.outl:
                conn.p.is_shared = True
                if conn not in conn.p.shared_connection:
                    conn.p.shared_connection.append(conn)
            for outconn in self.outl:
                outconn.target.simplify_pressure_enthalpy_mass_topology(outconn)

    def simplify_pressure_enthalpy_mass_topology_check(self):
        if self in self.network.branches_components:
            return False
        else:
            return True

    def spread_forward_pressure_values(self, inconn):
        for outconn in self.outl:
            if outconn not in self.network.connections_spread_pressure_container:
                self.network.connections_spread_pressure_container.append(outconn)
                outconn.target.spread_forward_pressure_values(outconn)
                outconn.spread_pressure_reference_check()
        return

    def spread_backward_pressure_values(self, outconni):
        for outconn in self.outl:
            if outconn != outconni and outconn not in self.network.connections_spread_pressure_container:
                self.network.connections_spread_pressure_container.append(outconn)
                outconn.target.spread_forward_pressure_values(outconn)
                outconn.spread_pressure_reference_check()
        inconn = self.inl[0]
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
            res = 0
            for o in self.outl:
                res -= o.fluid.val[fluid] * o.m.val_SI
            for i in self.inl:
                res += i.fluid.val[fluid] * i.m.val_SI
            residual += [res]
        return residual

    def fluid_variables_columns(self):
        num_eq = 0
        variables_columns = [[] for _ in range(len(self.outl[0].fluid.val))]
        for fluid, x in self.outl[0].fluid.val.items():
            for i in self.inl:
                if fluid in i.fluid.is_var:
                    variables_columns[num_eq].append(i.fluid.J_col[fluid])
            for o in self.outl:
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
        for fluid, x in self.outl[0].fluid.val.items():
            for i in self.inl:
                if fluid in i.fluid.is_var:
                    self.network.fluid_jacobian[k, i.fluid.J_col[fluid]] = i.m.val_SI
            for o in self.outl:
                if fluid in o.fluid.is_var:
                    self.network.fluid_jacobian[k, o.fluid.J_col[fluid]] = -o.m.val_SI
            k += 1  # k-th fluid equation

    def fluid_tensor(self, increment_filter, k):
        pass

    def energy_balance_func(self):
        r"""
        Calculate energy balance.

        Returns
        -------
        residual : float
            Residual value of energy balance.

            .. math::

                0 = \sum_i \left(\dot{m}_{in,i} \cdot h_{in,i} \right) -
                \sum_j \left(\dot{m}_{out,j} \cdot h_{out,j} \right)\\
                \forall i \in \text{inlets} \; \forall j \in \text{outlets}
        """
        res = 0
        for i in self.inl:
            res += i.m.val_SI * i.h.val_SI
        for o in self.outl:
            res -= o.m.val_SI * o.h.val_SI
        return res

    def energy_balance_variables_columns(self):
        variables_columns1 = []
        variables_columns1 += [data.J_col for c in self.inl + self.outl for data in [c.m, c.h] if data.is_var]  # [c.m, c.h]
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
        return True

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
            r'\right) - \sum_j \left(\dot{m}_{\mathrm{out,}j} \cdot '
            r'h_{\mathrm{out,}j} \right) \; \forall i \in \text{inlets} \;'
            r'\forall j \in \text{outlets}'
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

        for o in self.outl:
            if o.m.is_var:
                self.network.jacobian[k, o.m.J_col] = -o.h.val_SI
            if o.h.is_var:
                self.network.jacobian[k, o.h.J_col] = -o.m.val_SI

    def energy_balance_tensor(self, increment_filter, k):
        for i in self.inl:
            if i.m.is_var and i.h.is_var:
                self.network.tensor[i.m.J_col, i.h.J_col, k] = 1
                self.network.tensor[i.h.J_col, i.m.J_col, k] = 1
        for o in self.outl:
            if o.m.is_var and o.h.is_var:
                self.network.tensor[o.m.J_col, o.h.J_col, k] = 1
                self.network.tensor[o.h.J_col, o.m.J_col, k] = 1

    def outlet_states_func_doc(self, label):
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
            r'\begin{split}' + '\n'
            r'0 =&h_\mathrm{out,1} -h\left(p_\mathrm{out,1}, x=0\right)\\'
            r'0 =&h_\mathrm{out,2} -h\left(p_\mathrm{out,2}, x=1\right)\\'
            r'\end{split}'
        )
        return generate_latex_eq(self, latex, label)

    def outlet_states_tensor(self, increment_filter, k):
        o0 = self.outl[0]
        o1 = self.outl[1]
        if o0.p.is_var:
            self.network.tensor[o0.p.J_col, o0.p.J_col, k] = d2h_mix_d2p_Q(o0.p.val_SI, 0, o0.fluid_data)
        if o1.p.is_var:
            self.network.tensor[o1.p.J_col, o1.p.J_col, k + 1] = d2h_mix_d2p_Q(o1.p.val_SI, 1, o1.fluid_data)

    def outlet_states_func1(self):
        r"""
        Calculate energy balance.

        Returns
        -------
        residual
        """
        o0 = self.outl[0]  # liquid side
        return h_mix_pQ(o0.p.val_SI, 0, o0.fluid_data) - o0.h.val_SI

    def outlet_states_variables_columns1(self):
        o0 = self.outl[0]
        variables_columns1 = []
        # if o0.p.is_var:
        #     variables_columns1 += [o0.p.J_col]
        if o0.h.is_var:
            variables_columns1 += [o0.h.J_col]
        return [variables_columns1]

    def outlet_states_take_effect1(self):
        pass

    def outlet_states_solve_isolated1(self):
        o0 = self.outl[0]
        if not o0.p.is_var and not o0.h.is_var:
            return True
        elif not o0.p.is_var and o0.h.is_var:
            o0.h.val_SI = h_mix_pQ(o0.p.val_SI, 0, o0.fluid_data)
            o0.h.is_set = True
            o0.h.is_var = False
            return True
        elif o0.p.is_var and not o0.h.is_var:
            o0.p.val_SI = p_mix_hQ(o0.h.val_SI, 0, o0.fluid_data)
            o0.p.is_set = True
            o0.p.is_var = False
            return True
        return False

    def outlet_states_deriv1(self, increment_filter, k):
        r"""
        Calculate partial derivatives of outlet states.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        o0 = self.outl[0]
        # if o0.p.is_var:
        #     self.network.jacobian[k, o0.p.J_col] = (
        #         dh_mix_dpQ(o0.p.val_SI, 0, o0.fluid_data)
        #     )
        if o0.h.is_var:
            self.network.jacobian[k, o0.h.J_col] = -1

    def outlet_states_func2(self):
        r"""
        Calculate energy balance.

        Returns
        -------
        residual
        """
        o1 = self.outl[1]  # vapour side
        return h_mix_pQ(o1.p.val_SI, 1, o1.fluid_data) - o1.h.val_SI

    def outlet_states_variables_columns2(self):
        o1 = self.outl[1]
        variables_columns2 = []
        # if o1.p.is_var:
        #     variables_columns2 += [o1.p.J_col]
        if o1.h.is_var:
            variables_columns2 += [o1.h.J_col]
        return [variables_columns2]

    def outlet_states_take_effect2(self):
        pass

    def outlet_states_solve_isolated2(self):
        o1 = self.outl[1]
        if not o1.p.is_var and not o1.h.is_var:
            return True
        elif not o1.p.is_var and o1.h.is_var:
            o1.h.val_SI = h_mix_pQ(o1.p.val_SI, 1, o1.fluid_data)
            o1.h.is_set = True
            o1.h.is_var = False
            return True
        elif o1.p.is_var and not o1.h.is_var:
            o1.p.val_SI = p_mix_hQ(o1.h.val_SI, 1, o1.fluid_data)
            o1.p.is_set = True
            o1.p.is_var = False
            return True
        return False

    def outlet_states_deriv2(self, increment_filter, k):
        r"""
        Calculate partial derivatives of outlet states.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        o1 = self.outl[1]
        # if o1.p.is_var:
        #     self.network.jacobian[k, o1.p.J_col] = (
        #         dh_mix_dpQ(o1.p.val_SI, 1, o1.fluid_data)
        #     )
        if o1.h.is_var:
            self.network.jacobian[k, o1.h.J_col] = -1

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
                10^6 & \text{key = 'p'}\\
                h\left(p, x=1 \right) & \text{key = 'h' at outlet 1}\\
                h\left(p, x=0 \right) & \text{key = 'h' at outlet 2}
                \end{cases}
        """
        if key == 'p':  # 10e5  !!!
            return 10e5
        elif key == 'h':
            if c.source_id == 'out1':
                return h_mix_pQ(c.p.val_SI, 0, c.fluid_data)
            else:
                return h_mix_pQ(c.p.val_SI, 1, c.fluid_data)

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
                10^6 & \text{key = 'p'}\\
                h\left(p, x=0.5 \right) & \text{key = 'h' at inlet 1}
                \end{cases}
        """
        if key == 'p':
            return 10e5
        elif key == 'h':
            return h_mix_pQ(c.p.val_SI, 0.5, c.fluid_data)

    def boundary_check(self):
        o0 = self.outl[0]  # liquid side
        o1 = self.outl[1]  # vapour side
        try:
            for c in [o0, o1]:
                if c.p.val_SI > c.calc_p_critical():
                    c.p.val_SI = c.calc_p_critical() * 0.99
                    self.boundary_rectify = True
                    logger.debug(f'The pressure of connection: {c.label} in {self.__class__.__name__}: {self.label} above the critical pressure, '
                                 f'adjusting to {c.p.val_SI}')
        except ValueError as e:
            raise ValueError(f"The boundary check error in {self.__class__.__name__}: {self.label}" + str(e))

    def bounds_p_generate(self):
        o0 = self.outl[0]
        o1 = self.outl[1]
        o0.p.max_val = o0.calc_p_critical()
        o1.p.max_val = o1.calc_p_critical()

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
                'isoline_value': self.inl[0].p.val,
                'isoline_value_end': self.outl[i].p.val,
                'starting_point_property': 'v',
                'starting_point_value': self.inl[0].vol.val,
                'ending_point_property': 'v',
                'ending_point_value': self.outl[i].vol.val
            } for i in range(2)}
