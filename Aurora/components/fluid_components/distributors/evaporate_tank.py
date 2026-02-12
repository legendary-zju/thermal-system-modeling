# -*- coding: utf-8

"""Module for class EvaporateTank.
"""

import numpy as np

from Aurora.components.component import component_registry
from Aurora.components.fluid_components.distributors.drum import Drum
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.fluid_properties import h_mix_pQ
from Aurora.tools.helpers import AURORANetworkError
from Aurora.tools import helpers as hlp
from Aurora.tools import logger


@component_registry
class EvaporateTank(Drum):
    r"""
    A tank separates saturated gas and saturated liquid.

    **Mandatory Equations**

    - :py:meth:`AURORA.components.distributors.base.NodeBase.mass_flow_func`
    - :py:meth:`AURORA.components.distributors.drum.Drum.Ki_func`
    - :py:meth:`AURORA.components.distributors.droplet_separator.DropletSeparator.fluid_func`
    - :py:meth:`AURORA.components.distributors.droplet_separator.DropletSeparator.energy_balance_func`
    - :py:meth:`AURORA.components.distributors.droplet_separator.DropletSeparator.outlet_states_func`

    Inlets/Outlets

    - in1, in2 (index 1: from economiser, index 2: two phase from evaporator)
    - out1, out2, out3 (index 1: saturated liquid, index 2: saturated gas, index 3: saturated liquid drained)

    Parameters
    ----------
    Ki: float
         The ratio of mass flow between inner cycle and outer cycle.

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

    Note
    ----
    If you are using a evaporate tank in a network with multiple fluids, it is likely
    the fluid propagation causes trouble. If this is the case, try to
    specify the fluid composition at another connection of your network.

    The complex topological constructure of inner cycle be supported.

    """

    @staticmethod
    def component():
        return 'evaporate tank'

    @staticmethod
    def inlets():
        return ['in1', 'in2']

    @staticmethod
    def outlets():
        return ['out1', 'out2', 'out3']

    def propagate_wrapper_to_target(self, branch):
        return super().propagate_wrapper_to_target(branch)

    def propagate_to_target(self, branch):
        # two phase
        if branch["connections"][-1].target_id == "in2":
            return
        # saturated liquid, saturated vapour, saturated liquid drained
        for outconn in self.outl:
            subbranch = {
                "connections": [outconn],
                "components": [self, outconn.target],
                "subbranches": {}
            }
            outconn.target.propagate_to_target(subbranch)
            branch["subbranches"][outconn.label] = subbranch

    def simplify_pressure_enthalpy_mass_topology(self, inconni):
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
            # properties information
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
                    conn.p = inconni.p
            else:
                for inconn in self.inl:
                    inconn._p_tmp = inconn.p
                    inconn.p = inconni.p
                for outconn in self.outl:
                    outconn._p_tmp = outconn.p
                    outconn.p = inconni.p
            # set pressure value
            if conn_p_set_container:
                if len(set(p_value_set_container)) > 1:
                    msg = f"Has not set sole pressure value of branches of drum component: {self.label}"
                    raise hlp.AURORANetworkError(msg)
                else:
                    # set p value
                    inconni.p.val = p_value_set_container[0]
                    inconni.p.is_set = True
                    inconni.p.is_var = False
            # simplify enthalpy objective
            outconn_main = self.outl[0]
            if conn_h_shared_container:
                for conn in set([c for c_shared in conn_h_shared_container for c in c_shared.h.shared_connection]
                                + [self.outl[0], self.outl[2]]):
                    if not hasattr(conn, "_h_tmp"):
                        conn._h_tmp = conn.h
                    conn.h = outconn_main.h
            else:
                for outconn in [self.outl[0], self.outl[2]]:
                    outconn._h_tmp = outconn.h
                    outconn.h = outconn_main.h
            # set enthalpy value
            if conn_h_set_container:
                if len(set(h_value_set_container)) > 1:
                    msg = f"Has not set sole enthalpy value of branches of deaerator component: {self.label}"
                    raise AURORANetworkError(msg)
                else:
                    # set h value
                    outconn_main.h.val = h_value_set_container[0]
                    outconn_main.h.is_set = True
                    outconn_main.h.is_var = False
            # posterior
            for conn in self.inl + self.outl:
                conn.p.is_shared = True
                if conn not in conn.p.shared_connection:
                    conn.p.shared_connection.append(conn)
            for outconn in [self.outl[0], self.outl[2]]:
                outconn.h.is_shared = True
                if outconn not in outconn.h.shared_connection:
                    outconn.h.shared_connection.append(outconn)
            for outconn in self.outl:
                outconn.target.simplify_pressure_enthalpy_mass_topology(outconn)

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

    def summarize_equations(self):
        """
        The difference of mass flow objective between inner cycle and outer cycle be supported.
        This method will inherit from Component.

        :return:
        """
        # generate equation group, jump parent class.
        super(self.__class__.__bases__[0], self).summarize_equations()

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
        if self.inl[1].m == self.outl[0].m:
            return self.inl[0].m.val_SI - self.outl[1].m.val_SI - self.outl[2].m.val_SI
        else:
            res = 0
            for i in self.inl:
                res += i.m.val_SI
            for o in self.outl:
                res -= o.m.val_SI
            return res

    def mass_flow_variables_columns(self):
        variables_columns1 = []
        if self.inl[1].m == self.outl[0].m:
            if self.inl[0].m.is_var:
                variables_columns1.append(self.inl[0].m.J_col)
            if self.outl[1].m.is_var:
                variables_columns1.append(self.outl[1].m.J_col)
            if self.outl[2].m.is_var:
                variables_columns1.append(self.outl[2].m.J_col)
        else:
            for i in self.inl:
                if i.m.is_var:
                    variables_columns1.append(i.m.J_col)
            for o in self.outl:
                if o.m.is_var:
                    variables_columns1.append(o.m.J_col)
        variables_columns1.sort()
        return [variables_columns1]

    def mass_flow_take_effect(self):
        pass

    def mass_flow_solve_isolated(self):
        if self.inl[1].m == self.outl[0].m:
            if not self.inl[0].m.is_var and not self.outl[1].m.is_var and not self.outl[2].m.is_var:
                return True
            elif self.inl[0].m.is_var and not self.outl[1].m.is_var and not self.outl[2].m.is_var:
                self.inl[0].m.val_SI = self.outl[1].m.val_SI + self.outl[2].m.val_SI
                self.inl[0].m.is_set = True
                self.inl[0].m.is_var = False
                return True
            elif not self.inl[0].m.is_var and self.outl[1].m.is_var and not self.outl[2].m.is_var:
                self.outl[1].m.val_SI = self.inl[0].m.val_SI - self.outl[2].m.val_SI
                self.outl[1].m.is_set = True
                self.outl[1].m.is_var = False
                return True
            elif not self.inl[0].m.is_var and not self.outl[1].m.is_var and self.outl[2].m.is_var:
                self.outl[2].m.val_SI = self.inl[0].m.val_SI - self.outl[1].m.val_SI
                self.outl[2].m.is_set = True
                self.outl[2].m.is_var = False
                return True
            return False
        else:
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
            return True

    def mass_flow_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives for mass flow equation.

        Returns
        -------
        deriv : list
            Matrix with partial derivatives for the fluid equations.
        """
        if self.inl[1].m == self.outl[0].m:
            if self.inl[0].m.is_var:
                self.network.jacobian[k, self.inl[0].m.J_col] = 1
            if self.outl[1].m.is_var:
                self.network.jacobian[k, self.outl[1].m.J_col] = -1
            if self.outl[2].m.is_var:
                self.network.jacobian[k, self.outl[2].m.J_col] = -1
        else:
            for i in self.inl:
                if i.m.is_var:
                    self.network.jacobian[k, i.m.J_col] = 1
            for o in self.outl:
                if o.m.is_var:
                    self.network.jacobian[k, o.m.J_col] = -1

    def mass_flow_tensor(self, increment_filter, k):
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
        if self.inl[1].m == self.outl[0].m:
            res = (
                (self.inl[1].h.val_SI - self.outl[0].h.val_SI)
                * self.outl[0].m.val_SI
                + (self.inl[0].h.val_SI * self.inl[0].m.val_SI -
                   self.outl[1].h.val_SI * self.outl[1].m.val_SI -
                   self.outl[2].h.val_SI * self.outl[2].m.val_SI)
            )
        else:
            res = 0
            for i in self.inl:
                res += i.m.val_SI * i.h.val_SI
            for o in self.outl:
                res -= o.m.val_SI * o.h.val_SI
        return res

    def energy_balance_variables_columns(self):
        if self.inl[1].m == self.outl[0].m:
            variables_columns1 = [data.J_col for data in
                                  [self.outl[0].m,
                                   self.inl[1].h, self.outl[0].h,
                                   self.inl[0].m, self.outl[1].m, self.outl[2].m,
                                   self.inl[0].h, self.outl[1].h, self.outl[2].h]
                                  if data.is_var]
            variables_columns1.sort()  # [self.outl[0].m, self.inl[1].h, self.outl[0].h, self.inl[0].m, self.inl[0].h, self.outl[1].h]
            return [variables_columns1]
        else:
            return super().energy_balance_variables_columns()

    def energy_balance_take_effect(self):
        pass

    def energy_balance_solve_isolated(self):
        if self.inl[1].m == self.outl[0].m:
            return False
        else:
            if sum([1 if data.is_var else 0 for conn in self.inl + self.outl for data in [conn.m, conn.h]]) > 1:
                return False
            for inconn in self.inl:
                if inconn.m.is_var:
                    inconn.m.val_SI = ((sum([outconn.m.val_SI * outconn.h.val_SI if not outconn.m.is_var and not outconn.h.is_var else 0
                                                for outconn in self.outl]) -
                                        sum([inconn.m.val_SI * inconn.h.val_SI if not inconn.m.is_var and not inconn.h.is_var else 0
                                                for inconn in self.inl])) /
                                       inconn.h.val_SI)
                    inconn.m.is_set = True
                    inconn.m.is_var = False
                    return True
                if inconn.h.is_var:
                    inconn.h.val_SI = ((sum([outconn.m.val_SI * outconn.h.val_SI if not outconn.m.is_var and not outconn.h.is_var else 0
                                                for outconn in self.outl]) -
                                        sum([inconn.m.val_SI * inconn.h.val_SI if not inconn.m.is_var and not inconn.h.is_var else 0
                                                for inconn in self.inl])) /
                                       inconn.m.val_SI)
                    inconn.h.is_set = True
                    inconn.h.is_var = False
                    return True
            for outconn in self.outl:
                if outconn.m.is_var:
                    outconn.m.val_SI = ((sum([inconn.m.val_SI * inconn.h.val_SI if not inconn.m.is_var and not inconn.h.is_var else 0
                                                 for inconn in self.inl]) -
                                         sum([outconn.m.val_SI * outconn.h.val_SI if not outconn.m.is_var and not outconn.h.is_var else 0
                                                 for outconn in self.outl])) /
                                        outconn.h.val_SI)
                    outconn.m.is_set = True
                    outconn.m.is_var = False
                    return True
                if outconn.h.is_var:
                    outconn.h.val_SI = ((sum([inconn.m.val_SI * inconn.h.val_SI if not inconn.m.is_var and not inconn.h.is_var else 0
                                                 for inconn in self.inl]) -
                                         sum([outconn.m.val_SI * outconn.h.val_SI if not outconn.m.is_var and not outconn.h.is_var else 0
                                                 for outconn in self.outl])) /
                                        outconn.m.val_SI)
                    outconn.h.is_set = True
                    outconn.h.is_var = False
                    return True
            return True

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
        # due to topology reduction this is the case quite often
        if self.inl[1].m == self.outl[0].m:
            # saturated liquid >> two phase
            if self.outl[0].m.is_var:
                self.network.jacobian[k, self.outl[0].m.J_col] = (self.inl[1].h.val_SI - self.outl[0].h.val_SI)
            if self.inl[1].h.is_var:
                self.network.jacobian[k, self.inl[1].h.J_col] = self.outl[0].m.val_SI
            if self.outl[0].h.is_var:
                self.network.jacobian[k, self.outl[0].h.J_col] = - self.outl[0].m.val_SI - self.outl[2].m.val_SI
            # liquid inlet
            if self.inl[0].m.is_var:
                self.network.jacobian[k, self.inl[0].m.J_col] = self.inl[0].h.val_SI
            if self.inl[0].h.is_var:
                self.network.jacobian[k, self.inl[0].h.J_col] = self.inl[0].m.val_SI
            # saturated vapour
            if self.outl[1].m.is_var:
                self.network.jacobian[k, self.outl[1].m.J_col] = - self.outl[1].h.val_SI
            if self.outl[1].h.is_var:
                self.network.jacobian[k, self.outl[1].h.J_col] = - self.outl[1].m.val_SI
            # saturated liquid drained
            if self.outl[2].m.is_var:
                self.network.jacobian[k, self.outl[2].m.J_col] = - self.outl[0].h.val_SI
        else:
            super().energy_balance_deriv(increment_filter, k)

    def energy_balance_tensor(self, increment_filter, k):
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
                10^6 & \text{key = 'p'}\\
                h\left(p, x=1 \right) & \text{key = 'h' at outlet 1}\\
                h\left(p, x=0 \right) & \text{key = 'h' at outlet 2}
                \end{cases}
        """
        if key == 'p':  # 10e5  !!!
            return 10e5
        elif key == 'h':
            if c.source_id == 'out1' or c.source_id == 'out3':
                return h_mix_pQ(c.p.val_SI, 0, c.fluid_data)
            else:
                return h_mix_pQ(c.p.val_SI, 1, c.fluid_data)




