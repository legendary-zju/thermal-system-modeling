# -*- coding: utf-8

"""Module for class FlowAmplifier.
"""

import numpy as np

from Aurora.components.component import Component
from Aurora.components.fluid_components.fluid_component import FluidComponent
from Aurora.components.component import component_registry
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.helpers import AURORANetworkError


@component_registry
class FlowAmplifier(FluidComponent):
    r"""
    This class used to amplify the mass flow of branch in order to simplify topological constructure
    due to duplicate components and connections.
    """
    @staticmethod
    def component():
        return 'flow amplifier'

    @staticmethod
    def inlets():
        return ['in1']

    @staticmethod
    def outlets():
        return ['out1']

    @staticmethod
    def is_branch_source():
        return True

    def start_branch(self):
        outconn = self.outl[0]  # out connection object
        branch = {
            "connections": [outconn],
            "components": [self, outconn.target],
            "subbranches": {}
        }
        outconn.target.propagate_to_target(branch)

        return {outconn.label: branch}

    def propagate_to_target(self, branch):
        return

    def simplify_pressure_enthalpy_mass_topology_check(self):
        if self in self.network.branches_components:
            return False
        else:
            return True

    def simplify_pressure_enthalpy_mass_topology(self, inconn):
        if self.simplify_pressure_enthalpy_mass_topology_check():
            self.network.branches_components.append(self)
            outconn = self.outl[0]
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
                outconn._p_tmp = outconn.p
                outconn.p = inconn.p
            # set pressure value
            if conn_p_set_container:
                if len(set(p_value_set_container)) > 1:
                    msg = f"Has not set sole pressure value of branches of flow amplifier component: {self.label}"
                    raise AURORANetworkError(msg)
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
                outconn._h_tmp = outconn.h
                outconn.h = inconn.h
            # set enthalpy value
            if conn_h_set_container:
                if len(set(h_value_set_container)) > 1:
                    msg = f"Has not set sole enthalpy value of branches of flow amplifier component: {self.label}"
                    raise AURORANetworkError(msg)
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
            outconn.target.simplify_pressure_enthalpy_mass_topology(outconn)

    def get_parameters(self):
        return {
            'Ki': dc_cp(
                min_val=0,
                val=np.nan,
                num_eq=1,
                func=self.Ki_func,
                variables_columns=self.Ki_variables_columns,
                solve_isolated=self.Ki_solve_isolated,
                deriv=self.Ki_deriv,
                tensor=None,
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
                scale=ps['eff']['scale'],
                var_scale=ps['eff']['scale']
            ),
        }

    def Ki_func(self):
        i = self.inl[0]
        o = self.outl[0]
        return i.m.val_SI - o.m.val_SI * self.Ki.val_SI

    def Ki_variables_columns(self):
        i = self.inl[0]
        o = self.outl[0]
        variables_columns1 = [data.J_col for data in [i.m, o.m] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def Ki_solve_isolated(self):
        i = self.inl[0]
        o = self.outl[0]
        if i.m.is_var and o.m.is_var:
            return False
        elif i.m.is_var and not o.m.is_var:
            i.m.val_SI = o.m.val_SI * self.Ki.val_SI
            i.m.is_set = True
            i.m.is_var = False
            self.Ki.is_set = False
            return True
        elif not i.m.is_var and o.m.is_var:
            o.m.val_SI = i.m.val_SI / self.Ki.val_SI
            o.m.is_set = True
            o.m.is_var = False
            self.Ki.is_set = False
            return True
        else:
            self.Ki.is_set = False
            return True

    def Ki_deriv(self, increment_filter, k):
        i = self.inl[0]
        o = self.outl[0]
        if i.m.is_var:
            self.network.jacobian[k, i.m.J_col] = 1
        if o.m.is_var:
            self.network.jacobian[k, o.m.J_col] = - self.Ki.val_SI

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        i = self.inl[0]
        o = self.outl[0]
        self.Ki.val_SI = i.m.val_SI / o.m.val_SI



