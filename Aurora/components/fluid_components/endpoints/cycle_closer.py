# -*- coding: utf-8

"""Module for class CycleCloser
"""

import numpy as np

from Aurora.components.component import Component
from Aurora.components.fluid_components.fluid_component import FluidComponent
from Aurora.components.component import component_registry
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import fluid_property_data as fpd
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.helpers import AURORANetworkError


@component_registry
class CycleCloser(FluidComponent):
    r"""
    Component for closing cycles.

    Simplify the topological constructure through sharing objective of pressure、enthalpy property.

    Image not available

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

    Note
    ----
    This component can be used to close a cycle process. The system of
    equations describing your plant will overdetermined, if you close a cycle
    without this component or a cut the cycle with a sink and a source at
    some point of the cycle. This component can be used instead of cutting
    the cycle.
    """

    @staticmethod
    def component():
        return 'cycle closer'

    @staticmethod
    def get_parameters():
        return {
            'mass_deviation': dc_cp(
                val=0,
                max_val=1e-3,
                is_result=True,
                property_data=fpd['m'],
                SI_unit=fpd['m']['SI_unit'],
                scale=ps['m']['scale']),
            'fluid_deviation': dc_cp(
                val=0,
                max_val=1e-5,
                is_result=True,
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
                scale=ps['fluid']['scale']),
        }

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

    @staticmethod
    def is_wrapper_branch_source():
        return True

    def start_fluid_wrapper_branch(self):
        outconn = self.outl[0]
        branch = {
            "connections": [outconn],
            "components": [self],
            "massflow": []
        }
        outconn.target.propagate_wrapper_to_target(branch)

        return {outconn.label: branch}

    def propagate_to_target(self, branch):
        return

    def propagate_wrapper_to_target(self, branch):
        branch["components"] += [self]
        return

    @staticmethod
    def is_simplify_topology_start():
        return True

    def simplify_pressure_enthalpy_mass_topology_start(self):
        outconn = self.outl[0]
        outconn.target.simplify_pressure_enthalpy_mass_topology(outconn)

    @staticmethod
    def is_spread_pressure_values_start():
        return True

    def spread_pressure_values_start(self):
        outconn = self.outl[0]
        inconn = self.inl[0]
        outconn.target.looking_forward_pressure_values(outconn)
        inconn.source.looking_backward_pressure_values(inconn)

    @staticmethod
    def is_spread_pressure_initial_start():
        return True

    def spread_pressure_initial_start(self):
        outconn = self.outl[0]
        inconn = self.inl[0]
        outconn.target.looking_for_pressure_set_boundary(outconn)

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
                    msg = f"Has not set sole pressure value of branches of cycle closer component: {self.label}"
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
                    msg = f"Has not set sole enthalpy value of branches of cycle closer component: {self.label}"
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

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        # calculate deviation in mass flow
        self.mass_deviation.val_SI = abs(
            self.inl[0].m.val_SI - self.outl[0].m.val_SI
        )

        # calculate deviation in fluid composition
        d1 = self.inl[0].fluid.val
        d2 = self.outl[0].fluid.val
        diff = [d1[key] - d2[key] for key in d1.keys()]
        self.fluid_deviation.val_SI = np.linalg.norm(diff)
