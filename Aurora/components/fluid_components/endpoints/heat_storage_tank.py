# -*- coding: utf-8

"""Module for class HeatStorageTank.
"""

import numpy as np

from Aurora.components.component import Component
from Aurora.components.fluid_components.fluid_component import FluidComponent
from Aurora.components.component import component_registry
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.global_vars import fluid_property_data as fpd
from Aurora.tools.fluid_properties import h_mix_pT
from Aurora.tools.fluid_properties import p_mix_hT
from Aurora.tools.fluid_properties import dT_mix_pdh
from Aurora.tools.helpers import AURORANetworkError


@component_registry
class HeatStorageTank(FluidComponent):
    r"""
    This class used to contain the fuse salt in order to storage heat.
    The work media must be salt.
    """
    @staticmethod
    def component():
        return 'heat storage tank'

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
            # statistics
            for conn in self.inl + self.outl:
                if conn.p.is_set:
                    conn_p_set_container.append(conn)
                    p_value_set_container.append(conn.p.val)
                if conn.p.is_shared:
                    conn_p_shared_container.append(conn)
            # contain all pressure shared connection within the system
            if conn_p_shared_container:
                all_sys_conn_p_shared_list = list(set([c for c_shared in conn_p_shared_container
                                                        for c in c_shared.p.shared_connection]
                                                        + self.inl + self.outl))
            else:
                all_sys_conn_p_shared_list = self.inl + self.outl
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
                    inconn.p.is_var = False
            # pressure object posterior
            for conn in all_sys_conn_p_shared_list:
                conn.p.is_shared = True
                if conn not in conn.p.shared_connection:
                    conn.p.shared_connection.append(conn)
            #
            outconn.target.simplify_pressure_enthalpy_mass_topology(outconn)

    def get_parameters(self):
        return {
            'Q': dc_cp(
                val=np.nan,
                num_eq=1,
                func=self.Q_func,
                variables_columns=self.Q_variables_columns,
                solve_isolated=self.Q_solve_isolated,
                deriv=self.Q_deriv,
                tensor=None,
                property_data=cpd['Q'],
                SI_unit=cpd['Q']['SI_unit'],
                scale=ps['m']['scale'] * ps['h']['scale'],
                var_scale=ps['m']['scale'] * ps['h']['scale']
            ),
            'T_in': dc_cp(
                min_val=0,
                func=self.T_in_func,
                variables_columns=self.T_in_variables_columns,
                solve_isolated=self.T_in_solve_isolated,
                deriv=self.T_in_deriv,
                tensor=self.T_in_tensor,
                latex=self.T_in_func_doc,
                num_eq=1,
                property_data=fpd['T'],
                SI_unit=fpd['T']['SI_unit'],
                scale=ps['DT']['scale'],
                var_scale=ps['T']['scale']
            ),
            'T_out': dc_cp(
                min_val=0,
                func=self.T_out_func,
                variables_columns=self.T_out_variables_columns,
                solve_isolated=self.T_out_solve_isolated,
                deriv=self.T_out_deriv,
                tensor=self.T_out_tensor,
                latex=self.T_out_func_doc,
                num_eq=1,
                property_data=fpd['T'],
                SI_unit=fpd['T']['SI_unit'],
                scale=ps['DT']['scale'],
                var_scale=ps['T']['scale']
            ),
            'dm': dc_cp(
                num_eq=1,
                func=self.dm_func,
                variables_columns=self.dm_variables_columns,
                solve_isolated=self.dm_solve_isolated,
                deriv=self.dm_deriv,
                tensor=None,
                property_data=fpd['m'],
                SI_unit=fpd['m']['SI_unit'],
                scale=ps['m']['scale'],
                var_scale=ps['m']['scale']
            ),
        }

    def Q_func(self):
        i = self.inl[0]
        o = self.outl[0]
        return i.m.val_SI * i.h.val_SI - o.m.val_SI * o.h.val_SI - self.Q.val_SI

    def Q_variables_columns(self):
        i = self.inl[0]
        o = self.outl[0]
        variables_columns1 = [data.J_col for data in [i.m, o.m, i.h, o.h] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def Q_solve_isolated(self):
        i = self.inl[0]
        o = self.outl[0]
        if sum([1 if data.is_var else 0 for data in [i.m, o.m, i.h, o.h]]) > 1:
            return False
        elif i.m.is_var and not o.m.is_var and not i.h.is_var and not o.h.is_var:
            i.m.val_SI = (o.m.val_SI * o.h.val_SI + self.Q.val_SI) / i.h.val_SI
            i.m.is_set = True
            i.m.is_var = False
            self.Q.is_set = False
            return True
        elif not i.m.is_var and o.m.is_var and not i.h.is_var and not o.h.is_var:
            o.m.val_SI = (i.m.val_SI * i.h.val_SI - self.Q.val_SI) / o.h.val_SI
            o.m.is_set = True
            o.m.is_var = False
            self.Q.is_set = False
            return True
        elif not i.m.is_var and not o.m.is_var and i.h.is_var and not o.h.is_var:
            i.h.val_SI = (o.m.val_SI * o.h.val_SI + self.Q.val_SI) / i.m.val_SI
            i.h.is_set = True
            i.h.is_var = False
            self.Q.is_set = False
            return True
        elif not i.m.is_var and not o.m.is_var and not i.h.is_var and o.h.is_var:
            o.h.val_SI = (i.m.val_SI * i.h.val_SI - self.Q.val_SI) / o.m.val_SI
            o.h.is_set = True
            o.h.is_var = False
            self.Q.is_set = False
            return True
        else:
            self.Q.is_set = False
            return True

    def Q_deriv(self, increment_filter, k):
        i = self.inl[0]
        o = self.outl[0]
        if i.m.is_var:
            self.network.jacobian[k, i.m.J_col] = i.h.val_SI
        if i.h.is_var:
            self.network.jacobian[k, i.h.J_col] = i.m.val_SI
        if o.m.is_var:
            self.network.jacobian[k, o.m.J_col] = - o.h.val_SI
        if o.h.is_var:
            self.network.jacobian[k, o.h.J_col] = - o.m.val_SI

    def T_in_func(self):
        return self.inl[0].calc_T() - self.T_in.val_SI

    def T_in_variables_columns(self):
        variables_columns1 = [data.J_col
                              for data in [self.inl[0].h]
                              if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def T_in_solve_isolated(self):
        if self.inl[0].fluid.is_var:
            return False
        if self.inl[0].p.is_set and not self.inl[0].h.is_set:
            self.inl[0].h.val_SI = h_mix_pT(self.inl[0].p.val_SI,
                                             self.T_in.val_SI,
                                             self.inl[0].fluid_data,
                                             self.inl[0].mixing_rule)
            self.inl[0].h.is_set = True
            self.inl[0].h.is_var = False
            self.T_in.is_set = False
            return True
        elif not self.inl[0].p.is_set and self.inl[0].h.is_set:
            self.inl[0].p.val_SI = p_mix_hT(self.inl[0].h.val_SI,
                                             self.T_in.val_SI,
                                             self.inl[0].fluid_data,
                                             self.inl[0].mixing_rule
                                             )
            self.inl[0].p.is_set = True
            self.inl[0].p.is_var = False
            self.T_in.is_set = False
            return True
        elif self.inl[0].p.is_set and self.inl[0].h.is_set:
            self.T_in.is_set = False
            return True
        return False

    def T_in_deriv(self, increment_filter, k):
        if self.inl[0].h.is_var:
            self.network.jacobian[k, self.inl[0].h.J_col] = dT_mix_pdh(self.inl[0].p.val_SI,
                                                                        self.inl[0].h.val_SI,
                                                                        self.inl[0].fluid_data,
                                                                        self.inl[0].mixing_rule)

    def T_in_tensor(self, increment_filter, k):
        pass

    def T_in_func_doc(self, label):
        pass

    def T_out_func(self):
        return self.outl[0].calc_T() - self.T_out.val_SI

    def T_out_variables_columns(self):
        variables_columns1 = [data.J_col
                              for data in [self.outl[0].h]
                              if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def T_out_solve_isolated(self):
        if self.outl[0].fluid.is_var:
            return False
        if self.outl[0].p.is_set and not self.outl[0].h.is_set:
            self.outl[0].h.val_SI = h_mix_pT(self.outl[0].p.val_SI,
                                             self.T_out.val_SI,
                                             self.outl[0].fluid_data,
                                             self.outl[0].mixing_rule)
            self.outl[0].h.is_set = True
            self.outl[0].h.is_var = False
            self.T_out.is_set = False
            return True
        elif not self.outl[0].p.is_set and self.outl[0].h.is_set:
            self.outl[0].p.val_SI = p_mix_hT(self.outl[0].h.val_SI,
                                             self.T_out.val_SI,
                                             self.outl[0].fluid_data,
                                             self.outl[0].mixing_rule
                                             )
            self.outl[0].p.is_set = True
            self.outl[0].p.is_var = False
            self.T_out.is_set = False
            return True
        elif self.outl[0].p.is_set and self.outl[0].h.is_set:
            self.T_out.is_set = False
            return True
        return False

    def T_out_deriv(self, increment_filter, k):
        if self.outl[0].h.is_var:
            self.network.jacobian[k, self.outl[0].h.J_col] = dT_mix_pdh(self.outl[0].p.val_SI,
                                                                        self.outl[0].h.val_SI,
                                                                        self.outl[0].fluid_data,
                                                                        self.outl[0].mixing_rule)

    def T_out_tensor(self, increment_filter, k):
        pass

    def T_out_func_doc(self, label):
        pass

    def dm_func(self):
        i = self.inl[0]
        o = self.outl[0]
        return i.m.val_SI - o.m.val_SI - self.dm.val_SI

    def dm_variables_columns(self):
        i = self.inl[0]
        o = self.outl[0]
        variables_columns1 = [data.J_col for data in [i.m, o.m] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def dm_solve_isolated(self):
        i = self.inl[0]
        o = self.outl[0]
        if i.m.is_var and o.m.is_var:
            return False
        elif i.m.is_var and not o.m.is_var:
            i.m.val_SI = o.m.val_SI + self.dm.val_SI
            i.m.is_set = True
            i.m.is_var = False
            self.dm.is_set = False
            return True
        elif not i.m.is_var and o.m.is_var:
            o.m.val_SI = i.m.val_SI - self.dm.val_SI
            o.m.is_set = True
            o.m.is_var = False
            self.dm.is_set = False
            return True
        else:
            self.dm.is_set = False
            return True

    def dm_deriv(self, increment_filter, k):
        i = self.inl[0]
        o = self.outl[0]
        if i.m.is_var:
            self.network.jacobian[k, i.m.J_col] = 1
        if o.m.is_var:
            self.network.jacobian[k, o.m.J_col] = - 1

    def initialise_source(self, c, key):
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
        """
        if key == 'p':
            return 50e5
        elif key == 'h':
            T = 400 + 273.15
            return h_mix_pT(c.p.val_SI, T, c.fluid_data, c.mixing_rule)

    def initialise_target(self, c, key):
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
        """
        if key == 'p':
            return 50e5
        elif key == 'h':
            T = 400 + 273.15
            return h_mix_pT(c.p.val_SI, T, c.fluid_data, c.mixing_rule)

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        i = self.inl[0]
        o = self.outl[0]
        self.Q.val_SI = i.m.val_SI * i.h.val_SI - o.m.val_SI * o.h.val_SI
        self.T_in.val_SI = i.calc_T()
        self.T_out.val_SI = o.calc_T()
        self.dm.val_SI = i.m.val_SI - o.m.val_SI



