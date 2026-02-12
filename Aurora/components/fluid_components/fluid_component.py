# -*- coding: utf-8

"""Module class FluidComponent.
"""

import math

import numpy as np

from Aurora.components.component import Component
from Aurora.components.component import component_registry

from Aurora.tools import logger
from Aurora.tools.characteristics import CharLine
from Aurora.tools.characteristics import CharMap
from Aurora.tools.characteristics import load_default_char as ldc
from Aurora.tools.data_containers import ComponentCharacteristicMaps as dc_cm
from Aurora.tools.data_containers import ComponentCharacteristics as dc_cc
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import GroupedComponentCharacteristics as dc_gcc
from Aurora.tools.data_containers import GroupedComponentProperties as dc_gcp
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.data_containers import FluidProperties as dc_prop
from Aurora.tools.document_models import generate_latex_eq

from Aurora.tools.fluid_properties import v_mix_ph
from Aurora.tools.fluid_properties import dv_mix_dph
from Aurora.tools.fluid_properties import dv_mix_pdh
from Aurora.tools.fluid_properties import dv_mix_ph_dfluid
from Aurora.tools.fluid_properties import d2v_mix_d2p_h
from Aurora.tools.fluid_properties import d2v_mix_p_d2h
from Aurora.tools.fluid_properties import d2v_mix_ph_d2fluid
from Aurora.tools.fluid_properties import d2v_mix_dp_dh
from Aurora.tools.fluid_properties import d2v_mix_dp_h_dfluid
from Aurora.tools.fluid_properties import d2v_mix_p_dh_dfluid
from Aurora.tools.fluid_properties import d2v_mix_ph_dfluid1_dfluid2

from Aurora.tools.global_vars import ERR
from Aurora.tools.helpers import bus_char_derivative
from Aurora.tools.helpers import bus_char_evaluation
from Aurora.tools.helpers import newton_with_kwargs
from Aurora.tools.helpers import convert_to_SI
from Aurora.tools.helpers import convert_from_SI


@component_registry
class FluidComponent(Component):
    r"""
    Class FluidComponent is the base class of all AURORA fluid components.
    """

    @staticmethod
    def is_wrapper_branch_source():
        return False

    def start_fluid_wrapper_branch(self):
        msg = f'The fluid component {self.__class__.__name__}: {self.label} has no wrapper branch start attribute.'
        logger.error(msg)
        raise NotImplementedError(msg)

    def propagate_wrapper_to_target(self, branch):  # ???
        inconn = branch["connections"][-1]
        conn_idx = self.inl.index(inconn)
        outconn = self.outl[conn_idx]
        branch["connections"] += [outconn]
        branch["components"] += [self]
        outconn.target.propagate_wrapper_to_target(branch)

    def simplify_pressure_enthalpy_mass_topology_start(self):
        msg = f'The fluid component {self.__class__.__name__}: {self.label} has no fluid properties simplify start attribute.'
        logger.error(msg)
        raise NotImplementedError(msg)

    def simplify_pressure_enthalpy_mass_topology(self, inconn):
        if self.simplify_pressure_enthalpy_mass_topology_check():
            conn_idx = self.inl.index(inconn)  # the index of branch
            outconn = self.outl[conn_idx]  # outconn: object of outlet connection.
            outconn.target.simplify_pressure_enthalpy_mass_topology(outconn)

    def simplify_pressure_enthalpy_mass_topology_check(self):
        return True

    def get_p_ef_obj(self, conn_idx):
        p_ef_list = []
        if hasattr(self, 'pr') and hasattr(self, 'dp'):
            p_ef_list += [self.pr, self.dp]
        elif hasattr(self, f'pr{conn_idx + 1}') and hasattr(self, f'dp{conn_idx + 1}'):
            p_ef_list += [getattr(self, f'pr{conn_idx + 1}'), getattr(self, f'dp{conn_idx + 1}')]
        for p_ef in p_ef_list:
            if p_ef.is_set:
                return p_ef
        if hasattr(self, 'pr'):
            p_ef = self.pr
        elif hasattr(self, 'dp'):
            p_ef = self.dp
        elif hasattr(self, f'pr{conn_idx + 1}'):
            p_ef = getattr(self, f'pr{conn_idx + 1}')
        elif hasattr(self, f'dp{conn_idx + 1}'):
            p_ef = getattr(self, f'dp{conn_idx + 1}')
        else:
            p_ef = None
        return p_ef

    @staticmethod
    def is_spread_pressure_values_start():
        return False

    def spread_pressure_values_start(self):
        msg = f'The fluid component {self.__class__.__name__}: {self.label} has no pressure values spread start attribute.'
        logger.error(msg)
        raise NotImplementedError(msg)

    def looking_forward_pressure_values(self, inconn):
        """
        Looking forward for original pressure set point along branch, in order to spread pressure value.
        The single branch component could inherit this method directly.

        :param inconn:
        :return:
        """
        conn_idx = self.inl.index(inconn)  # the index of branch
        outconn = self.outl[conn_idx]  # outconn: object of outlet connection.
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
        else:
            return

    def looking_backward_pressure_values(self, outconn):
        """
        Looking backward for original pressure set point along branch, in order to spread pressure value.
        The single branch component could inherit this method directly.

        :param outconn:
        :return:
        """
        conn_idx = self.outl.index(outconn)
        inconn = self.inl[conn_idx]
        p_ef = self.get_p_ef_obj(conn_idx)
        if outconn not in self.network.connections_looking_pressure_container:
            self.network.connections_looking_pressure_container.append(outconn)
            # start spreading pressure value
            if inconn.p.is_set and not outconn.p.is_set and p_ef.is_set:
                self.spread_forward_pressure_values(inconn)
                inconn.spread_pressure_reference_check()
            elif not inconn.p.is_set and outconn.p.is_set and p_ef.is_set:
                self.spread_backward_pressure_values(outconn)
                outconn.spread_pressure_reference_check()
            # looking for p.set never looked
            inconn.source.looking_backward_pressure_values(inconn)
        else:
            return

    def spread_forward_pressure_values(self, inconn):
        """
        Spread forward pressure value set along branch.
        The single branch component combined pressure objective could inherit this method directly.

        :param inconn:
        :return:
        """
        conn_idx = self.inl.index(inconn)  # the index of branch
        outconn = self.outl[conn_idx]  # outconn: object of outlet connection.
        if not outconn in self.network.connections_spread_pressure_container:
            self.network.connections_spread_pressure_container.append(outconn)
            outconn.target.spread_forward_pressure_values(outconn)
            outconn.spread_pressure_reference_check()
        else:
            return

    def spread_backward_pressure_values(self, outconn):
        """
        Spread backward pressure value set along branch.
        The single branch component combined pressure objective could inherit this method directly.

        :param outconn:
        :return:
        """
        conn_idx = self.outl.index(outconn)
        inconn = self.inl[conn_idx]
        if not inconn in self.network.connections_spread_pressure_container:
            self.network.connections_spread_pressure_container.append(inconn)
            inconn.source.spread_backward_pressure_values(inconn)
            inconn.spread_pressure_reference_check()
        else:
            return

    @staticmethod
    def is_spread_pressure_initial_start():
        return False

    def spread_pressure_initial_start(self):
        msg = f'The fluid component {self.__class__.__name__}: {self.label} has no pressure initial values spread start attribute.'
        logger.error(msg)
        raise NotImplementedError(msg)

    def looking_for_pressure_set_boundary(self, inconn):
        """
        Looking for pressure set boundary.
        The single branch component could inherit this method directly.
        The single branch component combined pressure objective won't run this method.

        :param inconn:
        :return:
        """
        conn_idx = self.inl.index(inconn)  # the index of branch
        outconn = self.outl[conn_idx]  # outconn: object of outlet connection.
        if outconn.p.is_set:
            outconn.target.spread_forward_pressure_initial(outconn)
            self.spread_backward_pressure_initial(outconn)
            return
        if inconn.p.is_set:
            self.spread_forward_pressure_initial(inconn)
            inconn.source.spread_backward_pressure_initial(inconn)
            return
        if inconn not in self.network.connections_pressure_boundary_container:
            self.network.connections_pressure_boundary_container.append(inconn)
            outconn.target.looking_for_pressure_set_boundary(outconn)
        return

    def spread_forward_pressure_initial(self, inconn):
        """
        Set initial pressure value based on pressure value set without pressure ratio or pressure drop.
        The single branch component could inherit this method directly.

        :param inconn:
        :return:
        """
        conn_idx = self.inl.index(inconn)  # the index of branch
        outconn = self.outl[conn_idx]  # outconn: object of outlet connection.
        if outconn not in self.network.connections_pressure_initial_container:
            self.network.connections_pressure_initial_container.append(outconn)
            if inconn.p.val_SI and not outconn.p.val_SI:
                outconn.p.val_SI = inconn.p.val_SI * self.set_pressure_initial_factor(conn_idx)
            outconn.target.spread_forward_pressure_initial(outconn)
        else:
            return

    def spread_backward_pressure_initial(self, outconn):
        """
        Set initial pressure value based on pressure value set without pressure ratio or pressure drop.
        The single branch component could inherit this method directly.

        :param outconn:
        :return:
        """
        conn_idx = self.outl.index(outconn)
        inconn = self.inl[conn_idx]
        if inconn not in self.network.connections_pressure_initial_container:
            self.network.connections_pressure_initial_container.append(inconn)
            if outconn.p.val_SI and not inconn.p.val_SI:
                inconn.p.val_SI = outconn.p.val_SI / self.set_pressure_initial_factor(conn_idx)
            inconn.source.spread_backward_pressure_initial(inconn)
        else:
            return

    def set_pressure_initial_factor(self, branch_index=0):
        """
        Set initial pressure ratio factor for pressure initial value generation.
        Has to be rewritten in component has pressure change.

        :param branch_index:
        :return:
        """
        return 1

    @staticmethod
    def is_reaction_component():
        return False

    @staticmethod
    def is_fluid_composition_component():
        return False

    def component_type(self):
        return ['fluid']

    def interface_type(self, port_id):
        return 'fluid'

    def init_nodes_properties_units_(self):
        for row in self.nodes.index:
            for col in range(self.nodes_num):
                node = self.nodes.loc[row, col]
                for key in node.properties.keys():
                    if hasattr(node.get_attr(key), 'unit') and isinstance(node.get_attr(key), dc_prop):
                        node.get_attr(key).unit = self.inl[row].get_attr(key).unit

    def manage_fluid_equations(self):
        msg = f"The component: {self.label} of type: {self.__class__.__name__} has no fluid equations."
        logger.error(msg)

    def numeric_deriv(self, func, dx, conn=None, **kwargs):  # calculate discrete derivatives
        r"""
        Calculate partial derivative of the function func to dx.
        For details see :py:func:`Aurora.tools.helpers._numeric_deriv`
        """

        def _numeric_deriv(obj, func, dx, conn=None, **kwargs):
            r"""
            Calculate partial derivative of the function func to dx.

            Parameters
            ----------
            obj : object
                Instance, which provides the equation to calculate the derivative for.

            func : function
                Function :math:`f` to calculate the partial derivative for.

            dx : str
                Parameters of Partial derivative.

            conn : tespy.connections.connection.Connection
                Connection to calculate the numeric derivative for.

            Returns
            -------
            deriv : float/list
                Partial derivative(s) of the function :math:`f` to variable(s)
                :math:`x`.

                .. math::

                    \frac{\partial f}{\partial x} = \frac{f(x + d) + f(x - d)}{2 d}
            """
            if conn is None:
                d = obj.get_attr(dx).d  # d: the value of parameter delta in data container
                exp = 0
                obj.get_attr(dx).val_SI += d
                exp += func(**kwargs)

                obj.get_attr(dx).val_SI -= 2 * d
                exp -= func(**kwargs)
                deriv = exp / (2 * d)  # discrete derivative (slope coefficient)

                obj.get_attr(dx).val_SI += d  # restore the value of data container

            elif dx in conn.fluid.is_var:  # calculate the derivative of variable of fluid composition
                d = 1e-5
                val = conn.fluid.val[dx]  # save the original value of fluid composition temporarily
                if conn.fluid.val[dx] + d <= 1:
                    conn.fluid.val[dx] += d
                else:
                    conn.fluid.val[dx] = 1

                conn.build_fluid_data()  #
                exp = func(**kwargs)

                if conn.fluid.val[dx] - 2 * d >= 0:
                    conn.fluid.val[dx] -= 2 * d
                else:
                    conn.fluid.val[dx] = 0

                conn.build_fluid_data()
                exp -= func(**kwargs)

                conn.fluid.val[dx] = val  # restore the value of fluid composition
                conn.build_fluid_data()  # initialize fluid property again ???

                deriv = exp / (2 * d)

            elif dx in ['m', 'p', 'h']:
                if dx == 'm':
                    d = 1e-4
                else:
                    d = 1e-1
                conn.get_attr(dx).val_SI += d
                exp = func(**kwargs)

                conn.get_attr(dx).val_SI -= 2 * d
                exp -= func(**kwargs)
                deriv = exp / (2 * d)

                conn.get_attr(dx).val_SI += d

            else:
                msg = (
                    "Your variable specification for the numerical derivative "
                    "calculation seems to be wrong. It has to be a fluid name, m, "
                    "p, h or the name of a component variable."
                )
                logger.exception(msg)
                raise ValueError(msg)
            return deriv

        return _numeric_deriv(self, func, dx, conn, **kwargs)

    def numeric_tensor(self, func, dx1, dx2, conn1=None, conn2=None, **kwargs):
        if dx1 == dx2 and conn1 == conn2:
            if conn1 is None and hasattr(self, dx1):
                d = self.get_attr(dx1).d
                original_val = self.get_attr(dx1).val
                exp = 0
                exp -= 2 * func(**kwargs)
                self.get_attr(dx1).val = original_val + 2*d
                exp += func(**kwargs)
                self.get_attr(dx1).val = original_val - 2*d
                exp += func(**kwargs)
                self.get_attr(dx1).val = original_val
                tensor = exp / (4 * d**2)

            elif dx1 in conn1.fluid.is_var:  # calculate the derivative of variable of fluid composition
                d = 1e-5
                original_val = conn1.fluid.val[dx1]
                exp = 0
                exp -= 2 * func(**kwargs)

                if original_val + 2*d <= 1:
                    conn1.fluid.val[dx1] = original_val + 2*d
                else:
                    conn1.fluid.val[dx1] = 1
                conn1.build_fluid_data()
                exp += func(**kwargs)

                if original_val - 2 * d >= 0:
                    conn1.fluid.val[dx1] = original_val - 2*d
                else:
                    conn1.fluid.val[dx1] = 0
                conn1.build_fluid_data()
                exp += func(**kwargs)

                conn1.fluid.val[dx1] = original_val
                conn1.build_fluid_data()
                tensor = exp / (4 * d**2)

            elif dx1 in ['m', 'p', 'h']:
                if dx1 == 'm':
                    d = 1e-4
                else:
                    d = 1e-1
                original_val = conn1.get_attr(dx1).val_SI
                exp = 0
                exp -= 2 * func(**kwargs)
                conn1.get_attr(dx1).val_SI = original_val + 2*d
                exp += func(**kwargs)
                conn1.get_attr(dx1).val_SI = original_val - 2*d
                exp += func(**kwargs)
                conn1.get_attr(dx1).val_SI = original_val
                tensor = exp / (4 * d**2)
            else:
                msg = (
                    "Your variable specification for the numerical derivative "
                    "calculation seems to be wrong. It has to be a fluid name, m, "
                    "p, h or the name of a component variable."
                )
                logger.exception(msg)
                raise ValueError(msg)
        else:
            if dx1 in ['m', 'p', 'h']:
                if dx1 == 'm':
                    d1 = 1e-4
                else:
                    d1 = 1e-1
                original_val1 = conn1.get_attr(dx1).val_SI
                exp = 0
                if dx2 in ['m', 'p', 'h']:
                    if dx2 == 'm':
                        d2 = 1e-4
                    else:
                        d2 = 1e-1
                    original_val2 = conn2.get_attr(dx2).val_SI

                    conn1.get_attr(dx1).val_SI = original_val1 + d1
                    conn2.get_attr(dx2).val_SI = original_val2 + d2
                    exp += func(**kwargs)
                    conn1.get_attr(dx1).val_SI = original_val1 - d1
                    conn2.get_attr(dx2).val_SI = original_val2 - d2
                    exp += func(**kwargs)
                    conn1.get_attr(dx1).val_SI = original_val1 + d1
                    conn2.get_attr(dx2).val_SI = original_val2 - d2
                    exp -= func(**kwargs)
                    conn1.get_attr(dx1).val_SI = original_val1 - d1
                    conn2.get_attr(dx2).val_SI = original_val2 + d2
                    exp -= func(**kwargs)

                    conn2.get_attr(dx2).val_SI = original_val2
                    tensor = exp / (4 * d1 * d2)
                elif dx2 in conn2.fluid.is_var:
                    d2 = 1e-5
                    original_val2 = conn2.fluid.val[dx2]

                    conn1.get_attr(dx1).val_SI = original_val1 + d1
                    if original_val2 + d2 <= 1:
                        conn2.fluid.val[dx2] = original_val2 + d2
                    else:
                        conn2.fluid.val[dx2] = 1
                    conn2.build_fluid_data()
                    exp += func(**kwargs)

                    conn1.get_attr(dx1).val_SI = original_val1 - d1
                    if original_val2 - d2 >= 0:
                        conn2.fluid.val[dx2] = original_val2 - d2
                    else:
                        conn2.fluid.val[dx2] = 0
                    conn2.build_fluid_data()
                    exp += func(**kwargs)

                    conn1.get_attr(dx1).val_SI = original_val1 - d1
                    if original_val2 + d2 <= 1:
                        conn2.fluid.val[dx2] = original_val2 + d2
                    else:
                        conn2.fluid.val[dx2] = 1
                    conn2.build_fluid_data()
                    exp -= func(**kwargs)

                    conn1.get_attr(dx1).val_SI = original_val1 + d1
                    if original_val2 - d2 >= 0:
                        conn2.fluid.val[dx2] = original_val2 - d2
                    else:
                        conn2.fluid.val[dx2] = 0
                    conn2.build_fluid_data()
                    exp -= func(**kwargs)

                    conn2.fluid.val[dx2] = original_val2
                    conn2.build_fluid_data()
                    tensor = exp / (4 * d1 * d2)
                elif conn2 is None and hasattr(self, dx2):
                    d2 = self.get_attr(dx2).d
                    original_val2 = self.get_attr(dx2).val
                    conn1.get_attr(dx1).val_SI = original_val1 + d1

                    self.get_attr(dx2).val = original_val2 + d2
                    exp += func(**kwargs)
                    conn1.get_attr(dx1).val_SI = original_val1 - d1
                    self.get_attr(dx2).val = original_val2 - d2
                    exp += func(**kwargs)
                    conn1.get_attr(dx1).val_SI = original_val1 + d1
                    self.get_attr(dx2).val = original_val2 - d2
                    exp -= func(**kwargs)
                    conn1.get_attr(dx1).val_SI = original_val1 - d1
                    self.get_attr(dx2).val = original_val2 + d2
                    exp -= func(**kwargs)

                    self.get_attr(dx2).val = original_val2
                    tensor = exp / (4 * d1 * d2)
                else:
                    msg = (
                        "Your variable specification for the numerical derivative "
                        "calculation seems to be wrong. It has to be a fluid name, m, "
                        "p, h or the name of a component variable."
                    )
                    logger.exception(msg)
                    raise ValueError(msg)
                conn1.get_attr(dx1).val_SI = original_val1
            elif dx1 in conn1.fluid.is_var:
                d1 = 1e-5
                original_val1 = conn1.fluid.val[dx1]
                exp = 0
                if dx2 in ['m', 'p', 'h']:
                    if dx2 == 'm':
                        d2 = 1e-4
                    else:
                        d2 = 1e-1
                    original_val2 = conn2.get_attr(dx2).val_SI

                    if original_val1 + d1 <= 1:
                        conn1.fluid.val[dx1] = original_val1 + d1
                    else:
                        conn1.fluid.val[dx1] = 1
                    conn2.get_attr(dx2).val_SI = original_val2 + d2
                    conn1.build_fluid_data()
                    exp += func(**kwargs)

                    if original_val1 - d1 >= 0:
                        conn1.fluid.val[dx1] = original_val1 - d1
                    else:
                        conn1.fluid.val[dx1] = 0
                    conn2.get_attr(dx2).val_SI = original_val2 - d2
                    conn1.build_fluid_data()
                    exp += func(**kwargs)

                    if original_val1 + d1 <= 1:
                        conn1.fluid.val[dx1] = original_val1 + d1
                    else:
                        conn1.fluid.val[dx1] = 1
                    conn2.get_attr(dx2).val_SI = original_val2 - d2
                    conn1.build_fluid_data()
                    exp -= func(**kwargs)

                    if original_val1 - d1 >= 0:
                        conn1.fluid.val[dx1] = original_val1 - d1
                    else:
                        conn1.fluid.val[dx1] = 0
                    conn2.get_attr(dx2).val_SI = original_val2 + d2
                    conn1.build_fluid_data()
                    exp -= func(**kwargs)

                    conn2.get_attr(dx2).val_SI = original_val2
                    tensor = exp / (4 * d1 * d2)
                elif dx2 in conn2.fluid.is_var:
                    d2 = 1e-5
                    original_val2 = conn2.fluid.val[dx2]

                    if original_val1 + d1 <= 1:
                        conn1.fluid.val[dx1] = original_val1 + d1
                    else:
                        conn1.fluid.val[dx1] = 1
                    if original_val2 + d2 <= 1:
                        conn2.fluid.val[dx2] = original_val2 + d2
                    else:
                        conn2.fluid.val[dx2] = 1
                    conn1.build_fluid_data()
                    conn2.build_fluid_data()
                    exp += func(**kwargs)

                    if original_val1 - d1 >= 0:
                        conn1.fluid.val[dx1] = original_val1 - d1
                    else:
                        conn1.fluid.val[dx1] = 0
                    if original_val2 - d2 >= 0:
                        conn2.fluid.val[dx2] = original_val2 - d2
                    else:
                        conn2.fluid.val[dx2] = 0
                    conn1.build_fluid_data()
                    conn2.build_fluid_data()
                    exp += func(**kwargs)

                    if original_val1 + d1 <= 1:
                        conn1.fluid.val[dx1] = original_val1 + d1
                    else:
                        conn1.fluid.val[dx1] = 1
                    if original_val2 - d2 >= 0:
                        conn2.fluid.val[dx2] = original_val2 - d2
                    else:
                        conn2.fluid.val[dx2] = 0
                    conn1.build_fluid_data()
                    conn2.build_fluid_data()
                    exp -= func(**kwargs)

                    if original_val1 - d1 >= 0:
                        conn1.fluid.val[dx1] = original_val1 - d1
                    else:
                        conn1.fluid.val[dx1] = 0
                    if original_val2 + d2 <= 1:
                        conn2.fluid.val[dx2] = original_val2 + d2
                    else:
                        conn2.fluid.val[dx2] = 1
                    conn1.build_fluid_data()
                    conn2.build_fluid_data()
                    exp -= func(**kwargs)

                    conn2.fluid.val[dx2] = original_val2
                    conn2.build_fluid_data()
                    tensor = exp / (4 * d1 * d2)
                elif conn2 is None and hasattr(self, dx2):
                    d2 = self.get_attr(dx2).d
                    original_val2 = self.get_attr(dx2).val

                    if original_val1 + d1 <= 1:
                        conn1.fluid.val[dx1] = original_val1 + d1
                    else:
                        conn1.fluid.val[dx1] = 1
                    self.get_attr(dx2).val = original_val2 + d2
                    conn1.build_fluid_data()
                    exp += func(**kwargs)

                    if original_val1 - d1 >= 0:
                        conn1.fluid.val[dx1] = original_val1 - d1
                    else:
                        conn1.fluid.val[dx1] = 0
                    self.get_attr(dx2).val = original_val2 - d2
                    conn1.build_fluid_data()
                    exp += func(**kwargs)

                    if original_val1 + d1 <= 1:
                        conn1.fluid.val[dx1] = original_val1 + d1
                    else:
                        conn1.fluid.val[dx1] = 1
                    self.get_attr(dx2).val = original_val2 - d2
                    conn1.build_fluid_data()
                    exp -= func(**kwargs)

                    if original_val1 - d1 >= 0:
                        conn1.fluid.val[dx1] = original_val1 - d1
                    else:
                        conn1.fluid.val[dx1] = 0
                    self.get_attr(dx2).val = original_val2 + d2
                    conn1.build_fluid_data()
                    exp -= func(**kwargs)

                    self.get_attr(dx2).val = original_val2
                    tensor = exp / (4 * d1 * d2)
                else:
                    msg = (
                        "Your variable specification for the numerical derivative "
                        "calculation seems to be wrong. It has to be a fluid name, m, "
                        "p, h or the name of a component variable."
                    )
                    logger.exception(msg)
                    raise ValueError(msg)
                conn1.fluid.val[dx1] = original_val1
                conn1.build_fluid_data()
            elif conn1 is None and hasattr(self, dx1):
                d1 = self.get_attr(dx1).d
                original_val1 = self.get_attr(dx1).val
                exp = 0
                if dx2 in ['m', 'p', 'h']:
                    if dx2 == 'm':
                        d2 = 1e-4
                    else:
                        d2 = 1e-1
                    original_val2 = conn2.get_attr(dx2).val_SI

                    self.get_attr(dx1).val = original_val1 + d1
                    conn2.get_attr(dx2).val_SI = original_val2 + d2
                    exp += func(**kwargs)
                    self.get_attr(dx1).val = original_val1 - d1
                    conn2.get_attr(dx2).val_SI = original_val2 - d2
                    exp += func(**kwargs)
                    self.get_attr(dx1).val = original_val1 + d1
                    conn2.get_attr(dx2).val_SI = original_val2 - d2
                    exp -= func(**kwargs)
                    self.get_attr(dx1).val = original_val1 - d1
                    conn2.get_attr(dx2).val_SI = original_val2 + d2
                    exp -= func(**kwargs)

                    conn2.get_attr(dx2).val_SI = original_val2
                    tensor = exp / (4 * d1 * d2)
                elif dx2 in conn2.fluid.is_var:
                    d2 = 1e-5
                    original_val2 = conn2.fluid.val[dx2]

                    self.get_attr(dx1).val = original_val1 + d1
                    if original_val2 + d2 <= 1:
                        conn2.fluid.val[dx2] = original_val2 + d2
                    else:
                        conn2.fluid.val[dx2] = 1
                    conn2.build_fluid_data()
                    exp += func(**kwargs)

                    self.get_attr(dx1).val = original_val1 - d1
                    if original_val2 - d2 >= 0:
                        conn2.fluid.val[dx2] = original_val2 - d2
                    else:
                        conn2.fluid.val[dx2] = 0
                    conn2.build_fluid_data()
                    exp += func(**kwargs)

                    self.get_attr(dx1).val = original_val1 - d1
                    if original_val2 + d2 <= 1:
                        conn2.fluid.val[dx2] = original_val2 + d2
                    else:
                        conn2.fluid.val[dx2] = 1
                    conn2.build_fluid_data()
                    exp -= func(**kwargs)

                    self.get_attr(dx1).val = original_val1 + d1
                    if original_val2 - d2 >= 0:
                        conn2.fluid.val[dx2] = original_val2 - d2
                    else:
                        conn2.fluid.val[dx2] = 0
                    conn2.build_fluid_data()
                    exp -= func(**kwargs)

                    conn2.fluid.val[dx2] = original_val2
                    conn2.build_fluid_data()
                    tensor = exp / (4 * d1 * d2)
                else:
                    msg = (
                        "Your variable specification for the numerical derivative "
                        "calculation seems to be wrong. It has to be a fluid name, m, "
                        "p, h or the name of a component variable."
                    )
                    logger.exception(msg)
                    raise ValueError(msg)
                self.get_attr(dx1).val = original_val1
            else:
                msg = (
                    "Your variable specification for the numerical derivative "
                    "calculation seems to be wrong. It has to be a fluid name, m, "
                    "p, h or the name of a component variable."
                )
                logger.exception(msg)
                raise ValueError(msg)
        return tensor

    def get_char_expr(self, param, type='rel', inconn=0, outconn=0):  # need to be rewritten !!!
        r"""
        Generic method to access characteristic function parameters (ratio).

        Parameters
        ----------
        param : str
            Parameter for characteristic function evaluation.

        type : str
            Type of expression:

            - :code:`rel`: relative to design value
            - :code:`abs`: absolute value

        inconn : int
            Index of inlet connection.

        outconn : int
            Index of outlet connection.

        Returns
        -------
        expr : float
            Value of expression
        """
        if type == 'rel':
            if param == 'm':
                return self.inl[inconn].m.val_SI / self.inl[inconn].m.design
            elif param == 'm_out':
                return self.outl[outconn].m.val_SI / self.outl[outconn].m.design
            elif param == 'v':
                v = self.inl[inconn].m.val_SI * v_mix_ph(
                    self.inl[inconn].p.val_SI, self.inl[inconn].h.val_SI,
                    self.inl[inconn].fluid_data, self.inl[inconn].mixing_rule,
                    T0=self.inl[inconn].T.val_SI
                )
                return v / self.inl[inconn].v.design
            elif param == 'pr':
                return (
                    (self.outl[outconn].p.val_SI * self.inl[inconn].p.design)
                    / (self.inl[inconn].p.val_SI * self.outl[outconn].p.design)
                )
            else:
                msg = (
                    f"The parameter {param}) is not available for "
                    "characteristic function evaluation."
                )
                logger.error(msg)
                raise ValueError(msg)
        else:
            if param == 'm':
                return self.inl[inconn].m.val_SI
            elif param == 'm_out':
                return self.outl[outconn].m.val_SI
            elif param == 'v':
                return self.inl[inconn].m.val_SI * v_mix_ph(
                    self.inl[inconn].p.val_SI, self.inl[inconn].h.val_SI,
                    self.inl[inconn].fluid_data, self.inl[inconn].mixing_rule,
                    T0=self.inl[inconn].T.val_SI
                )
            elif param == 'pr':
                return self.outl[outconn].p.val_SI / self.inl[inconn].p.val_SI
            else:
                return False

    def get_char_expr_doc(self, param, type='rel', inconn=0, outconn=0):
        r"""
        Generic method to access characteristic function parameters.

        Parameters
        ----------
        param : str
            Parameter for characteristic function evaluation.

        type : str
            Type of expression:

            - :code:`rel`: relative to design value
            - :code:`abs`: absolute value

        inconn : int
            Index of inlet connection.

        outconn : int
            Index of outlet connection.

        Returns
        -------
        expr : str
            LaTeX code for documentation
        """
        if type == 'rel':
            if param == 'm':
                return (
                    r'\frac{\dot{m}_\mathrm{in,' + str(inconn + 1) + r'}}'
                    r'{\dot{m}_\mathrm{in,' + str(inconn + 1) +
                    r',design}}')
            elif param == 'm_out':
                return (
                    r'\frac{\dot{m}_\mathrm{out,' + str(outconn + 1) +
                    r'}}{\dot{m}_\mathrm{out,' + str(outconn + 1) +
                    r',design}}')
            elif param == 'v':
                return (
                    r'\frac{\dot{V}_\mathrm{in,' + str(inconn + 1) + r'}}'
                    r'{\dot{V}_\mathrm{in,' + str(inconn + 1) +
                    r',design}}')
            elif param == 'pr':
                return (
                    r'\frac{p_\mathrm{out,' + str(outconn + 1) +
                    r'}\cdot p_\mathrm{in,' + str(inconn + 1) +
                    r',design}}{p_\mathrm{out,' + str(outconn + 1) +
                    r',design}\cdot p_\mathrm{in,' + str(inconn + 1) +
                    r'}}')
        else:
            if param == 'm':
                return r'\dot{m}_\mathrm{in,' + str(inconn + 1) + r'}'
            elif param == 'm_out':
                return r'\dot{m}_\mathrm{out,' + str(outconn + 1) + r'}'
            elif param == 'v':
                return r'\dot{V}_\mathrm{in,' + str(inconn + 1) + r'}'
            elif param == 'pr':
                return (
                    r'\frac{p_\mathrm{out,' + str(outconn + 1) +
                    r'}}{p_\mathrm{in,' + str(inconn + 1) + r'}}')

    def bounds_p_generate(self):
        return

    def bounds_h_generate(self):
        return

    def correct_massflow_enthalpy(self):
        return

    def constraints_h_generate(self):
        return

    def pressure_equality_func(self):
        r"""
        Residual Equation for pressure equality.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = p_{in,i} - p_{out,i} \;\forall i\in\text{inlets}
        """
        residual = []
        for i in range(self.num_i):
            residual += [self.inl[i].p.val_SI - self.outl[i].p.val_SI]
        return residual

    def pressure_equality_variables_columns(self):
        variables_columns = []
        for i in range(self.num_i):
            variables_columnsi = []
            variables_columnsi += [data.J_col for data in [self.inl[i].p, self.outl[i].p] if data.is_var]
            variables_columnsi.sort()
            variables_columns.append(variables_columnsi)
        return variables_columns

    def pressure_equality_take_effect(self):
        pass

    def pressure_equality_solve_isolated(self):

        return False

    def pressure_equality_func_doc(self, label):
        r"""
        Equation for pressure equality.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        indices = list(range(1, self.num_i + 1))
        if len(indices) > 1:
            indices = ', '.join(str(idx) for idx in indices)
        else:
            indices = str(indices[0])
        latex = (
            r'0=p_{\mathrm{in,}i}-p_{\mathrm{out,}i}'
            r'\; \forall i \in [' + indices + r']')
        return generate_latex_eq(self, latex, label)

    def pressure_equality_deriv(self, increment_filter, k):  #
        r"""
        Calculate pressure partial derivatives for all mass flow balance equations.

        Parameters
        ----------
        k : int
            Position of equation in Jacobian matrix.

        Returns
        -------
        deriv : ndarray
            Matrix with partial derivatives for the mass flow balance
            equations.
        """
        for i in range(self.num_i):
            if self.inl[i].p.is_var:
                self.network.jacobian[k + i, self.inl[i].p.J_col] = 1  # due to the coefficient in pressure_func
            if self.outl[i].p.is_var:
                self.network.jacobian[k + i, self.outl[i].p.J_col] = -1  # due to the coefficient in pressure_func

    def pressure_equality_tensor(self, increment_filter, k):
        pass

    def enthalpy_equality_func(self):
        r"""
        Residual Equation for enthalpy equality.

        Returns
        -------
        residual : list
            Residual values of equations.

            .. math::

                0 = h_{in,i} - h_{out,i} \;\forall i\in\text{inlets}
        """
        residual = []
        for i in range(self.num_i):
            residual += [self.inl[i].h.val_SI - self.outl[i].h.val_SI]
        return residual

    def enthalpy_equality_variables_columns(self):
        variables_columns = []
        for i in range(self.num_i):
            variables_columnsi = []
            variables_columnsi += [data.J_col for data in [self.inl[i].h, self.outl[i].h] if data.is_var]
            variables_columnsi.sort()
            variables_columns.append(variables_columnsi)
        return variables_columns

    def enthalpy_equality_take_effect(self):
        pass

    def enthalpy_equality_solve_isolated(self):
        return False

    def enthalpy_equality_func_doc(self, label):
        r"""
        Equation for enthalpy equality.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        indices = list(range(1, self.num_i + 1))
        if len(indices) > 1:
            indices = ', '.join(str(idx) for idx in indices)
        else:
            indices = str(indices[0])
        latex = (
            r'0=h_{\mathrm{in,}i}-h_{\mathrm{out,}i}'
            r'\; \forall i \in [' + indices + r']'
        )
        return generate_latex_eq(self, latex, label)

    def enthalpy_equality_deriv(self, increment_filter, k):  #
        r"""
        Calculate enthalpy partial derivatives for all mass flow balance equations.

        Parameters
        ----------
        k : int
            Position of equation in Jacobian matrix.

        Returns
        -------
        deriv : ndarray
            Matrix with partial derivatives for the mass flow balance
            equations.
        """
        for i in range(self.num_i):
            if self.inl[i].h.is_var:
                self.network.jacobian[k + i, self.inl[i].h.J_col] = 1  # due to the coefficient in enthalpy_func
            if self.outl[i].h.is_var:
                self.network.jacobian[k + i, self.outl[i].h.J_col] = -1  # due to the coefficient in enthalpy_func

    def enthalpy_equality_tensor(self, increment_filter, k):
        pass

    def pr_func(self, pr='', inconn=0, outconn=0):
        r"""
        Calculate residual value of pressure ratio function.

        Parameters
        ----------
        pr : str
            Component parameter to evaluate the pr_func on, e.g.
            :code:`pr1`.

        inconn : int
            Connection index of inlet.

        outconn : int
            Connection index of outlet.

        Returns
        -------
        residual : float
            Residual value of function.

            .. math::

                0 = p_{in} \cdot pr - p_{out}
        """
        kwargs = {'pr': pr, 'inconn': inconn, 'outconn': outconn}
        return self.inl[inconn].p.val_SI * self.get_attr(pr + '_fit')(**kwargs) - self.outl[outconn].p.val_SI

    def pr_variables_columns(self, pr='', inconn=0, outconn=0):
        pr_obj = self.get_attr(pr)
        i = self.inl[inconn]
        o = self.outl[outconn]  #
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [i.p, o.p, pr_obj] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def pr_take_effect(self, pr='', inconn=0, outconn=0):
        pr = self.get_attr(pr)
        i = self.inl[inconn]
        o = self.outl[outconn]
        pass

    def pr_solve_isolated(self, pr='', inconn=0, outconn=0):
        kwargs = {'pr': pr, 'inconn': inconn, 'outconn': outconn}
        i = self.inl[inconn]
        o = self.outl[outconn]
        pr_obj = self.get_attr(pr)
        if not i.p.is_var and not o.p.is_var:
            pr_obj.is_set = False
            return True
        elif i.p.is_var and not o.p.is_var and self.get_attr(pr + '_fit').rule in ['constant', 'static']:
            i.p.val_SI = o.p.val_SI / self.get_attr(pr + '_fit')(**kwargs)
            i.p.is_set = True
            i.p.is_var = False
            pr_obj.is_set = False
            return True
        elif not i.p.is_var and o.p.is_var and self.get_attr(pr + '_fit').rule in ['constant', 'static']:
            o.p.val_SI = i.p.val_SI * self.get_attr(pr + '_fit')(**kwargs)
            o.p.is_set = True
            o.p.is_var = False
            pr_obj.is_set = False
            return True
        return False

    def pr_func_doc(self, label, pr='', inconn=0, outconn=0):
        r"""
        Calculate residual value of pressure ratio function.

        Parameters
        ----------
        pr : str
            Component parameter to evaluate the pr_func on, e.g.
            :code:`pr1`.

        inconn : int
            Connection index of inlet.

        outconn : int
            Connection index of outlet.

        Returns
        -------
        residual : float
            Residual value of function.
        """
        latex = (
            r'0=p_\mathrm{in,' + str(inconn + 1) + r'}\cdot ' + pr +
            r' - p_\mathrm{out,' + str(outconn + 1) + r'}'
        )
        return generate_latex_eq(self, latex, label)

    def pr_deriv(self, increment_filter, k, pr='', inconn=0, outconn=0):
        r"""
        Calculate residual value of pressure ratio function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of equation in Jacobian matrix.

        pr : str
            Component parameter to evaluate the pr_func on, e.g.
            :code:`pr1`.

        inconn : int
            Connection index of inlet.

        outconn : int
            Connection index of outlet.
        """
        kwargs = {'pr': pr, 'inconn': inconn, 'outconn': outconn}
        pr_obj = self.get_attr(pr)
        i = self.inl[inconn]
        o = self.outl[outconn]  #
        if i.p.is_var:
            self.network.jacobian[k, i.p.J_col] = self.get_attr(pr + '_fit')(**kwargs)  # due to the coefficient in pr_func
        if o.p.is_var:
            self.network.jacobian[k, o.p.J_col] = -1  # due to the coefficient in pr_func
        if pr_obj.is_var:
            self.network.jacobian[k, self.pr.J_col] = i.p.val_SI  # due to the coefficient in pr_func

    def pr_tensor(self, increment_filter, k, pr='', inconn=0, outconn=0):
        pr_obj = self.get_attr(pr)
        i = self.inl[inconn]
        o = self.outl[outconn]  #
        if i.p.is_var and pr_obj.is_var:
            self.network.tensor[i.p.J_col, self.pr.J_col, k] = 1
            self.network.tensor[self.pr.J_col, i.p.J_col, k] = 1

    def pr_constant_func_(self, pr='', inconn=0, outconn=0):
        return self.get_attr(pr).val_SI

    def pr_default_func_(self, pr='', inconn=0, outconn=0):
        i = self.inl[inconn]
        o = self.outl[outconn]
        zeta = self.get_attr(pr.replace('pr', 'zeta'))
        v_i = v_mix_ph(i.p.val_SI, i.h.val_SI, i.fluid_data, i.mixing_rule, T0=i.T.val_SI)
        v_o = v_mix_ph(o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule, T0=o.T.val_SI)
        dp = zeta.design * 4 * i.m.val_SI ** 2 * (v_i + v_o) / (math.pi ** 2)
        return 1 - dp / i.p.val_SI

    def pr_charline_func_(self, pr='', inconn=0, outconn=0):
        if not self.get_attr(pr + '_char').is_set:
            self.get_attr(pr + '_char').char_func = CharLine(x=[0, 1], y=[1, 1])
        p1 = self.get_attr(pr + '_char').param
        f1 = self.get_char_expr(p1, **self.get_attr(pr + '_char').char_params)
        alfa1 = self.get_attr(pr + '_char').char_func.evaluate(f1)
        return alfa1 * self.get_attr(pr).design

    def calc_zeta(self, i, o):
        if abs(i.m.val_SI) <= 1e-4:
            return 0
        else:
            return (
                (i.p.val_SI - o.p.val_SI) * math.pi ** 2
                / (4 * i.m.val_SI ** 2 * (i.vol.val_SI + o.vol.val_SI))
            )

    def dp_func(self, dp=None, inconn=None, outconn=None):
        """Calculate residual value of pressure difference function.

        Parameters
        ----------
        dp : str
            Component parameter to evaluate the dp_func on, e.g.
            :code:`dp1`.

        inconn : int
            Connection index of inlet.

        outconn : int
            Connection index of outlet.

        Returns
        -------
        residual : float
            Residual value of function.

            .. math::

                0 = p_{in} - p_{out} - dp
        """
        kwargs = dict(dp=dp, inconn=inconn, outconn=outconn)
        inlet_conn = self.inl[inconn]
        outlet_conn = self.outl[outconn]
        return inlet_conn.p.val_SI - outlet_conn.p.val_SI - self.get_attr(dp + '_fit')(**kwargs)

    def dp_variables_columns(self, dp=None, inconn=None, outconn=None):
        i = self.inl[inconn]
        o = self.outl[outconn]
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [i.p, o.p] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def dp_take_effect(self, dp=None, inconn=None, outconn=None):
        pass

    def dp_solve_isolated(self, dp=None, inconn=None, outconn=None):
        kwargs = dict(dp=dp, inconn=inconn, outconn=outconn)
        inlet_conn = self.inl[inconn]
        outlet_conn = self.outl[outconn]
        dp_obj = self.get_attr(dp)
        if not inlet_conn.p.is_var and not outlet_conn.p.is_var:
            dp_obj.is_set = False
            return True
        elif inlet_conn.p.is_var and not outlet_conn.p.is_var and self.get_attr(dp + '_fit').rule in ['constant', 'static']:
            inlet_conn.p.val_SI = outlet_conn.p.val_SI + self.get_attr(dp + '_fit')(**kwargs)
            inlet_conn.p.is_set = True
            inlet_conn.p.is_var = False
            dp_obj.is_set = False
            return True
        elif not inlet_conn.p.is_var and outlet_conn.p.is_var and self.get_attr(dp + '_fit').rule in ['constant', 'static']:
            outlet_conn.p.val_SI = inlet_conn.p.val_SI - self.get_attr(dp + '_fit')(**kwargs)
            outlet_conn.p.is_set = True
            outlet_conn.p.is_var = False
            dp_obj.is_set = False
            return True
        return False

    def dp_deriv(self, increment_filter, k, dp=None, inconn=None, outconn=None):
        r"""
        Calculate residual value of pressure difference function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of equation in Jacobian matrix.

        dp : str
            Component parameter to evaluate the dp_func on, e.g.
            :code:`dp1`.

        inconn : int
            Connection index of inlet.

        outconn : int
            Connection index of outlet.
        """
        inlet_conn = self.inl[inconn]
        outlet_conn = self.outl[outconn]
        if inlet_conn.p.is_var:
            self.network.jacobian[k, inlet_conn.p.J_col] = 1
        if outlet_conn.p.is_var:
            self.network.jacobian[k, outlet_conn.p.J_col] = -1

    def dp_tensor(self, increment_filter, k, dp='', inconn=None, outconn=None):
        pass

    def dp_constant_func_(self, dp=None, inconn=None, outconn=None):
        return self.get_attr(dp).val_SI

    def dp_default_func_(self, dp=None, inconn=None, outconn=None):
        i = self.inl[inconn]
        o = self.outl[outconn]
        if abs(i.m.val_SI) < 1e-4:
            return 0
        zeta = self.get_attr(dp.replace('dp', 'zeta'))
        v_i = v_mix_ph(i.p.val_SI, i.h.val_SI, i.fluid_data, i.mixing_rule, T0=i.T.val_SI)
        v_o = v_mix_ph(o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule, T0=o.T.val_SI)
        return zeta.design * 4 * i.m.val_SI ** 2 * (v_i + v_o) / (math.pi ** 2)

    def dp_charline_func_(self, dp=None, inconn=None, outconn=None):
        if not self.get_attr(dp + '_char').is_set:
            self.get_attr(dp + '_char').char_func = CharLine(x=[0, 1], y=[1, 1])
        p1 = self.get_attr(dp + '_char').param
        f1 = self.get_char_expr(p1, **self.get_attr(dp + '_char').char_params)
        alfa1 = self.get_attr(dp + '_char').char_func.evaluate(f1)
        return alfa1 * self.get_attr(dp).design

