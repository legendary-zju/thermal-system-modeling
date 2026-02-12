# -*- coding: utf-8

"""Module for class Sink.
"""

import numpy as np

from Aurora.components.component import Component
from Aurora.components.fluid_components.fluid_component import FluidComponent
from Aurora.components.component import component_registry


@component_registry
class Sink(FluidComponent):
    r"""
    A flow drains in a Sink.

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
        return 'sink'

    @staticmethod
    def inlets():
        return ['in1']

    def get_mandatory_constraints(self):
        return {}

    @staticmethod
    def get_bypass_constraints():
        return {}

    def propagate_to_target(self, branch):
        return

    def propagate_wrapper_to_target(self, branch):
        branch["components"] += [self]
        return

    def simplify_pressure_enthalpy_mass_topology(self, inconn):
        return

    def looking_forward_pressure_values(self, inconn):
        if inconn not in self.network.connections_looking_pressure_container:
            self.network.connections_looking_pressure_container.append(inconn)
        return

    def looking_for_pressure_set_boundary(self, inconn):
        if inconn not in self.network.connections_pressure_boundary_container:
            self.network.connections_pressure_boundary_container.append(inconn)
        return

    def spread_forward_pressure_values(self, inconn):
        return

    def spread_forward_pressure_initial(self, inconn):
        return

    def exergy_balance(self, T0):
        r"""Exergy balance calculation method of a sink.

        A sink does not destroy or produce exergy. The value of
        :math:`\dot{E}_\mathrm{bus}` is set to the exergy of the mass flow to
        make exergy balancing methods more simple as in general a mass flow can
        be fuel, product or loss.

        Parameters
        ----------
        T0 : float
            Ambient temperature T0 / K.

        Note
        ----
        .. math::

            \dot{E}_\mathrm{bus} = \dot{E}_\mathrm{in}^\mathrm{PH}
        """
        self.E_P = np.nan
        self.E_F = np.nan
        self.E_bus = {
            "chemical": self.inl[0].Ex_chemical,
            "physical": self.inl[0].Ex_physical,
            "massless": 0
        }
        self.E_D = np.nan
        self.epsilon = self._calc_epsilon()
