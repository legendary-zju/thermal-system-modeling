# -*- coding: utf-8

"""Module for class Source.
"""

import numpy as np

from Aurora.components.component import Component
from Aurora.components.fluid_components.fluid_component import FluidComponent
from Aurora.components.component import component_registry


@component_registry
class Source(FluidComponent):
    r"""
    A flow originates from a Source.

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
        return 'source'

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

    @staticmethod
    def is_simplify_topology_start():
        return True

    def simplify_pressure_enthalpy_mass_topology_start(self):
        """
        Start combining pressure/enthalpy/mass_flow objective of connections along branches.

        Returns
        -------

        """
        outconn = self.outl[0]
        outconn.target.simplify_pressure_enthalpy_mass_topology(outconn)

    @staticmethod
    def is_spread_pressure_values_start():
        return True

    def spread_pressure_values_start(self):
        """
        Start spreading pressure values from the source component along branches.

        Returns
        -------

        """
        outconn = self.outl[0]
        outconn.target.looking_forward_pressure_values(outconn)

    @staticmethod
    def is_spread_pressure_initial_start():
        return True

    def spread_pressure_initial_start(self):
        """
        Start setting initial pressure values along branches.

        Returns
        -------

        """
        outconn = self.outl[0]
        outconn.target.looking_for_pressure_set_boundary(outconn)

    def looking_backward_pressure_values(self, outconn):
        if outconn not in self.network.connections_looking_pressure_container:
            self.network.connections_looking_pressure_container.append(outconn)
        return

    def spread_backward_pressure_values(self, outconn):
        return

    def spread_backward_pressure_initial(self, outconn):
        return

    def get_mandatory_constraints(self):
        return {}

    @staticmethod
    def get_bypass_constraints():
        return {}

    def exergy_balance(self, T0):
        r"""Exergy balance calculation method of a source.

        A source does not destroy or produce exergy. The value of
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

            \dot{E}_\mathrm{bus} = \dot{E}_\mathrm{out}^\mathrm{PH}
        """
        self.E_P = np.nan
        self.E_F = np.nan
        self.E_bus = {
            "chemical": self.outl[0].Ex_chemical,
            "physical": self.outl[0].Ex_physical,
            "massless": 0
        }
        self.E_D = np.nan
        self.epsilon = self._calc_epsilon()
