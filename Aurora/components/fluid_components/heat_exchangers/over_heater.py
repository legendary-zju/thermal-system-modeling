# -*- coding: utf-8

"""Module of class OverHeater.
"""

import math

import numpy as np

from Aurora.tools import logger
from Aurora.tools.characteristics import CharLine
from Aurora.components.component import component_registry
from Aurora.components.fluid_components.heat_exchangers.base import HeatExchanger
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools.fluid_properties import p_mix_hT,p_mix_hQ
from Aurora.tools.fluid_properties import d2h_mix_d2p_Q
from Aurora.tools.fluid_properties import h_mix_pQ
from Aurora.tools.fluid_properties import h_mix_pT

from Aurora.tools.fluid_properties import dT_mix_pdh

from Aurora.tools.fluid_properties import T_sat_p, p_sat_T


@component_registry
class OverHeater(HeatExchanger):
    r"""
    A OverHeater cools a fluid above saturated vapour state.

    The superheated vapour is cooled by the cold side fluid. The fluid on the hot
    side of the OverHeater must be pure.

    **Mandatory Equations**

    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.energy_balance_func`

    **Optional Equations**

    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.energy_balance_hot_func`
    - :py:meth:`AURORA.components.heat_exchangers.over_heater.OverHeater.kA_func`
    - :py:meth:`AURORA.components.heat_exchangers.over_heater.OverHeater.DTNS_func`
    - :py:meth:`AURORA.components.heat_exchangers.over_heater.OverHeater.DTU_func`
    - :py:meth:`AURORA.components.heat_exchangers.over_heater.OverHeater.DTL_func`
    - hot side :py:meth:`AURORA.components.component.Component.pr_func`
    - cold side :py:meth:`AURORA.components.component.Component.pr_func`
    - hot side :py:meth:`AURORA.components.component.Component.zeta_func`
    - cold side :py:meth:`AURORA.components.component.Component.zeta_func`

    Inlets/Outlets

    - in1, in2 (index 1: hot side (superheated vapour), index 2: cold side)
    - out1, out2 (index 1: hot side (saturated or superheated vapour), index 2: cold side)

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

    Q : float, dict
        Heat transfer, :math:`Q/\text{W}`.

    pr1 : float, dict, :code:`"var"`
        Outlet to inlet pressure ratio at hot side, :math:`pr/1`.

    pr2 : float, dict, :code:`"var"`
        Outlet to inlet pressure ratio at cold side, :math:`pr/1`.

    dp1 : float, dict, :code:`"var"`
        Inlet to outlet pressure delta at hot side, unit is the network's
        pressure unit!.

    dp2 : float, dict, :code:`"var"`
        Inlet to outlet pressure delta at cold side, unit is the network's
        pressure unit!.

    zeta1 : float, dict, :code:`"var"`
        Geometry independent friction coefficient at hot side,
        :math:`\frac{\zeta}{D^4}/\frac{1}{\text{m}^4}`.

    zeta2 : float, dict, :code:`"var"`
        Geometry independent friction coefficient at cold side,
        :math:`\frac{\zeta}{D^4}/\frac{1}{\text{m}^4}`.

    DTL : float, dict
        Lower terminal temperature difference :math:`DT_\mathrm{L}/\text{K}`.

    DTU : float, dict
        Upper terminal temperature difference :math:`DT_\mathrm{U}/\text{K}`.

    DTNS : float, dict
        Hot_inside temperature difference (referring to saturation
        temprature of superheated vapour) :math:`DT_\mathrm{NS}/\text{K}`.

    DTN_min : float, dict
        Minumum terminal temperature difference :math:`DTN_\mathrm{min}/\text{K}`.

    kA : float, dict
        Area independent heat transfer coefficient,
        :math:`kA/\frac{\text{W}}{\text{K}}`.

    kA_char : AURORA.tools.data_containers.SimpleDataContainer
        Area independent heat transfer coefficient characteristic.

    kA_char1 : AURORA.tools.characteristics.CharLine, dict
        Characteristic line for hot side heat transfer coefficient.

    kA_char2 : AURORA.tools.characteristics.CharLine, dict
        Characteristic line for cold side heat transfer coefficient.

    Note
    ----
    The condenser has an additional equation for enthalpy at hot side outlet:
    The fluid leaves the component in saturated liquid state. If subcooling
    is activated, it possible to specify the enthalpy at the outgoing
    connection manually.

    It has different calculation method for given heat transfer coefficient and
    upper terminal temperature dierence: These parameters refer to the
    **condensing** temperature, even if the fluid on the hot side enters the
    component in superheated state.

    """

    @staticmethod
    def component():
        return 'over heater'

    def get_parameters(self):
        params = super().get_parameters()
        params.update({
            'DTNS': dc_cp(
                min_val=0,
                func=self.DTNS_func,
                variables_columns=self.DTNS_variables_columns,
                solve_isolated=self.DTNS_solve_isolated,
                deriv=self.DTNS_deriv,
                repair_matrix=self.DTNS_repair_matrix,
                tensor=None,
                latex=None,
                num_eq=1,
                property_data=cpd['DT'],
                SI_unit=cpd['DT']['SI_unit'],
                scale=ps['DT']['scale'],
                var_scale=ps['DT']['scale']
            )
        })
        return params

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

            .. math::

                val = \begin{cases}
                4 \cdot 10^5 & \text{key = 'p'}\\
                h\left(p, 300 \text{K} \right) & \text{key = 'h' at inlet 1}\\
                h\left(p, 220 \text{K} \right) & \text{key = 'h' at outlet 2}
                \end{cases}
        """
        if key == 'p':
            return 50e5
        elif key == 'h':
            if c.target_id == 'in1':
                if c.p.val_SI > c.calc_p_critical():
                    c.p.val_SI = c.calc_p_critical() - 1e1
                T = T_sat_p(c.p.val_SI, c.fluid_data, c.mixing_rule) + 140
            else:
                T = 220 + 273.15
            return h_mix_pT(c.p.val_SI, T, c.fluid_data, c.mixing_rule)

    def DTNS_func(self):
        """
        Measure the degree of super-heating.
        The temperature of outlet at hot side above the temperature of saturated vapour in same pressure.

        Returns
        -------
        residual: float

        """
        o = self.outl[0]
        return o.calc_T() - T_sat_p(o.p.val_SI, o.fluid_data) - self.DTNS.val_SI

    def DTNS_variables_columns(self):
        o = self.outl[0]
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [o.h] if data.is_var]  # [i.p, i.h]
        variables_columns1.sort()
        return [variables_columns1]

    def DTNS_solve_isolated(self):
        o = self.outl[0]
        if o.p.is_var and o.h.is_var:
            return False
        elif o.p.is_var and not o.h.is_var:
            return False
        elif o.h.is_var and not o.p.is_var:
            T_i1 = T_sat_p(o.p.val_SI, o.fluid_data) + self.DTNS.val_SI
            o.h.val_SI = h_mix_pT(o.p.val_SI, T_i1, o.fluid_data, o.mixing_rule)
            o.h.is_set = True
            o.h.is_var = False
            self.DTNS.is_set = False
            return True
        else:
            self.DTNS.is_set = False
            return True

    def DTNS_deriv(self, increment_filter, k):
        o = self.outl[0]
        if o.h.is_var:
            self.network.jacobian[k, o.h.J_col] = dT_mix_pdh(o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule)

    def DTNS_repair_matrix(self, property_):
        o = self.outl[0]
        if property_ == o.h:
            h0 = h_mix_pQ(o.p.val_SI, 0, o.fluid_data)
            h1 = h_mix_pQ(o.p.val_SI, 1, o.fluid_data)
            return abs(o.calc_T() - T_sat_p(o.p.val_SI, o.fluid_data) - self.DTNS.val_SI) / max(
                o.h.val_SI - h0, h1 - o.h.val_SI)
        else:
            msg = f"variable: {property_.label} is not a valid property in DTNS_repair_matrix of {self.__class__.__name__}: {self.label}"
            raise ValueError(msg)

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        super().calc_parameters()
        self.DTNS.val_SI = self.outl[0].calc_T() - self.outl[0].calc_T_sat()

