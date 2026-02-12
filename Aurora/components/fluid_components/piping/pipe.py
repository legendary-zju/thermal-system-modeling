# -*- coding: utf-8

"""Module of class Pipe.
"""

from Aurora.components.component import component_registry
from Aurora.components.fluid_components.heat_exchangers.simple import SimpleHeatExchanger


@component_registry
class Pipe(SimpleHeatExchanger):
    r"""
    The Pipe is a subclass of a SimpleHeatExchanger.

    **Mandatory Equations**

    - :py:meth:`AURORA.components.component.Component.fluid_func`
    - :py:meth:`AURORA.components.component.Component.mass_flow_func`

    **Optional Equations**

    - :py:meth:`AURORA.components.component.Component.pr_func`
    - :py:meth:`AURORA.components.component.Component.zeta_func`
    - :py:meth:`AURORA.components.heat_exchangers.simple.SimpleHeatExchanger.energy_balance_func`
    - :py:meth:`AURORA.components.heat_exchangers.simple.SimpleHeatExchanger.darcy_group_func`
    - :py:meth:`AURORA.components.heat_exchangers.simple.SimpleHeatExchanger.hw_group_func`
    - :py:meth:`AURORA.components.heat_exchangers.simple.SimpleHeatExchanger.kA_group_func`
    - :py:meth:`AURORA.components.heat_exchangers.simple.SimpleHeatExchanger.kA_char_group_func`

    Inlets/Outlets

    - in1
    - out1

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

    Q : float, dict, :code:`"var"`
        Heat transfer, :math:`Q/\text{W}`.

    pr : float, dict, :code:`"var"`
        Outlet to inlet pressure ratio, :math:`pr/1`.

    zeta : float, dict, :code:`"var"`
        Geometry independent friction coefficient,
        :math:`\frac{\zeta}{D^4}/\frac{1}{\text{m}^4}`.

    D : float, dict, :code:`"var"`
        Diameter of the pipes, :math:`D/\text{m}`.

    L : float, dict, :code:`"var"`
        Length of the pipes, :math:`L/\text{m}`.

    ks : float, dict, :code:`"var"`
        Pipe's roughness, :math:`ks/\text{m}`.

    darcy_group : str, dict
        Parametergroup for pressure drop calculation based on pipes dimensions
        using darcy weissbach equation.

    ks_HW : float, dict, :code:`"var"`
        Pipe's roughness, :math:`ks/\text{1}`.

    hw_group : str, dict
        Parametergroup for pressure drop calculation based on pipes dimensions
        using hazen williams equation.

    kA : float, dict, :code:`"var"`
        Area independent heat transfer coefficient,
        :math:`kA/\frac{\text{W}}{\text{K}}`.

    kA_char : tespy.tools.characteristics.CharLine, dict
        Characteristic line for heat transfer coefficient.

    Tamb : float, dict
        Ambient temperature, provide parameter in network's temperature
        unit.

    kA_group : str, dict
        Parametergroup for heat transfer calculation from ambient temperature
        and area independent heat transfer coefficient kA.

    """

    @staticmethod
    def component():
        return 'pipe'
