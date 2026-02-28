# -*- coding: utf-8

"""Module of class ParabolicTrough.
"""

from Aurora.components.component import component_registry
from Aurora.components.fluid_components.heat_exchangers.simple import SimpleHeatExchanger
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import GroupedComponentProperties as dc_gcp
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import fluid_property_data as fpd
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.global_vars import electromagnetic_property_data as epd
from Aurora.tools.global_vars import space_time_property_data as stpd


@component_registry
class ParabolicTrough(SimpleHeatExchanger):
    r"""
    The ParabolicTrough calculates heat output from irradiance.

    **Mandatory Equations**

    - :py:meth:`aurora.components.component.Component.fluid_func`
    - :py:meth:`aurora.components.component.Component.mass_flow_func`

    **Optional Equations**

    - :py:meth:`aurora.components.component.Component.pr_func`
    - :py:meth:`aurora.components.component.Component.zeta_func`
    - :py:meth:`aurora.components.heat_exchangers.simple.SimpleHeatExchanger.energy_balance_func`
    - :py:meth:`aurora.components.heat_exchangers.simple.SimpleHeatExchanger.darcy_group_func`
    - :py:meth:`aurora.components.heat_exchangers.simple.SimpleHeatExchanger.hw_group_func`
    - :py:meth:`aurora.components.heat_exchangers.parabolic_trough.ParabolicTrough.energy_group_func`

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
        Diameter of the absorber tube, :math:`D/\text{m}`.

    L : float, dict, :code:`"var"`
        Length of the absorber tube, :math:`L/\text{m}`.

    ks : float, dict, :code:`"var"`
        Pipe's roughness, :math:`ks/\text{m}`.

    ks_HW : float, dict, :code:`"var"`
        Pipe's roughness, :math:`ks/\text{1}`.

    E : float, dict, :code:`"var"`
        Direct irradiance to tilted collector,
        :math:`E/\frac{\text{W}}{\text{m}^2}`.

    aoi : float, dict, :code:`"var"`
        Angle of incidence, :math:`aoi/^\circ`.

    doc : float, dict, :code:`"var"`
        Degree of cleanliness (1: full absorption, 0: no absorption),
        :math:`X`.

    eta_opt : float, dict, :code:`"var"`
        (constant) optical losses due to surface reflection,
        :math:`\eta_{opt}`.

    iam_1 : float, dict, :code:`"var"`
        Linear incidence angle modifier,
        :math:`iam_1/\frac{1}{^\circ}`.

    iam_2 : float, dict, :code:`"var"`
        Quadratic incidence angle modifier,
        :math:`iam_2/\left(\frac{1}{^\circ}\right)^2`.

    fA : float, dict, :code:`"var"`
        Collector aperture surface area :math:`A/\text{m}^2`.

    hf : float
        Heat flow rate, :math:`hf/\frac{\text{W}}{\text{K}}`.

    Tamb : float, dict
        Ambient temperature, provide parameter in network's temperature unit.

    energy_group : str, dict
        Parametergroup for energy balance of solarthermal collector.
    """

    @staticmethod
    def component():
        return 'parabolic trough'

    def get_parameters(self):
        data = super().get_parameters()
        for k in ["kA", "kA_char", "kA_fit", "exm"]:
            del data[k]

        data.update({
            'E': dc_cp(
                min_val=0,
                is_property=True,
                is_result=True,
                property_data=epd['Ie'],
                SI_unit=epd['Ie']['SI_unit'],
            ),
            'eta_opt': dc_cp(
                min_val=0,
                max_val=1,
                is_property=True,
                is_result=True,
                property_data=cpd['eta'],
                SI_unit=cpd['eta']['SI_unit'],
            ),
            'iam_1': dc_cp(
                is_property=True,
                is_result=True,
                property_data=cpd['eta'],
                SI_unit=cpd['eta']['SI_unit'],
            ),
            'iam_2': dc_cp(
                is_property=True,
                is_result=True,
                property_data=cpd['eta'],
                SI_unit=cpd['eta']['SI_unit'],
            ),
            'aoi': dc_cp(
                min_val=-90,
                max_val=90,
                is_property=True,
                is_result=True,
                property_data=cpd['Angle'],
                SI_unit=cpd['Angle']['SI_unit'],
            ),
            'doc': dc_cp(
                min_val=0,
                max_val=1,
                is_property=True,
                is_result=True,
                property_data=cpd['eta'],
                SI_unit=cpd['eta']['SI_unit'],
            ),
            'Q_loss': dc_cp(
                max_val=0,
                val=0,
                is_result=True,
                property_data=cpd['Q'],
                SI_unit=cpd['Q']['SI_unit'],
            ),
            'energy': dc_gcp(
                elements=[
                    'E', 'eta_opt', 'aoi', 'doc', 'hf', 'iam_1',
                    'iam_2', 'fA', 'Tamb'
                ],
                num_eq=1,
                is_set=True,
                latex=self.energy_func_doc,
                func=self.energy_func,
                deriv=self.energy_deriv,
                variables_columns=self.energy_variables_columns,
                solve_isolated=self.energy_solve_isolated,
                scale=ps['m']['scale'] * ps['h']['scale'],
            )
        })
        return data

    def energy_func(self):
        r"""
        Equation for solar collector energy balance.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                \begin{split}
                T_m = & \frac{T_{out} + T_{in}}{2}\\
                iam = & 1 - iam_1 \cdot |aoi| - iam_2 \cdot aoi^2\\
                0 = & \dot{m} \cdot \left( h_{out} - h_{in} \right)\\
                & - A \cdot \left[E \cdot \eta_{opt} \cdot doc^{1.5} \cdot
                iam \right. \\
                & \left. - hf \cdot \left(T_m - T_{amb} \right)
                \vphantom{ \eta_{opt} \cdot doc^{1.5}} \right]
                \end{split}

            Reference: :cite:`Janotte2014`.
        """
        i = self.inl[0]
        o = self.outl[0]
        T_m = (i.calc_T() + o.calc_T()) / 2
        iam = (
            1 - self.iam_1.val_SI * abs(self.aoi.val_SI)
            - self.iam_2.val_SI * self.aoi.val_SI ** 2
        )
        return (
            i.m.val_SI * (o.h.val_SI - i.h.val_SI) - self.fA.val_SI * (
                self.E.val_SI * self.eta_opt.val_SI * self.doc.val_SI ** 1.5 * iam
                - (T_m - self.Tamb.val_SI) * self.hf.val_SI
            )
        )

    def energy_func_doc(self, label):
        r"""
        Equation for solar collector energy balance.

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
            r'0 = & \dot{m}_\mathrm{in} \cdot \left( h_\mathrm{out} - '
            r'h_\mathrm{in} \right)\\' + '\n'
            r'& - A \cdot \left[E \cdot \eta_\mathrm{opt} \cdot doc^{1.5}'
            r'\cdot iam \right. \\' + '\n'
            r'&\left. -hf\cdot\left(T_\mathrm{m}-T_\mathrm{amb}\right)'
            r'\vphantom{\eta_\mathrm{opt}\cdot doc^{1.5}}\right]\\' + '\n'
            r'T_\mathrm{m}=&\frac{T_\mathrm{out}+T_\mathrm{in}}{2}\\' +
            '\n'
            r'iam = & 1 - iam_1 \cdot |aoi| - iam_2 \cdot aoi^2\\' + '\n'
            r'\end{split}'
        )
        return generate_latex_eq(self, latex, label)

    def energy_variables_columns(self):
        i = self.inl[0]
        o = self.outl[0]
        variables_columns1 = [data.J_col for data in [i.m, i.h, o.h] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def energy_solve_isolated(self):
        i = self.inl[0]
        o = self.outl[0]
        if i.m.is_var and not i.h.is_var and not o.h.is_var:
            T_m = (i.calc_T() + o.calc_T()) / 2
            iam = (
                    1 - self.iam_1.val_SI * abs(self.aoi.val_SI)
                    - self.iam_2.val_SI * self.aoi.val_SI ** 2
            )
            i.m.val_SI = self.fA.val_SI * (
                self.E.val_SI * self.eta_opt.val_SI * self.doc.val_SI ** 1.5 * iam
                - (T_m - self.Tamb.val_SI) * self.hf.val_SI
            ) / (o.h.val_SI - i.h.val_SI)
            i.m.is_set = True
            i.m.is_var = False
            return True
        return False

    def energy_deriv(self, increment_filter, k):
        r"""
        Calculate partial derivatives of energy group function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        f = self.energy_func
        i = self.inl[0]
        o = self.outl[0]
        if self.is_variable(i.m, increment_filter):
            self.network.jacobian[k, i.m.J_col] = o.h.val_SI - i.h.val_SI
        if self.is_variable(i.h, increment_filter):
            self.network.jacobian[k, i.h.J_col] = self.numeric_deriv(f, 'h', i)
        if self.is_variable(o.h, increment_filter):
            self.network.jacobian[k, o.h.J_col] = self.numeric_deriv(f, 'h', o)
        # custom variables for the energy-group
        # for variable_name in self.energy_group.elements:
        #     parameter = self.get_attr(variable_name)
        #     if parameter == self.Tamb:
        #         continue
        #     if parameter.is_var:
        #         self.network.jacobian[k, parameter.J_col] = (
        #             self.numeric_deriv(f, variable_name, None)
        #         )

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        i = self.inl[0]
        o = self.outl[0]
        self.Q.val_SI = i.m.val_SI * (o.h.val_SI - i.h.val_SI)
        self.pr.val_SI = o.p.val_SI / i.p.val_SI
        self.dp.val_SI = i.p.val_SI - o.p.val_SI
        self.zeta.val_SI = self.calc_zeta(i, o)
        if self.energy_group.is_set:
            self.Q_loss.val_SI = - self.E.val_SI * self.fA.val_SI + self.Q.val_SI
            self.Q_loss.is_result = True
        else:
            self.Q_loss.is_result = False
