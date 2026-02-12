# -*- coding: utf-8

"""Module of class Turbine.
"""

import numpy as np

from Aurora.components.component import component_registry
from Aurora.components.fluid_components.turbomachinery.base import Turbomachine
from Aurora.tools import logger, CharLine, CharMap
from Aurora.tools.data_containers import ComponentCharacteristics as dc_cc
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
from Aurora.tools.data_containers import FitCoefficient as dc_fit
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools.fluid_properties import isentropic


@component_registry
class Turbine(Turbomachine):
    r"""
    Class for gas or steam turbines.

    **Mandatory Equations**

    - :py:meth:`Aurora.components.component.Component.fluid_func`
    - :py:meth:`Aurora.components.component.Component.mass_flow_func`

    **Optional Equations**

    - :py:meth:`Aurora.components.component.Component.pr_func`
    - :py:meth:`Aurora.components.turbomachinery.base.Turbomachine.energy_balance_func`
    - :py:meth:`Aurora.components.turbomachinery.turbine.Turbine.eta_s_func`
    - :py:meth:`Aurora.components.turbomachinery.turbine.Turbine.eta_s_char_func`
    - :py:meth:`Aurora.components.turbomachinery.turbine.Turbine.cone_func`

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

    P : float, dict
        Power, :math:`P/\text{W}`

    eta_s : float, dict
        Isentropic efficiency, :math:`\eta_s/1`

    pr : float, dict, :code:`"var"`
        Outlet to inlet pressure ratio, :math:`pr/1`

    eta_s_char : Aurora.tools.characteristics.CharLine, dict
        Characteristic curve for isentropic efficiency, provide CharLine as
        function :code:`func`.

    cone : dict
        Apply Stodola's cone law (works in offdesign only).

    """

    @staticmethod
    def component():
        return 'turbine'

    def get_parameters(self):
        parameters = super().get_parameters()
        parameters.update({
            'eta_s': dc_cp(
                min_val=0,
                max_val=1,
                func=self.eta_s_func,
                variables_columns=self.eta_s_variables_columns,
                solve_isolated=self.eta_s_solve_isolated,
                deriv=self.eta_s_deriv,
                tensor=self.eta_s_tensor,
                latex=self.eta_s_func_doc,
                num_eq=1,
                property_data=cpd['eta'],
                SI_unit=cpd['eta']['SI_unit'],
                scale=ps['h']['scale'],
                var_scale=ps['eff']['scale']
            ),
            'eta_s_fit': dc_fit(
                rule='constant',
                constant=self.eta_s_constant_func_,
                default=self.eta_s_default_func_,
                charline=self.eta_s_charline_func_,
                charmap=self.eta_s_charmap_func_,
                self_defined=self.eta_s_self_defined_func_,
            ),
            'eta_s_char': dc_cc(
                param='m',
                char_params={'type': 'rel', 'inconn': 0, 'outconn': 0}
            ),
            'pr_fit': dc_fit(
                rule='constant',
                constant=self.pr_constant_func_,
                default=self.pr_default_func_,
            ),
            'dp_fit': dc_fit(
                rule='constant',
                constant=self.dp_constant_func_,
                default=self.dp_default_func_,
            ),
        })
        return parameters

    def calc_eta_s(self):
        inl = self.inl[0]
        outl = self.outl[0]
        return (
            (outl.h.val_SI - inl.h.val_SI)
            / (isentropic(
                    inl.p.val_SI,
                    inl.h.val_SI,
                    outl.p.val_SI,
                    inl.fluid_data,
                    inl.mixing_rule,
                    T0=inl.T.val_SI
                ) - inl.h.val_SI
            )
        )

    def eta_s_func(self):
        r"""
        Equation for given isentropic efficiency of a turbine.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = -\left( h_{out} - h_{in} \right) +
                \left( h_{out,s} - h_{in} \right) \cdot \eta_{s,e}
        """
        inl = self.inl[0]
        outl = self.outl[0]
        return (
            -(outl.h.val_SI - inl.h.val_SI)
            + (
                isentropic(
                    inl.p.val_SI,
                    inl.h.val_SI,
                    outl.p.val_SI,
                    inl.fluid_data,
                    inl.mixing_rule,
                    T0=inl.T.val_SI
                )
                - inl.h.val_SI
            ) * self.eta_s_fit()
        )

    def eta_s_variables_columns(self):
        i = self.inl[0]
        o = self.outl[0]
        variables_columns1 = [data.J_col for data in [i.p, o.p, i.h, o.h] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def eta_s_take_effect(self):
        pass

    def eta_s_solve_isolated(self):
        return False

    def eta_s_func_doc(self, label):
        r"""
        Equation for given isentropic efficiency of a turbine.

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
            r'0=-\left(h_\mathrm{out}-h_\mathrm{in}\right)+\left('
            r'h_\mathrm{out,s}-h_\mathrm{in}\right)\cdot\eta_\mathrm{s}')
        return generate_latex_eq(self, latex, label)

    def eta_s_deriv(self, increment_filter, k):
        r"""
        Partial derivatives for isentropic efficiency function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        f = self.eta_s_func
        i = self.inl[0]
        o = self.outl[0]
        if self.is_variable(i.p, increment_filter):
            self.network.jacobian[k, i.p.J_col] = self.numeric_deriv(f, "p", i)
        if self.is_variable(o.p, increment_filter):
            self.network.jacobian[k, o.p.J_col] = self.numeric_deriv(f, "p", o)
        if self.is_variable(i.h, increment_filter):
            self.network.jacobian[k, i.h.J_col] = self.numeric_deriv(f, "h", i)
        if o.h.is_var:
            self.network.jacobian[k, o.h.J_col] = -1

    def eta_s_tensor(self, increment_filter, k):
        f = self.eta_s_func
        i = self.inl[0]
        o = self.outl[0]
        numeric_variables_list = (
                [('h', self.is_variable(c.h, increment_filter), c, c.h.J_col) for c in [i]] +
                [('p', self.is_variable(c.p, increment_filter), c, c.p.J_col) for c in [i, o]])
        self.generate_numerical_tensor(f, k, numeric_variables_list)

    def cone_func(self):
        r"""
        Equation for stodolas cone law.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = \frac{\dot{m}_{in,ref} \cdot p_{in}}{p_{in,ref}} \cdot
                \sqrt{\frac{p_{in,ref} \cdot v_{in}}{p_{in} \cdot v_{in,ref}}}
                \cdot \sqrt{\frac{1 - \left(\frac{p_{out}}{p_{in}} \right)^{2}}
                {1 - \left(\frac{p_{out,ref}}{p_{in,ref}} \right)^{2}}} -
                \dot{m}_{in}
        """
        n = 1
        i = self.inl[0]
        o = self.outl[0]
        vol = i.calc_vol(T0=i.T.val_SI)
        return (
            - i.m.val_SI + i.m.design * i.p.val_SI / i.p.design
            * (i.p.design * i.vol.design / (i.p.val_SI * vol)) ** 0.5
            * abs(
                    (1 - (o.p.val_SI / i.p.val_SI) ** ((n + 1) / n))
                    / (1 - (self.pr.design) ** ((n + 1) / n))
            ) ** 0.5
        )

    def eta_s_constant_func_(self, **kwargs):
        return self.eta_s.val_SI

    def eta_s_default_func_(self, **kwargs):
        pass

    def eta_s_charline_func_(self, **kwargs):
        if not self.eta_s_char.is_set:
            self.eta_s_char.char_func = CharLine(x=[0, 1], y=[1, 1])
        p = self.eta_s_char.param
        expr = self.get_char_expr(p, **self.eta_s_char.char_params)  # x: variable
        if not expr:
            msg = (
                "Please choose a valid parameter, you want to link the "
                f"isentropic efficiency to at component {self.label}."
            )
            logger.error(msg)
            raise ValueError(msg)
        eta = self.eta_s.design * self.eta_s_char.char_func.evaluate(expr)
        return eta

    def eta_s_charmap_func_(self, **kwargs):
        pass

    def eta_s_self_defined_func_(self, **kwargs):
        pass

    def pr_default_func_(self, pr='', inconn=0, outconn=0):
        pass

    def dp_default_func_(self, dp='', inconn=0, outconn=0):
        pass

    def convergence_check(self):
        r"""
        Perform a convergence check.

        Note
        ----
        Manipulate enthalpies/pressure at inlet and outlet if not specified by
        user to match physically feasible constraints.
        """
        i, o = self.inl[0], self.outl[0]

        if not i.good_starting_values:
            if i.p.val_SI <= 1e5 and i.p.is_var:
                i.p.val_SI = 1e5

            if i.h.val_SI < 10e5 and i.h.is_var:
                i.h.val_SI = 10e5

            if o.h.val_SI < 5e5 and o.h.is_var:
                o.h.val_SI = 5e5

        if i.h.val_SI <= o.h.val_SI and o.h.is_var:
            o.h.val_SI = i.h.val_SI * 0.9

        if i.p.val_SI <= o.p.val_SI and o.p.is_var:
            o.p.val_SI = i.p.val_SI * 0.9

    @staticmethod
    def initialise_source(c, key):
        r"""
        Return a starting value for pressure and enthalpy at outlet.

        Parameters
        ----------
        """
        if key == 'p':
            return 0.5e5
        elif key == 'h':
            return 1.5e6

    @staticmethod
    def initialise_target(c, key):
        r"""
        Return a starting value for pressure and enthalpy at inlet.

        Parameters
        ----------
        """
        if key == 'p':
            return 2.5e6
        elif key == 'h':
            return 2e6

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        super().calc_parameters()
        inl = self.inl[0]
        outl = self.outl[0]
        self.eta_s.val_SI = self.calc_eta_s()
        self.pr.val_SI = outl.p.val_SI / inl.p.val_SI

    def exergy_balance(self, T0):
        r"""
        Calculate exergy balance of a turbine.

        Parameters
        ----------
        T0 : float
            Ambient temperature T0 / K.

        Note
        ----
        .. math::

            \dot{E}_\mathrm{P} =
            \begin{cases}
            -P & T_\mathrm{in}, T_\mathrm{out} \geq T_0\\
            -P + \dot{E}_\mathrm{out}^\mathrm{T}
            & T_\mathrm{in} > T_0 \geq T_\mathrm{out}\\
            -P +\dot{E}_\mathrm{out}^\mathrm{T}- \dot{E}_\mathrm{in}^\mathrm{T}
            & T_0 \geq T_\mathrm{in}, T_\mathrm{out}\\
            \end{cases}

           \dot{E}_\mathrm{F} =
           \begin{cases}
           \dot{E}_\mathrm{in}^\mathrm{PH} - \dot{E}_\mathrm{out}^\mathrm{PH}
           & T_\mathrm{in}, T_\mathrm{out} \geq T_0\\
           \dot{E}_\mathrm{in}^\mathrm{T} + \dot{E}_\mathrm{in}^\mathrm{M} -
           \dot{E}_\mathrm{out}^\mathrm{M}
           & T_\mathrm{in} > T_0 \geq T_\mathrm{out}\\
           \dot{E}_\mathrm{in}^\mathrm{M} - \dot{E}_\mathrm{out}^\mathrm{M}
           & T_0 \geq T_\mathrm{in}, T_\mathrm{out}\\
           \end{cases}

           \dot{E}_\mathrm{bus} = -P
        """
        if self.inl[0].T.val_SI >= T0 and self.outl[0].T.val_SI >= T0:
            self.E_P = -self.P.val_SI
            self.E_F = self.inl[0].Ex_physical - self.outl[0].Ex_physical
        elif self.inl[0].T.val_SI > T0 and self.outl[0].T.val_SI <= T0:
            self.E_P = -self.P.val_SI + self.outl[0].Ex_therm
            self.E_F = self.inl[0].Ex_therm + (
                self.inl[0].Ex_mech - self.outl[0].Ex_mech)
        elif self.inl[0].T.val_SI <= T0 and self.outl[0].T.val_SI <= T0:
            self.E_P = -self.P.val_SI + (
                self.outl[0].Ex_therm - self.inl[0].Ex_therm)
            self.E_F = self.inl[0].Ex_mech - self.outl[0].Ex_mech
        else:
            msg = ('Exergy balance of a turbine, where outlet temperature is '
                   'larger than inlet temperature is not implmented.')
            logger.warning(msg)
            self.E_P = np.nan
            self.E_F = np.nan

        self.E_bus = {"chemical": 0, "physical": 0, "massless": -self.P.val_SI}
        self.E_D = self.E_F - self.E_P
        self.epsilon = self._calc_epsilon()
