# -*- coding: utf-8

"""Module of class Compressor.
"""

import numpy as np

from Aurora.components.component import component_registry
from Aurora.components.fluid_components.turbomachinery.base import Turbomachine
from Aurora.tools import logger, CharLine, CharMap
from Aurora.tools.data_containers import ComponentCharacteristicMaps as dc_cm
from Aurora.tools.data_containers import ComponentCharacteristics as dc_cc
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import GroupedComponentProperties as dc_gcp
from Aurora.tools.data_containers import FitCoefficient as dc_fit
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools.fluid_properties import isentropic


@component_registry
class Compressor(Turbomachine):
    r"""
    Class for axial or radial compressor.

    **Mandatory Equations**

    - :py:meth:`Aurora.components.component.Component.fluid_func`
    - :py:meth:`Aurora.components.component.Component.mass_flow_func`

    **Optional Equations**

    - :py:meth:`Aurora.components.component.Component.pr_func`
    - :py:meth:`Aurora.components.turbomachinery.base.Turbomachine.energy_balance_func`
    - :py:meth:`Aurora.components.turbomachinery.compressor.Compressor.eta_s_func`
    - :py:meth:`Aurora.components.turbomachinery.compressor.Compressor.eta_s_char_func`
    - :py:meth:`Aurora.components.turbomachinery.compressor.Compressor.char_map_eta_s_func`
    - :py:meth:`Aurora.components.turbomachinery.compressor.Compressor.char_map_pr_func`

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

    char_map : Aurora.tools.characteristics.CharMap, dict
        Characteristic map for pressure rise and isentropic efficiency vs.
        nondimensional mass flow, see
        :py:class:`Aurora.tools.characteristics.CharMap` for further
        information. Provide a CompressorMap as function :code:`func`.

    igva : float, dict, :code:`"var"`
        Inlet guide vane angle, :math:`igva/^\circ`.

    """

    @staticmethod
    def component():
        return 'compressor'

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
            'igva': dc_cp(
                min_val=-90,
                max_val=90,
                d=1e-3,
                val=0,
                is_property=True,
                is_result=True,
                property_data=cpd['Angle'],
                SI_unit=cpd['Angle']['SI_unit'],
            ),
            'char_map_eta_s': dc_cm(

            ),
            'pr_fit': dc_fit(
                rule='constant',
                constant=self.pr_constant_func_,
                default=self.pr_default_func_,
                charline=self.pr_charline_func_,
                charmap=self.pr_charmap_func_,
            ),
            'dp_fit': dc_fit(
                rule='constant',
                constant=self.dp_constant_func_,
                default=self.dp_default_func_,
                charmap=self.dp_charmap_func_,
            ),
            'pr_char': dc_cc(
                param='v',
                char_params={'type': 'rel', 'inconn': 0, 'outconn': 0}
            ),
            'char_map_pr': dc_cm(

            ),
            'dp_char': dc_cc(
                param='v',
                char_params={'type': 'rel', 'inconn': 0, 'outconn': 0}
            ),
            'char_map_dp': dc_cm(

            ),
        })
        return parameters

    def eta_s_func(self):
        r"""
        Equation for given isentropic efficiency of a compressor.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = -\left( h_{out} - h_{in} \right) \cdot \eta_{s} +
                \left( h_{out,s} - h_{in} \right)
        """
        i = self.inl[0]
        o = self.outl[0]
        return (
            (o.h.val_SI - i.h.val_SI) * self.eta_s_fit() - (
                isentropic(
                    i.p.val_SI,
                    i.h.val_SI,
                    o.p.val_SI,
                    i.fluid_data,
                    i.mixing_rule,
                    T0=None
                ) - i.h.val_SI
            )
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
        Equation for given isentropic efficiency of a compressor.

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
            r'0 =-\left(h_\mathrm{out}-h_\mathrm{in}\right)\cdot'
            r'\eta_\mathrm{s}+\left(h_\mathrm{out,s}-h_\mathrm{in}\right)')
        return generate_latex_eq(self, latex, label)

    def eta_s_deriv(self, increment_filter, k):
        r"""
        Partial derivatives for isentropic efficiency.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        i = self.inl[0]
        o = self.outl[0]
        f = self.eta_s_func
        if self.is_variable(i.p, increment_filter):
            self.network.jacobian[k, i.p.J_col] = self.numeric_deriv(f, 'p', i)
        if self.is_variable(o.p, increment_filter):
            self.network.jacobian[k, o.p.J_col] = self.numeric_deriv(f, 'p', o)
        if self.is_variable(i.h, increment_filter):
            self.network.jacobian[k, i.h.J_col] = self.numeric_deriv(f, 'h', i)
        if self.is_variable(o.h, increment_filter):
            self.network.jacobian[k, o.h.J_col] = self.eta_s_fit()

    def eta_s_tensor(self, increment_filter, k):
        i = self.inl[0]
        o = self.outl[0]
        f = self.eta_s_func
        numeric_variables_list = (
                    [('h', self.is_variable(c.h, increment_filter), c, c.h.J_col) for c in [i]] +
                    [('p', self.is_variable(c.p, increment_filter), c, c.p.J_col) for c in [i, o]])
        self.generate_numerical_tensor(f, k, numeric_variables_list)

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
        return self.eta_s.design * self.eta_s_char.char_func.evaluate(expr)

    def eta_s_charmap_func_(self, **kwargs):
        i = self.inl[0]
        o = self.outl[0]
        x = np.sqrt(i.T.design / i.calc_T())
        y = (i.m.val_SI * i.p.design) / (i.m.design * i.p.val_SI * x)
        yarr, zarr = self.char_map_eta_s.char_func.evaluate_x(x)
        # value manipulation with igva
        if not self.igva.is_set:
            msg = f'Has not set igva property in {self.__class__.__name__}: {self.label}.'
            logger.error(msg)
            raise AttributeError(msg)
        yarr *= (1 - self.igva.val_SI / 100)
        zarr *= (1 - self.igva.val_SI ** 2 / 10000)
        eta_c = self.char_map_eta_s.char_func.evaluate_y(y, yarr, zarr)
        return eta_c * self.eta_s.design

    def eta_s_self_defined_func_(self, **kwargs):
        pass

    def pr_charmap_func_(self, pr='', inconn=0, outconn=0):
        i = self.inl[0]
        o = self.outl[0]
        x = np.sqrt(i.T.design / i.calc_T())
        y = (i.m.val_SI * i.p.design) / (i.m.design * i.p.val_SI * x)
        yarr, zarr = self.char_map_pr.char_func.evaluate_x(x)
        # value manipulation with igva
        if not self.igva.is_set:
            msg = f'Has not set igva property in {self.__class__.__name__}: {self.label}.'
            logger.error(msg)
            raise AttributeError(msg)
        yarr *= (1 - self.igva.val_SI / 100)
        zarr *= (1 - self.igva.val_SI / 100)
        pr_c = self.char_map_pr.char_func.evaluate_y(y, yarr, zarr)
        return pr_c * self.pr.design

    def dp_charmap_func_(self, dp='', inconn=0, outconn=0):
        i = self.inl[0]
        o = self.outl[0]
        x = np.sqrt(i.T.design / i.calc_T())
        y = (i.m.val_SI * i.p.design) / (i.m.design * i.p.val_SI * x)
        yarr, zarr = self.char_map_dp.char_func.evaluate_x(x)
        # value manipulation with igva
        if not self.igva.is_set:
            msg = f'Has not set igva property in {self.__class__.__name__}: {self.label}.'
            logger.error(msg)
            raise AttributeError(msg)
        yarr *= (1 - self.igva.val_SI / 100)
        zarr *= (1 - self.igva.val_SI / 100)
        dp_c = self.char_map_dp.char_func.evaluate_y(y, yarr, zarr)
        return dp_c * self.dp.design

    def convergence_check(self):
        r"""
        Perform a convergence check.

        Note
        ----
        Manipulate enthalpies/pressure at inlet and outlet if not specified by
        user to match physically feasible constraints.
        """
        i, o = self.inl, self.outl

        if o[0].p.is_var and o[0].p.val_SI < i[0].p.val_SI:
            o[0].p.val_SI = i[0].p.val_SI * 1.1

        if o[0].h.is_var and o[0].h.val_SI < i[0].h.val_SI:
            o[0].h.val_SI = i[0].h.val_SI * 1.1

        if i[0].p.is_var and o[0].p.val_SI < i[0].p.val_SI:
            i[0].p.val_SI = o[0].p.val_SI * 0.9

        if i[0].h.is_var and o[0].h.val_SI < i[0].h.val_SI:
            i[0].h.val_SI = o[0].h.val_SI * 0.9

    @staticmethod
    def initialise_source(c, key):
        r"""
        Return a starting value for pressure and enthalpy at outlet.

        Parameters
        ----------

        """
        if key == 'p':
            return 10e5
        elif key == 'h':
            return 6e5

    @staticmethod
    def initialise_target(c, key):
        r"""
        Return a starting value for pressure and enthalpy at inlet.

        Parameters
        ----------
        """
        if key == 'p':
            return 1e5
        elif key == 'h':
            return 4e5

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        super().calc_parameters()
        i = self.inl[0]
        o = self.outl[0]
        if o.h.val_SI == i.h.val_SI:
            if not self.eta_s.is_set:
                self.eta_s.val_SI = np.nan
        else:
            self.eta_s.val_SI = (isentropic(
                                            i.p.val_SI,
                                            i.h.val_SI,
                                            o.p.val_SI,
                                            i.fluid_data,
                                            i.mixing_rule,
                                            T0=None
                                        ) - self.inl[0].h.val_SI
                                ) / (o.h.val_SI - i.h.val_SI)

    def check_parameter_bounds(self):
        r"""Check parameter value limits."""
        super().check_parameter_bounds()

        for data in [self.char_map_pr, self.char_map_eta_s]:
            if data.is_set:
                x = np.sqrt(self.inl[0].T.design / self.inl[0].T.val_SI)
                y = (self.inl[0].m.val_SI * self.inl[0].p.design) / (
                    self.inl[0].m.design * self.inl[0].p.val_SI * x)
                yarr = data.char_func.get_domain_errors_x(x, self.label)
                yarr *= (1 - self.igva.val_SI / 100)
                data.char_func.get_domain_errors_y(y, yarr, self.label)

    def exergy_balance(self, T0):
        r"""
        Calculate exergy balance of a compressor.

        Parameters
        ----------
        T0 : float
            Ambient temperature T0 / K.

        Note
        ----
        .. math::

            \dot{E}_\mathrm{P} =
            \begin{cases}
            \dot{E}_\mathrm{out}^\mathrm{PH} - \dot{E}_\mathrm{in}^\mathrm{PH}
            & T_\mathrm{in}, T_\mathrm{out} \geq T_0\\
            \dot{E}_\mathrm{out}^\mathrm{T} + \dot{E}_\mathrm{out}^\mathrm{M} -
            \dot{E}_\mathrm{in}^\mathrm{M}
            & T_\mathrm{out} > T_0 \leq T_\mathrm{in}\\
            \dot{E}_\mathrm{out}^\mathrm{M} - \dot{E}_\mathrm{in}^\mathrm{M}
            & T_0 \geq T_\mathrm{in}, T_\mathrm{out}\\
            \end{cases}

            \dot{E}_\mathrm{F} =
            \begin{cases}
            P & T_\mathrm{in}, T_\mathrm{out} \geq T_0\\
            P + \dot{E}_\mathrm{in}^\mathrm{T}
            & T_\mathrm{out} > T_0 \leq T_\mathrm{in}\\
            P + \dot{E}_\mathrm{in}^\mathrm{T} -\dot{E}_\mathrm{out}^\mathrm{T}
            & T_0 \geq T_\mathrm{in}, T_\mathrm{out}\\
            \end{cases}

            \dot{E}_\mathrm{bus} = P
        """
        if self.inl[0].T.val_SI >= T0 and self.outl[0].T.val_SI >= T0:
            self.E_P = self.outl[0].Ex_physical - self.inl[0].Ex_physical
            self.E_F = self.P.val_SI
        elif self.inl[0].T.val_SI <= T0 and self.outl[0].T.val_SI > T0:
            self.E_P = self.outl[0].Ex_therm + (
                self.outl[0].Ex_mech - self.inl[0].Ex_mech)
            self.E_F = self.P.val_SI + self.inl[0].Ex_therm
        elif self.inl[0].T.val_SI <= T0 and self.outl[0].T.val_SI <= T0:
            self.E_P = self.outl[0].Ex_mech - self.inl[0].Ex_mech
            self.E_F = self.P.val_SI + (
                self.inl[0].Ex_therm - self.outl[0].Ex_therm)
        else:
            msg = ('Exergy balance of a compressor, where outlet temperature '
                   'is smaller than inlet temperature is not implmented.')
            logger.warning(msg)
            self.E_P = np.nan
            self.E_F = np.nan

        self.E_bus = {
            "chemical": 0, "physical": 0, "massless": self.P.val_SI
        }
        self.E_D = self.E_F - self.E_P
        self.epsilon = self._calc_epsilon()
