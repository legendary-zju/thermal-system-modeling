# -*- coding: utf-8

"""Module of class Desuperheater.
"""

from Aurora.components.component import component_registry
from Aurora.components.fluid_components.heat_exchangers.base import HeatExchanger
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools.fluid_properties import d2h_mix_d2p_Q
from Aurora.tools.fluid_properties import h_mix_pQ, h_mix_pT, p_mix_hT, dT_mix_pdh
from Aurora.tools.fluid_properties import p_mix_hQ
from Aurora.tools.fluid_properties import p_sat_T
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools import logger


@component_registry
class Desuperheater(HeatExchanger):
    r"""
    The Desuperheater cools a fluid to the saturated gas state.

    **Mandatory Equations**

    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.energy_balance_func`
    - :py:meth:`AURORA.components.heat_exchangers.desuperheater.Desuperheater.saturated_gas_func`

    **Optional Equations**

    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.energy_balance_hot_func`
    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.kA_func`
    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.kA_char_func`
    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.DTU_func`
    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.DTL_func`
    - hot side :py:meth:`AURORA.components.component.Component.pr_func`
    - cold side :py:meth:`AURORA.components.component.Component.pr_func`
    - hot side :py:meth:`AURORA.components.component.Component.zeta_func`
    - cold side :py:meth:`AURORA.components.component.Component.zeta_func`

    Inlets/Outlets

    - in1, in2 (index 1: hot side, index 2: cold side)
    - out1, out2 (index 1: hot side, index 2: cold side)

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

    kA : float, dict
        Area independent heat transfer coefficient,
        :math:`kA/\frac{\text{W}}{\text{K}}`.

    kA_char1 : AURORA.tools.characteristics.CharLine, dict
        Characteristic line for hot side heat transfer coefficient.

    kA_char2 : AURORA.tools.characteristics.CharLine, dict
        Characteristic line for cold side heat transfer coefficient.

    Note
    ----
    The desuperheater has an additional equation for enthalpy at hot side
    outlet: The fluid leaves the component in saturated gas state.

    """

    @staticmethod
    def component():
        return 'desuperheater'

    def get_mandatory_constraints(self):
        if self.KDTA.is_set:
            constraints_dict = {
                'energy_balance_constraints': dc_cons(
                    func=self.energy_balance_func,
                    variables_columns=self.energy_balance_variables_columns,
                    deriv=self.energy_balance_deriv,
                    solve_isolated=self.energy_balance_solve_isolated,
                    tensor=self.energy_balance_tensor,
                    constant_deriv=False,
                    latex=self.energy_balance_func_doc,
                    num_eq=1,
                    scale=ps['m']['scale'] * ps['h']['scale']),
            }
        else:
            constraints_dict = {}
        constraints_dict.update({
            'saturated_gas_constraints': dc_cons(
                func=self.saturated_gas_func,
                variables_columns=self.saturated_gas_variables_columns,
                deriv=self.saturated_gas_deriv,
                tensor=self.saturated_gas_tensor,
                constant_deriv=False,
                latex=self.saturated_gas_func_doc,
                num_eq=1,
                scale=ps['h']['scale'])
        })
        return constraints_dict

    def saturated_gas_func(self):
        r"""
        Calculate hot side outlet state.

        Returns
        -------
        residual : float
            Residual value of equation

            .. math::

                0 = h_{out,1} - h\left(p_{out,1}, x=1 \right)
        """
        o = self.outl[0]
        return o.h.val_SI - h_mix_pQ(o.p.val_SI, 1, o.fluid_data)

    def saturated_gas_variables_columns(self):
        o = self.outl[0]
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [o.h] if data.is_var]  # [o.p, o.h]
        variables_columns1.sort()
        return [variables_columns1]

    def saturated_gas_solve_isolated(self):
        o = self.outl[0]
        if o.p.is_var and o.h.is_var:
            return False
        elif o.p.is_var and not o.h.is_var:
            o.p.val_SI = p_mix_hQ(o.h.val_SI, 1, o.fluid_data)
            o.p.is_set = True
            o.p.is_var = False
            return True
        elif o.h.is_var and not o.p.is_var:
            o.h.val_SI = h_mix_pQ(o.p.val_SI, 1, o.fluid_data)
            o.h.is_set = True
            o.h.is_var = False
            return True
        else:
            return True

    def saturated_gas_func_doc(self, label):
        r"""
        Calculate hot side outlet state.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        latex = r'0=h_\mathrm{out,1}-h\left(p_\mathrm{out,1}, x=1 \right)'
        return generate_latex_eq(self, latex, label)

    def saturated_gas_deriv(self, increment_filter, k):
        r"""
        Partial derivatives of saturated gas at hot side outlet function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        o = self.outl[0]
        # if self.is_variable(o.p):
        #     self.network.jacobian[k, o.p.J_col] = -dh_mix_dpQ(o.p.val_SI, 1, o.fluid_data)
        if self.is_variable(o.h):
            self.network.jacobian[k, o.h.J_col] = 1

    def saturated_gas_tensor(self, increment_filter, k):
        o = self.outl[0]
        if self.is_variable(o.p):
            self.network.tensor[o.p.J_col, o.p.J_col, k] = -d2h_mix_d2p_Q(o.p.val_SI, 1, o.fluid_data)

    def DTL_func(self):
        r"""
        Equation for lower terminal temperature difference.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::
                0 = DT_{L} - T_{out,1} + T_{in,2}
        """
        i = self.inl[1]
        o = self.outl[0]
        T_i2 = i.calc_T()
        T_o1 = o.calc_T_sat()
        return self.DTL.val_SI - T_o1 + T_i2

    def DTL_variables_columns(self):
        variables_colmns1 = [data.J_col for c in [self.inl[1]] for data in [c.h] if data.is_var]  # [c.p, c.h]
        variables_colmns1.sort()
        return [variables_colmns1]

    def DTL_solve_isolated(self):
        i = self.inl[1]
        o = self.outl[0]
        if i.fluid.is_var or o.fluid.is_var:
            return False
        if sum([1 if data.is_var else 0 for data in [i.p, i.h, o.p]]) > 1:  # [i.p, i.h, o.p, o.h]
            return False
        if i.h.is_var:
            T_i2 = o.calc_T_sat() - self.DTL.val_SI
            i.h.val_SI = h_mix_pT(i.p.val_SI, T_i2, i.fluid_data, i.mixing_rule)
            i.h.is_set = True
            i.h.is_var = False
            self.DTL.is_set = False
            return True
        elif i.p.is_var:
            T_i2 = o.calc_T_sat() - self.DTL.val_SI
            i.p.val_SI = p_mix_hT(i.h.val_SI, T_i2, i.fluid_data, i.mixing_rule)
            i.p.is_set = True
            i.p.is_var = False
            self.DTL.is_set = False
            return True
        elif o.p.is_var:
            T_o1 = i.calc_T() + self.DTL.val_SI
            o.p.val_SI = p_sat_T(T_o1, o.fluid_data)
            o.p.is_set = True
            o.p.is_var = False
            self.DTL.is_set = False
            return True
        else:
            self.DTL.is_set = False
            return True

    def DTL_deriv(self, increment_filter, k):
        """
        Calculate partial derivates of lower terminal temperature function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        i = self.inl[1]
        o = self.outl[0]
        if self.is_variable(i.h, increment_filter):
            self.network.jacobian[k, i.h.J_col] = dT_mix_pdh(i.p.val_SI, i.h.val_SI, i.fluid_data, i.mixing_rule, T0=i.T.val_SI)

    def DTL_repair_matrix(self, property_):
        i = self.inl[1]
        o = self.outl[0]
        if property_ == i.h:
            h0 = h_mix_pQ(i.p.val_SI, 0, i.fluid_data)
            h1 = h_mix_pQ(i.p.val_SI, 1, i.fluid_data)
            return abs(self.DTL.val_SI - o.calc_T_sat() + i.calc_T()) / max(i.h.val_SI - h0, h1 - i.h.val_SI)
        else:
            msg = f"variable: {property_.label} is not a valid property in DTL_repair_matrix of {self.__class__.__name__}: {self.label}"
            raise ValueError(msg)

    def boundary_check(self):
        o0 = self.outl[0]  # liquid side
        try:
            for c in [o0]:
                if c.p.val_SI > c.calc_p_critical():
                    c.p.val_SI = c.calc_p_critical() * 0.99
                    self.boundary_rectify = True
                    logger.debug(f'The pressure of connection: {c.label} in {self.__class__.__name__}: {self.label} above the critical pressure, '
                                 f'adjusting to {c.p.val_SI}')
        except ValueError as e:
            raise ValueError(f"The boundary check error in {self.__class__.__name__}: {self.label}" + str(e))
        super().boundary_check()

    def bounds_p_generate(self):
        o0 = self.outl[0]
        o0.p.max_val = o0.calc_p_critical()
