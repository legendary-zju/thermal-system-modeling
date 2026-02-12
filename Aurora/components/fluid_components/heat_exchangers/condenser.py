# -*- coding: utf-8

"""Module of class Condenser.
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
class Condenser(HeatExchanger):
    r"""
    A Condenser cools a fluid until it is in liquid state.

    The condensing fluid is cooled by the cold side fluid. The fluid on the hot
    side of the condenser must be pure. Subcooling is available.

    **Mandatory Equations**

    - :py:meth:`AURORA.components.component.Component.fluid_func`
    - :py:meth:`AURORA.components.component.Component.mass_flow_func`
    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.energy_balance_func`
    - condensate outlet state, function can be disabled by specifying
      :code:`set_attr(subcooling=True)`
      :py:meth:`AURORA.components.heat_exchangers.condenser.Condenser.subcooling_func`

    **Optional Equations**

    - :py:meth:`AURORA.components.heat_exchangers.base.HeatExchanger.energy_balance_hot_func`
    - :py:meth:`AURORA.components.heat_exchangers.condenser.Condenser.kA_func`
    - :py:meth:`AURORA.components.heat_exchangers.condenser.Condenser.kA_char_func`
    - :py:meth:`AURORA.components.heat_exchangers.condenser.Condenser.DTU_func`
    - :py:meth:`AURORA.components.heat_exchangers.condenser.Condenser.DTL_func`
    - :py:meth:`AURORA.components.heat_exchangers.condenser.Condenser.supercooling_dT_func`
    - hot side :py:meth:`AURORA.components.component.Component.pr_func`
    - cold side :py:meth:`AURORA.components.component.Component.pr_func`
    - hot side :py:meth:`AURORA.components.component.Component.zeta_func`
    - cold side :py:meth:`AURORA.components.component.Component.zeta_func`

    Inlets/Outlets

    - in1, in2 (index 1: hot side (superheated vapour or two_phase steam), index 2: cold side)
    - out1, out2 (index 1: hot side (saturated or supercooling liquid), index 2: cold side)

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
        Upper terminal temperature difference (referring to saturation
        temprature of condensing fluid) :math:`DT_\mathrm{U}/\text{K}`.

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

    subcooling : boolean
        Enable/disable subcooling, default value: disabled.

    supercooling_dT : float, dict
        Temperature difference about supercooling temperature and saturated temperature at outlet in hot side.

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
        return 'condenser'

    def get_parameters(self):
        params = super().get_parameters()
        params.update({
            'subcooling': dc_simple(
                val=False,
                func=self.subcooling_func,
                variables_columns=self.subcooling_variables_columns,
                solve_isolated=self.subcooling_solve_isolated,
                latex=self.subcooling_func_doc,
                deriv=self.subcooling_deriv,
                tensor=self.subcooling_tensor,
                num_eq=1,
                scale=ps['h']['scale']),
            'supercooling_dT': dc_cp(
                min_val=0,
                func=self.supercooling_dT_func,
                variables_columns=self.supercooling_dT_variables_columns,
                solve_isolated=self.supercooling_dT_solve_isolated,
                deriv=self.supercooling_dT_deriv,
                repair_matrix=self.supercooling_dT_repair_matrix,
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

    def summarize_equations(self):
        # if subcooling is True, outlet state method must not be calculated
        i = self.inl[0]
        o = self.outl[0]
        # determine whether to constraint the saturated phase state of outlet in hot side
        if (self.supercooling_dT.is_set or o.x.is_set or o.Td_bp.is_set or o.Td_dew.is_set
                or o.target.__class__.__name__ in ['DropletSeparator', 'Drum', 'EvaporateTank']):
            pass
        else:
            self.subcooling.is_set = not self.subcooling.val
        # confirm phase state in hot side
        if (self.supercooling_dT.is_set or o.x.is_set or o.Td_dew.is_set or self.subcooling.is_set
                or (i.source.__class__.__name__ in ['DropletSeparator', 'Drum',
                                                    'EvaporateTank'] and i.source_id == 'out2')):
            self.two_phase_hot_side = True
        else:
            self.two_phase_hot_side = False
        super().summarize_equations()

    def initialise_source(self, c, key):  #
        r"""
        Return a starting value for pressure and enthalpy at outlet.

        Parameters
        ----------
        c : tespy.connections.connection.Connection
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
                h\left(p, 200 \text{K} \right) & \text{key = 'h' at outlet 1}\\
                h\left(p, 250 \text{K} \right) & \text{key = 'h' at outlet 2}
                \end{cases}
        """
        if key == 'p':  # 50e5  !!!
            return 10e5 * 0.11
        elif key == 'h':
            if c.source_id == 'out1':  # saturated liquid
                return h_mix_pQ(c.p.val_SI, 0, c.fluid_data)
            else:
                T = 100 + 273.15
            return h_mix_pT(c.p.val_SI, T, c.fluid_data, c.mixing_rule)

    def subcooling_func(self):
        r"""
        Equation for hot side outlet state.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0=h_{out,1} -h\left(p_{out,1}, x=0 \right)

        Note
        ----
        This equation is applied in case subcooling is False!
        """
        o = self.outl[0]
        return o.h.val_SI - h_mix_pQ(o.p.val_SI, 0, o.fluid_data)

    def subcooling_variables_columns(self):
        o = self.outl[0]
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [o.h] if data.is_var]  # [o.p, o.h]
        variables_columns1.sort()
        return [variables_columns1]

    def subcooling_take_effect(self):
        pass

    def subcooling_solve_isolated(self):
        o = self.outl[0]
        if o.p.is_var and o.h.is_var:
            return False
        elif o.p.is_var and not o.h.is_var:
            o.p.val_SI = p_mix_hQ(o.h.val_SI, 0, o.fluid_data)
            o.p.is_set = True
            o.p.is_var = False
            self.subcooling.is_set = False
            return True
        elif o.h.is_var and not o.p.is_var:
            o.h.val_SI = h_mix_pQ(o.p.val_SI, 0, o.fluid_data)
            o.h.is_set = True
            o.h.is_var = False
            self.subcooling.is_set = False
            return True
        else:
            self.subcooling.is_set = False
            return True

    def subcooling_func_doc(self, label):
        r"""
        Equation for hot side outlet state.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        latex = r'0=h_\mathrm{out,1} -h\left(p_\mathrm{out,1}, x=0 \right)'
        return generate_latex_eq(self, latex, label)

    def subcooling_deriv(self, increment_filter, k):
        """
        Calculate partial derivates of subcooling function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        o = self.outl[0]
        # if self.is_variable(o.p):
        #     self.network.jacobian[k, o.p.J_col] = -dh_mix_dpQ(o.p.val_SI, 0, o.fluid_data)
        if self.is_variable(o.h):
            self.network.jacobian[k, o.h.J_col] = 1

    def subcooling_tensor(self, increment_filter, k):
        o = self.outl[0]
        if self.is_variable(o.p):
            self.network.tensor[o.p.J_col, o.p.J_col, k] = -d2h_mix_d2p_Q(o.p.val_SI, 0, o.fluid_data)

    def supercooling_dT_func(self):
        """
        Measure the degree of supercooling.
        The temperature of outlet of hot side under the temperature of saturated liquid in same pressure.

        :return:
        """
        o = self.outl[0]
        return T_sat_p(o.p.val_SI, o.fluid_data) - o.calc_T() - self.supercooling_dT.val_SI

    def supercooling_dT_variables_columns(self):
        o = self.outl[0]
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [o.h] if data.is_var]  # [o.p, o.h]
        variables_columns1.sort()
        return [variables_columns1]

    def supercooling_dT_take_effect(self):
        pass

    def supercooling_dT_solve_isolated(self):
        o = self.outl[0]
        if o.p.is_var and o.h.is_var:
            return False
        elif o.p.is_var and not o.h.is_var:
            return False
        elif o.h.is_var and not o.p.is_var:
            T_o1 = T_sat_p(o.p.val_SI, o.fluid_data) - self.supercooling_dT.val_SI
            o.h.val_SI = h_mix_pT(o.p.val_SI, T_o1, o.fluid_data, o.mixing_rule)
            o.h.is_set = True
            o.h.is_var = False
            self.supercooling_dT.is_set = False
            return True
        else:
            self.supercooling_dT.is_set = False
            return True

    def supercooling_dT_deriv(self, increment_filter, k):
        o = self.outl[0]
        if o.h.is_var:
            self.network.jacobian[k, o.h.J_col] = -dT_mix_pdh(o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule)

    def supercooling_dT_repair_matrix(self, property_):
        o = self.outl[0]
        if property_ == o.h:
            h0 = h_mix_pQ(o.p.val_SI, 0, o.fluid_data)
            h1 = h_mix_pQ(o.p.val_SI, 1, o.fluid_data)
            return -abs(T_sat_p(o.p.val_SI, o.fluid_data) - o.calc_T() - self.supercooling_dT.val_SI) / max(o.h.val_SI - h0, h1 - o.h.val_SI)
        else:
            msg = f"variable: {property_.label} is not a valid property in supercooling_dT_repair_matrix of {self.__class__.__name__}: {self.label}"
            raise ValueError(msg)

    def KDTA_constant_func_(self, **kwargs):
        if self.fA.is_set and self.hf2.is_set:
            return self.fA.val_SI * self.hf2.val_SI
        else:
            return self.KDTA.val_SI

    def KDTA_default_func_(self, **kwargs):
        alfa2 = (self.inl[1].m.val_SI / self.inl[1].m.design) ** self.exm2.val_SI
        kA = self.fA.design * (self.hf2.design * alfa2)
        return kA

    def KDTA_charline_func_(self, **kwargs):
        if not self.KDTA_char2.is_set:
            self.KDTA_char2.char_func = CharLine(x=[0, 1], y=[1, 1])
        p2 = self.KDTA_char2.param
        f2 = self.get_char_expr(p2, **self.KDTA_char2.char_params)
        alfa2 = self.KDTA_char2.char_func.evaluate(f2)
        kA = self.fA.design * (self.hf2.design * alfa2)
        return kA

    def KDTA_charmap_func_(self, **kwargs):
        pass

    def KDTA_self_defined_func_(self, **kwargs):
        pass

    def calculate_DT_log(self):
        i1 = self.inl[0]
        i2 = self.inl[1]
        o1 = self.outl[0]
        o2 = self.outl[1]

        # T_i1 = i1.calc_T_sat()  # !!!!!
        T_i1 = i1.calc_T()
        T_i2 = i2.calc_T()
        T_o1 = o1.calc_T()
        T_o2 = o2.calc_T()

        if T_i1 <= T_o2 and not i1.T.is_set:  # ????
            T_i1 = T_o2 + 0.5
        if T_i1 <= T_o2 and not o2.T.is_set:
            T_o2 = T_i1 - 0.5
        if T_o1 <= T_i2 and not o1.T.is_set:
            T_o1 = T_i2 + 1
        if T_o1 <= T_i2 and not i2.T.is_set:
            T_i2 = T_o1 - 1

        DTU = T_i1 - T_o2
        DTL = T_o1 - T_i2

        if DTU == DTL:
            DT_log = DTL
        else:
            DT_log = (DTL - DTU) / math.log((DTL) / (DTU))
        return DT_log

    def DTU_func(self):
        r"""
        Equation for upper terminal temperature difference.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = DT_{U} - T_{sat} \left(p_{in,1}\right) + T_{out,2}

        Note
        ----
        The upper terminal temperature difference DTU refers to boiling
        temperature at hot side inlet.
        """
        i = self.inl[0]
        o = self.outl[1]
        if i.x.is_set:
            T_i1 = i.calc_T_sat()
            T_o2 = o.calc_T()
            return self.DTU.val_SI - T_i1 + T_o2
        else:
            return super().DTU_func()

    def DTU_variables_columns(self):
        i = self.inl[0]
        o = self.outl[1]
        if i.x.is_set:
            variables_colmns1 = [data.J_col for c in [o] for data in [c.h] if data.is_var]  # [c.p, c.h]
            variables_colmns1.sort()
            return [variables_colmns1]
        else:
            return super().DTU_variables_columns()

    def DTU_solve_isolated(self):
        i = self.inl[0]
        o = self.outl[1]
        if i.fluid.is_var or o.fluid.is_var:
            return False
        if i.x.is_set:
            if sum([1 if data.is_var else 0 for data in [i.p, o.p, o.h]]) > 1:  # [i.p, i.h, o.p, o.h]
                return False
            if i.p.is_var:
                T_i1 = self.DTU.val_SI + o.calc_T()
                i.p.val_SI = p_sat_T(T_i1, i.fluid_data)
                i.p.is_set = True
                i.p.is_var = False
                self.DTU.is_set = False
                return True
            elif o.p.is_var:
                T_o2 = i.calc_T_sat() - self.DTU.val_SI
                o.p.val_SI = p_mix_hT(o.h.val_SI, T_o2, o.fluid_data, o.mixing_rule)
                o.p.is_set = True
                o.p.is_var = False
                self.DTU.is_set = False
                return True
            elif o.h.is_var:
                T_o2 = i.calc_T_sat() - self.DTU.val_SI
                o.h.val_SI = h_mix_pT(o.p.val_SI, T_o2, o.fluid_data, o.mixing_rule)
                o.h.is_set = True
                o.h.is_var = False
                self.DTU.is_set = False
                return True
            else:
                self.DTU.is_set = False
                return True
        else:
            return super().DTU_solve_isolated()

    def DTU_func_doc(self, label):
        r"""
        Equation for upper terminal temperature difference.

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
            r'0=DT_\mathrm{U}-T_\mathrm{sat}\left(p_\mathrm{in,1}\right)'
            r' + T_\mathrm{out,2}')
        return generate_latex_eq(self, latex, label)

    def DTU_deriv(self, increment_filter, k):
        """
        Calculate partial derivates of upper terminal temperature function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        i = self.inl[0]
        o = self.outl[1]
        if i.x.is_set:
            if self.is_variable(o.h, increment_filter):
                self.network.jacobian[k, o.h.J_col] = dT_mix_pdh(o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule, T0=o.T.val_SI)
        else:
            super().DTU_deriv(increment_filter, k)
        # if self.is_variable(i.p, increment_filter):
        #     # self.network.jacobian[k, i.p.J_col] = -dT_sat_dp(i.p.val_SI, i.fluid_data, i.mixing_rule)
        #     self.network.jacobian[k, i.p.J_col] = -dT_mix_dph(i.p.val_SI, i.h.val_SI, i.fluid_data, i.mixing_rule, T0=i.T.val_SI)
        # if self.is_variable(o.p, increment_filter):
        #     self.network.jacobian[k, o.p.J_col] = dT_mix_dph(o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule, T0=o.T.val_SI)

    def DTU_repair_matrix(self, property_):
        i = self.inl[0]
        o = self.outl[1]
        if i.x.is_set:
            if property_ == o.h:
                h0 = h_mix_pQ(o.p.val_SI, 0, o.fluid_data)
                h1 = h_mix_pQ(o.p.val_SI, 1, o.fluid_data)
                return abs(self.DTU.val_SI - i.calc_T_sat() + o.calc_T()) / max(o.h.val_SI - h0, h1 - o.h.val_SI)
            else:
                msg = f"variable: {property_.label} is not a valid property in DTU_repair_matrix of {self.__class__.__name__}: {self.label}"
                raise ValueError(msg)
        else:
            return super().DTU_repair_matrix(property_)

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
        if self.subcooling.is_set or o.x.is_set:
            T_i2 = i.calc_T()
            T_o1 = o.calc_T_sat()
            return self.DTL.val_SI - T_o1 + T_i2
        else:
            return super().DTL_func()

    def DTL_variables_columns(self):
        i = self.inl[1]
        o = self.outl[0]
        if self.subcooling.is_set or o.x.is_set:
            variables_colmns1 = [data.J_col for c in [i] for data in [c.h] if data.is_var]  # [c.p, c.h]
            variables_colmns1.sort()
            return [variables_colmns1]
        else:
            return super().DTL_variables_columns()

    def DTL_solve_isolated(self):
        i = self.inl[1]
        o = self.outl[0]
        if i.fluid.is_var or o.fluid.is_var:
            return False
        if self.subcooling.is_set or o.x.is_set:
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
        else:
            return super().DTL_solve_isolated()

    def DTL_repair_matrix(self, property_):
        i = self.inl[1]
        o = self.outl[0]
        if self.subcooling.is_set or o.x.is_set:
            if property_ == i.h:
                h0 = h_mix_pQ(i.p.val_SI, 0, i.fluid_data)
                h1 = h_mix_pQ(i.p.val_SI, 1, i.fluid_data)
                return abs(self.DTL.val_SI - o.calc_T_sat() + i.calc_T()) / max(i.h.val_SI - h0, h1 - i.h.val_SI)
            else:
                msg = f"variable: {property_.label} is not a valid property in DTL_repair_matrix of {self.__class__.__name__}: {self.label}"
                raise ValueError(msg)
        else:
            return super().DTL_repair_matrix(property_)

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
        if self.subcooling.is_set or o.x.is_set:
            if self.is_variable(i.h, increment_filter):
                self.network.jacobian[k, i.h.J_col] = dT_mix_pdh(i.p.val_SI, i.h.val_SI, i.fluid_data, i.mixing_rule, T0=i.T.val_SI)
        else:
            super().DTL_deriv(increment_filter, k)

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

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        super().calc_parameters()
        if self.network.converged:
            self.fA.val_SI = self.KDTA.val_SI / self.hf2.val_SI
        else:
            self.fA.val_SI = np.nan
        try:
            o = self.outl[0]
            self.supercooling_dT.val_SI = T_sat_p(o.p.val_SI, o.fluid_data) - o.calc_T()
        except ValueError:
            self.supercooling_dT.val_SI = np.nan

