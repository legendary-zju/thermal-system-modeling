# -*- coding: utf-8

"""Module of class EvaporatorSimple.
"""

import numpy as np

from Aurora.components.component import component_registry
from Aurora.components.fluid_components.heat_exchangers.base import HeatExchanger
from Aurora.tools import logger
from Aurora.tools.characteristics import CharLine
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools.fluid_properties import h_mix_pT, p_mix_hQ, p_mix_hT
from Aurora.tools.fluid_properties import d2h_mix_d2p_Q
from Aurora.tools.fluid_properties import dT_mix_pdh
from Aurora.tools.fluid_properties import h_mix_pQ
from Aurora.tools.fluid_properties import T_mix_ph
from Aurora.tools.fluid_properties import T_sat_p, p_sat_T


@component_registry
class Evaporator(HeatExchanger):
    """
    Class for evaporator component used to generate vapour.

    If not set superheat_dT, not set the phase state constraint of outlet in cold side,
    not connect evaporator to vapour/liquid separate component, the working medium will be constrained to saturated vapour.

    Inlets/Outlets
    - in1, in2 (index 1: hot side (gas), index 2: cold side (liquid or two phase or saturated liquid))
    - out1, out2 (index 1: hot side (gas), index 2: cold side(two phase or saturated vapour))

    """

    @staticmethod
    def component():
        return 'evaporator simple'

    def initialise_source(self, c, key):
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

        """
        if key == 'p':
            if c.source_id == 'out1':
                return 1.1e5
            else:
                return 50e5
        elif key == 'h':
            if c.source_id == 'out1':  # gas side
                T = 400 + 273.15  #
                return h_mix_pT(c.p.val_SI, T, c.fluid_data, c.mixing_rule)
            else:  # steam side (be heated)
                return h_mix_pQ(c.p.val_SI, 1, c.fluid_data)

    def initialise_target(self, c, key):
        r"""
        Return a starting value for pressure and enthalpy at inlet.

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
                h\left(p, 300 \text{K} \right) & \text{key = 'h' at inlet 1}\\
                h\left(p, 220 \text{K} \right) & \text{key = 'h' at outlet 2}
                \end{cases}
        """
        if key == 'p':
            if c.target_id == 'in1':
                return 1.1e5
            else:
                return 50e5
        elif key == 'h':
            if c.target_id == 'in1':
                T = 300 + 273.15
                return h_mix_pT(c.p.val_SI, T, c.fluid_data, c.mixing_rule)
            else:
                T = 220 + 273.15
            return h_mix_pQ(c.p.val_SI, 0, c.fluid_data) * 0.99

    def get_parameters(self):
        params = super().get_parameters()
        params.update({
            'superheat_dT': dc_cp(
                min_val=0,
                func=self.superheat_dT_func,
                variables_columns=self.superheat_dT_variables_columns,
                solve_isolated=self.superheat_dT_solve_isolated,
                deriv=self.superheat_dT_deriv,
                repair_matrix=self.superheat_dT_repair_matrix,
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

    def get_mandatory_constraints(self):
        if self.KDTA.is_set:
            constraints_dict = {}
        else:
            constraints_dict = {
                'energy_balance_constraints': dc_cons(
                    func=self.energy_balance_func,
                    variables_columns=self.energy_balance_variables_columns,
                    solve_isolated=self.energy_balance_solve_isolated,
                    deriv=self.energy_balance_deriv,
                    tensor=self.energy_balance_tensor,
                    constant_deriv=False,
                    latex=self.energy_balance_func_doc,
                    num_eq=1,
                    scale=ps['m']['scale'] * ps['h']['scale']),
            }
        # determine whether to constraint the saturated phase state of outlet in cold side
        o = self.outl[1]
        if (self.superheat_dT.is_set or o.x.is_set or o.Td_bp.is_set or o.Td_dew.is_set
                or o.target.__class__.__name__ in ['DropletSeparator', 'Drum', 'EvaporateTank']):
            pass
        else:
            constraints_dict.update({
                'saturated_vapour_constraints': dc_cons(
                    func=self.saturated_vapour_func,
                    variables_columns=self.saturated_vapour_variables_columns,
                    solve_isolated=self.saturated_vapour_solve_isolated,
                    deriv=self.saturated_vapour_deriv,
                    tensor=self.saturated_vapour_tensor,
                    constant_deriv=False,
                    latex=self.saturated_vapour_func_doc,
                    num_eq=1,
                    scale=ps['h']['scale'])
            })
        if o.Td_dew.is_set:
            self.two_phase_cold_side = False
        else:
            self.two_phase_cold_side = True
        return constraints_dict

    def superheat_dT_func(self):
        """
        Calculate temperature difference between superheat temperature and saturated temperature in outlet of cold side.
        Measure the degree of superheat.

        :return:
        """
        o = self.outl[1]
        return o.calc_T() - T_sat_p(o.p.val_SI, o.fluid_data) - self.superheat_dT.val_SI

    def superheat_dT_variables_columns(self):
        o = self.outl[1]
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [o.h] if data.is_var]  # [o.p, o.h]
        variables_columns1.sort()
        return [variables_columns1]

    def superheat_dT_take_effect(self):
        pass

    def superheat_dT_solve_isolated(self):
        o = self.outl[1]
        if o.p.is_var and o.h.is_var:
            return False
        elif o.p.is_var and not o.h.is_var:
            return False
        elif o.h.is_var and not o.p.is_var:
            T_o2 = T_sat_p(o.p.val_SI, o.fluid_data) + self.superheat_dT.val_SI
            o.h.val_SI = h_mix_pT(o.p.val_SI, T_o2, o.fluid_data, o.mixing_rule)
            o.h.is_set = True
            o.h.is_var = False
            self.superheat_dT.is_set = False
            return True
        else:
            self.superheat_dT.is_set = False
            return True

    def superheat_dT_deriv(self, increment_filter, k):
        o = self.outl[1]
        if o.h.is_var:
            self.network.jacobian[k, o.h.J_col] = dT_mix_pdh(o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule)

    def superheat_dT_repair_matrix(self, property_):
        o = self.outl[1]
        if property_ == o.h:
            h0 = h_mix_pQ(o.p.val_SI, 0, o.fluid_data)
            h1 = h_mix_pQ(o.p.val_SI, 1, o.fluid_data)
            return abs(o.calc_T() - T_sat_p(o.p.val_SI, o.fluid_data) - self.superheat_dT.val_SI) / max(o.h.val_SI - h0, h1 - o.h.val_SI)
        else:
            msg = f"variable: {property_.label} is not a valid property in superheat_dT_repair_matrix of {self.__class__.__name__}: {self.label}"
            raise ValueError(msg)

    def saturated_vapour_func(self):
        r"""
        Calculate outlet liquid state.

        Returns
        -------
        residual : float
            Residual value of equation

            .. math::

                0 = h_{out,2} - h\left(p_{out,2}, x=1 \right)
        """
        o = self.outl[1]
        return o.h.val_SI - h_mix_pQ(o.p.val_SI, 1, o.fluid_data)

    def saturated_vapour_variables_columns(self):
        o = self.outl[1]
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [o.h] if data.is_var]  # [o.p, o.h]
        variables_columns1.sort()
        return [variables_columns1]

    def saturated_vapour_take_effect(self):
        pass

    def saturated_vapour_solve_isolated(self):
        o = self.outl[1]
        if o.p.is_var and o.h.is_var:
            return False
        elif o.p.is_var and not o.h.is_var:
            o.p.val_SI = p_mix_hQ(o.h.val_SI, 1, o.fluid_data)
            o.p.is_set = True
            o.p.is_var = False
            return True
        elif not o.p.is_var and o.h.is_var:
            o.h.val_SI = h_mix_pQ(o.p.val_SI, 1, o.fluid_data)
            o.h.is_set = True
            o.h.is_var = False
            return True
        else:
            return True

    def saturated_vapour_func_doc(self, label):
        r"""
        Calculate outlet liquid state.

        Parameters
        ----------
        label : str
            Label for equation.

        Returns
        -------
        latex : str
            LaTeX code of equations applied.
        """
        latex = r'0=h_\mathrm{out,2}-h\left(p_\mathrm{out,2}, x=1 \right)'
        return generate_latex_eq(self, latex, label)

    def saturated_vapour_deriv(self, increment_filter, k):
        r"""
        Partial derivatives of saturated liquid at mere outlet function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        o = self.outl[1]
        # if self.is_variable(o.p):
        #     self.network.jacobian[k, o.p.J_col] = -dh_mix_dpQ(o.p.val_SI, 1, o.fluid_data)
        if self.is_variable(o.h):
            self.network.jacobian[k, o.h.J_col] = 1

    def saturated_vapour_tensor(self, increment_filter, k):
        o = self.outl[1]
        if self.is_variable(o.p):
            self.network.tensor[o.p.J_col, o.p.J_col, k] = -d2h_mix_d2p_Q(o.p.val_SI, 1, o.fluid_data)

    def KDTA_constant_func_(self, **kwargs):
        if self.fA.is_set and self.hf1.is_set:
            return self.fA.val_SI * self.hf1.val_SI
        else:
            return self.KDTA.val_SI

    def KDTA_default_func_(self, **kwargs):
        alfa1 = (self.inl[0].m.val_SI / self.inl[0].m.design) ** self.exm1.val_SI
        kA = self.fA.design * (self.hf1.design * alfa1)
        return kA

    def KDTA_charline_func_(self, **kwargs):
        if not self.KDTA_char1.is_set:
            self.KDTA_char1.char_func = CharLine(x=[0, 1], y=[1, 1])
        p1 = self.KDTA_char1.param
        f1 = self.get_char_expr(p1, **self.KDTA_char1.char_params)
        alfa1 = self.KDTA_char1.char_func.evaluate(f1)
        kA = self.fA.design * (self.hf1.design * alfa1)
        return kA

    def KDTA_charmap_func_(self, **kwargs):
        pass

    def KDTA_self_defined_func_(self, **kwargs):
        pass

    def calc_DTM(self):
        i1 = self.inl[0]
        o1 = self.outl[0]
        i2 = self.inl[1]
        o2 = self.outl[1]
        if self.two_phase_cold_side:
            h_mid_l = h_mix_pQ((o2.p.val_SI + i2.p.val_SI) / 2, 0, i2.fluid_data)
            # temperature of saturated liquid point in cold side
            T_lower = T_sat_p((o2.p.val_SI + i2.p.val_SI) / 2, i2.fluid_data)
            # enthalpy of corresponding point in hot side
            h_mid_u1 = o1.h.val_SI + ((h_mid_l - i2.h.val_SI) * i2.m.val_SI / o1.m.val_SI)
            h_mid_u2 = i1.h.val_SI - ((o2.h.val_SI - h_mid_l) * i2.m.val_SI / o1.m.val_SI)
            h_mid_u = (h_mid_u1 + h_mid_u2) / 2
            T_uper = T_mix_ph((o1.p.val_SI + i1.p.val_SI) / 2, h_mid_u, o1.fluid_data, o1.mixing_rule, T0=o1.T.val_SI)
            delta_T = T_uper - T_lower
            return delta_T
        else:
            return super().DTM_func()

    def DTM_func(self):
        i1 = self.inl[0]
        o1 = self.outl[0]
        i2 = self.inl[1]
        o2 = self.outl[1]
        if self.two_phase_cold_side:
            h_mid_l = h_mix_pQ((o2.p.val_SI + i2.p.val_SI) / 2, 0, i2.fluid_data)
            # temperature of saturated liquid point in cold side
            T_lower = T_sat_p((o2.p.val_SI + i2.p.val_SI) / 2, i2.fluid_data)
            # enthalpy of corresponding point in hot side
            h_mid_u1 = o1.h.val_SI + ((h_mid_l - i2.h.val_SI) * i2.m.val_SI / o1.m.val_SI)
            h_mid_u2 = i1.h.val_SI - ((o2.h.val_SI - h_mid_l) * i2.m.val_SI / o1.m.val_SI)
            h_mid_u = (h_mid_u1 + h_mid_u2) / 2
            T_uper = T_mix_ph((o1.p.val_SI + i1.p.val_SI) / 2, h_mid_u, o1.fluid_data, o1.mixing_rule, T0=o1.T.val_SI)
            delta_T = T_uper - T_lower
            return self.DTM.val_SI - delta_T
        else:
            return super().DTM_func()

    def DTM_variables_columns(self):
        i1 = self.inl[0]
        o1 = self.outl[0]
        i2 = self.inl[1]
        o2 = self.outl[1]
        variables_columns1 = []
        data_list = [data for c in [i1, o1, i2, o2] for data in [c.h]]  # [c.p, c.h]
        # if i1.m == i2.m:
        #     data_list += [i1.m]
        # else:
        #     data_list += [i1.m, i2.m]
        variables_columns1 += [data.J_col for data in data_list if data.is_var]
        # variables_columns1 += [c.fluid.J_col[fluid] for c in [o1] for fluid in c.fluid.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def DTM_take_effect(self):
        pass

    def DTM_solve_isolated(self):
        return False

    def DTM_func_doc(self, label):
        latex = f'nothing now'
        return generate_latex_eq(self, latex, label)

    def DTM_deriv(self, increment_filter, k):
        f = self.DTM_func
        i1 = self.inl[0]
        o1 = self.outl[0]
        i2 = self.inl[1]
        o2 = self.outl[1]
        if self.is_variable(i1.h):
            self.network.jacobian[k, i1.h.J_col] = self.numeric_deriv(f, 'h', i1)
        if self.is_variable(o1.h):
            self.network.jacobian[k, o1.h.J_col] = self.numeric_deriv(f, 'h', o1)
        if self.is_variable(i2.h):
            self.network.jacobian[k, i2.h.J_col] = self.numeric_deriv(f, 'h', i2)
        if self.is_variable(o2.h):
            self.network.jacobian[k, o2.h.J_col] = self.numeric_deriv(f, 'h', o2)

    def DTM_repair_matrix(self, property_):
        i1 = self.inl[0]
        o1 = self.outl[0]
        i2 = self.inl[1]
        o2 = self.outl[1]
        if self.two_phase_cold_side:
            if property_ == i2.h:
                h0 = h_mix_pQ(i2.p.val_SI, 0, i2.fluid_data)
                h1 = h_mix_pQ(i2.p.val_SI, 1, i2.fluid_data)
                return abs(self.DTM_func()) / max(i2.h.val_SI - h0, h1 - i2.h.val_SI) * i2.m.val_SI / o1.m.val_SI
            elif property_ == o2.h:
                h0 = h_mix_pQ(o2.p.val_SI, 0, o2.fluid_data)
                h1 = h_mix_pQ(o2.p.val_SI, 1, o2.fluid_data)
                return abs(self.DTM_func()) / max(o2.h.val_SI - h0, h1 - o2.h.val_SI) * i2.m.val_SI / o1.m.val_SI
            elif property_ == i1.h:
                if "H2O" in i1.fluid.val:
                    fluid_c = "H2O"
                elif 'h2o' in i1.fluid.val:
                    fluid_c = "h2o"
                else:
                    fluid_c = 'water'
                fluid_dict = {fluid_c: {
                    "wrapper": i1.fluid.wrapper[fluid_c],
                    "mass_fraction": 1}}
                T_wator_sat = T_sat_p(i1.p.val_SI, fluid_dict)
                h0 = h_mix_pT(i1.p.val_SI, T_wator_sat - 1, i1.fluid_data, i1.mixing_rule)
                h1 = h_mix_pT(i1.p.val_SI, T_wator_sat + 1, i1.fluid_data, i1.mixing_rule)
                return -abs(self.DTM_func()) / max(i1.h.val_SI - h0, h1 - i1.h.val_SI)
            elif property_ == o1.h:
                if "H2O" in o1.fluid.val:
                    fluid_c = "H2O"
                elif 'h2o' in o1.fluid.val:
                    fluid_c = "h2o"
                else:
                    fluid_c = 'water'
                fluid_dict = {fluid_c: {
                    "wrapper": o1.fluid.wrapper[fluid_c],
                    "mass_fraction": 1}}
                T_wator_sat = T_sat_p(o1.p.val_SI, fluid_dict)
                h0 = h_mix_pT(o1.p.val_SI, T_wator_sat - 1, o1.fluid_data, o1.mixing_rule)
                h1 = h_mix_pT(o1.p.val_SI, T_wator_sat + 1, o1.fluid_data, o1.mixing_rule)
                return -abs(self.DTM_func()) / max(o1.h.val_SI - h0, h1 - o1.h.val_SI)
            else:
                msg = f"variable: {property_.label} is not a valid property in mdt_repair_matrix of {self.__class__.__name__}: {self.label}"
                raise ValueError(msg)
        else:
            return super().DTM_repair_matrix(property_)

    def DTM_tensor(self, increment_filter, k):
        f = self.DTM_func
        o1 = self.outl[0]
        i2 = self.inl[1]
        numeric_variables_list = ([('p', self.is_variable(c.p, increment_filter), c, c.p.J_col) for c in [o1, i2]] +
                                  [('h', self.is_variable(c.h, increment_filter), c, c.h.J_col) for c in [o1, i2]] +
                                  [(fluid, True, c, c.fluid.J_col[fluid]) for c in [o1] for fluid in c.fluid.is_var])
        self.generate_numerical_tensor(f, k, numeric_variables_list)

    def DTU_func(self):
        r"""
        Equation for upper terminal temperature difference.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::
                0 = DT_{U} - T_{in,1} + T_{out,2}
        """
        i = self.inl[0]
        o = self.outl[1]
        if self.superheat_dT.is_set or o.Td_bp.is_set or o.Td_dew.is_set:
            return super().DTU_func()
        else:
            T_i1 = i.calc_T()
            T_o2 = o.calc_T_sat()
            return self.DTU.val_SI - T_i1 + T_o2

    def DTU_variables_columns(self):
        i = self.inl[0]
        o = self.outl[1]
        if self.superheat_dT.is_set or o.Td_bp.is_set or o.Td_dew.is_set:
            return super().DTU_variables_columns()
        else:
            variables_colmns1 = [data.J_col for c in [i] for data in [c.h] if data.is_var]  # [c.p, c.h]
            variables_colmns1.sort()
            return [variables_colmns1]

    def DTU_solve_isolated(self):
        i = self.inl[0]
        o = self.outl[1]
        if i.fluid.is_var or o.fluid.is_var:
            return False
        if self.superheat_dT.is_set or o.Td_bp.is_set or o.Td_dew.is_set:
            return super().DTU_solve_isolated()
        else:
            if sum([1 if data.is_var else 0 for data in [i.p, o.p, i.h]]) > 1:  # [i.p, i.h, o.p, o.h]
                return False
            if i.p.is_var:
                T_i1 = self.DTU.val_SI + o.calc_T_sat()
                i.p.val_SI = p_mix_hT(i.h.val_SI, T_i1, i.fluid_data, i.mixing_rule)
                i.p.is_set = True
                i.p.is_var = False
                self.DTU.is_set = False
                return True
            elif o.p.is_var:
                T_o2 = i.calc_T() - self.DTU.val_SI
                o.p.val_SI = p_sat_T(T_o2, o.fluid_data)
                o.p.is_set = True
                o.p.is_var = False
                self.DTU.is_set = False
                return True
            elif i.h.is_var:
                T_i1 = self.DTU.val_SI + o.calc_T_sat()
                i.h.val_SI = h_mix_pT(i.p.val_SI, T_i1, i.fluid_data, i.mixing_rule)
                i.h.is_set = True
                i.h.is_var = False
                self.DTU.is_set = False
                return True
            else:
                self.DTU.is_set = False
                return True

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
        if self.superheat_dT.is_set or o.Td_bp.is_set or o.Td_dew.is_set:
            super().DTU_deriv(increment_filter, k)
        else:
            if self.is_variable(i.h, increment_filter):
                self.network.jacobian[k, i.h.J_col] = -dT_mix_pdh(i.p.val_SI, i.h.val_SI, i.fluid_data, i.mixing_rule, T0=i.T.val_SI)

    def DTU_repair_matrix(self, property_):
        i = self.inl[0]
        o = self.outl[1]
        if self.superheat_dT.is_set or o.Td_bp.is_set or o.Td_dew.is_set:
            return super().DTU_repair_matrix(property_)
        else:
            if property_ == i.h:
                h0 = h_mix_pQ(i.p.val_SI, 0, i.fluid_data)
                h1 = h_mix_pQ(i.p.val_SI, 1, i.fluid_data)
                return -abs(self.DTU.val_SI - i.calc_T() + o.calc_T_sat()) / max(i.h.val_SI - h0, h1 - i.h.val_SI)
            else:
                msg = f"variable: {property_.label} is not a valid property in DTU_repair_matrix of {self.__class__.__name__}: {self.label}"
                raise ValueError(msg)

    def DTL_func(self):
        r"""
        Equation for lower terminal temperature difference.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::
                0 = ttd_{l} - T_{out,1} + T_{in,2}
        """
        i = self.inl[1]
        o = self.outl[0]
        if i.x.is_set:
            T_i2 = i.calc_T_sat()
            T_o1 = o.calc_T_()
            return self.DTL.val_SI - T_o1 + T_i2
        else:
            return super().DTL_func()

    def DTL_variables_columns(self):
        i = self.inl[1]
        o = self.outl[0]
        if i.x.is_set:
            variables_colmns1 = [data.J_col for c in [o] for data in [c.h] if data.is_var]  # [c.p, c.h]
            variables_colmns1.sort()
            return [variables_colmns1]
        else:
            return super().DTL_variables_columns()

    def DTL_solve_isolated(self):
        i = self.inl[1]
        o = self.outl[0]
        if i.fluid.is_var or o.fluid.is_var:
            return False
        if i.x.is_set:
            if sum([1 if data.is_var else 0 for data in [i.p, o.h, o.p]]) > 1:  # [i.p, i.h, o.p, o.h]
                return False
            if o.h.is_var:
                T_o1 = i.calc_T_sat() + self.DTL.val_SI
                o.h.val_SI = h_mix_pT(o.p.val_SI, T_o1, o.fluid_data, o.mixing_rule)
                o.h.is_set = True
                o.h.is_var = False
                self.DTL.is_set = False
                return True
            elif i.p.is_var:
                T_i2 = o.calc_T() - self.DTL.val_SI
                i.p.val_SI = p_sat_T(T_i2, i.fluid_data)
                i.p.is_set = True
                i.p.is_var = False
                self.DTL.is_set = False
                return True
            elif o.p.is_var:
                T_o1 = i.calc_T_sat() + self.DTL.val_SI
                o.p.val_SI = p_mix_hT(o.h.val_SI, T_o1, o.fluid_data, o.mixing_rule)
                o.p.is_set = True
                o.p.is_var = False
                self.DTL.is_set = False
                return True
            else:
                self.DTL.is_set = False
                return True
        else:
            return super().DTL_solve_isolated()

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
        if i.x.is_set:
            if self.is_variable(o.h, increment_filter):
                self.network.jacobian[k, o.h.J_col] = -dT_mix_pdh(o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule, T0=o.T.val_SI)
        else:
            super().DTL_deriv(increment_filter, k)

    def DTL_repair_matrix(self, property_):
        i = self.inl[1]
        o = self.outl[0]
        if i.x.is_set:
            if property_ == o.h:
                h0 = h_mix_pQ(o.p.val_SI, 0, o.fluid_data)
                h1 = h_mix_pQ(o.p.val_SI, 1, o.fluid_data)
                return -abs(self.DTL.val_SI - o.calc_T_() + i.calc_T_sat()) / max(o.h.val_SI - h0, h1 - o.h.val_SI)
            else:
                msg = f"variable: {property_.label} is not a valid property in DTL_repair_matrix of {self.__class__.__name__}: {self.label}"
                raise ValueError(msg)
        else:
            return super().DTL_repair_matrix(property_)

    def boundary_check(self):
        o = self.outl[1]  # liquid side
        try:
            for c in [o]:
                if c.p.val_SI > c.calc_p_critical():
                    c.p.val_SI = c.calc_p_critical() * 0.99
                    self.boundary_rectify = True
                    logger.debug(f'The pressure of connection: {c.label} in {self.__class__.__name__}: {self.label} above the critical pressure, '
                                 f'adjusting to {c.p.val_SI}')
        except ValueError as e:
            raise ValueError(f"The boundary check error in {self.__class__.__name__}: {self.label}" + str(e))
        super().boundary_check()

    def bounds_p_generate(self):
        o = self.outl[1]
        o.p.max_val = o.calc_p_critical()

    def bounds_h_generate(self):
        o = self.outl[1]
        if not o.p.is_var:
            if o.p.val_SI < o.calc_p_critical():
                o.h.max_val = h_mix_pQ(o.p.val_SI, 1, o.fluid_data, o.mixing_rule) * 1.05
                o.h.min_val = h_mix_pQ(o.p.val_SI, 0, o.fluid_data, o.mixing_rule)

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        super().calc_parameters()
        if self.network.converged:
            self.fA.val_SI = self.KDTA.val_SI / self.hf1.val_SI
        else:
            self.fA.val_SI = np.nan
        try:
            o = self.outl[1]
            self.superheat_dT.val_SI = o.calc_T() - T_sat_p(o.p.val_SI, o.fluid_data)
        except ValueError:
            self.superheat_dT.val_SI = np.nan

