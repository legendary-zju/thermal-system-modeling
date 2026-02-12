# -*- coding: utf-8

"""Module of class ExtractHeatExchanger.
"""

import numpy as np

from Aurora.components.component import component_registry
from Aurora.components.fluid_components.heat_exchangers.base import HeatExchanger
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools.fluid_properties import h_mix_pT, p_mix_hQ, p_mix_hT
from Aurora.tools.fluid_properties import d2h_mix_d2p_Q
from Aurora.tools.fluid_properties import dT_mix_pdh
from Aurora.tools.fluid_properties import h_mix_pQ
from Aurora.tools.fluid_properties import T_sat_p, p_sat_T


@component_registry
class ExtractHeatExchanger(HeatExchanger):
    """
    Class for heat exchanger component used to cool vapour extracted.

    Inlets/Outlets
    - in1, in2 (index 1: hot side (extract vapour), index 2: cold side (reheated liquid))
    - out1, out2 (index 1: hot side (extract vapour), index 2: cold side(reheated liquid))

    """

    @staticmethod
    def component():
        return 'extract heat exchanger'

    def initialise_source(self, c, key):  #
        r"""
        Return a starting value for pressure and enthalpy at outlet.
        """
        if key == 'p':  # 50e5  !!!
            return 10e5 * 0.11
        elif key == 'h':
            if c.source_id == 'out1':  # saturated liquid
                return h_mix_pQ(c.p.val_SI, 0, c.fluid_data)
            else:
                T = 100 + 273.15
            return h_mix_pT(c.p.val_SI, T, c.fluid_data, c.mixing_rule)

    def get_parameters(self):
        params = super().get_parameters()
        params.update({
            'saturated_cooling': dc_simple(
                val=True,
                func=self.saturated_cooling_func,
                variables_columns=self.saturated_cooling_variables_columns,
                solve_isolated=self.saturated_cooling_solve_isolated,
                latex=self.saturated_cooling_func_doc,
                deriv=self.saturated_cooling_deriv,
                tensor=self.saturated_cooling_tensor,
                num_eq=1,
                scale=ps['h']['scale']),
            'DTU_sh': dc_cp(
                min_val=0,
                func=self.DTU_sh_func,
                variables_columns=self.DTU_sh_variables_columns,
                solve_isolated=self.DTU_sh_solve_isolated,
                deriv=self.DTU_sh_deriv,
                repair_matrix=self.DTU_sh_repair_matrix,
                tensor=self.DTU_sh_tensor,
                latex=None,
                num_eq=1,
                property_data=cpd['DT'],
                SI_unit=cpd['DT']['SI_unit'],
                scale=ps['DT']['scale'],
                var_scale=ps['DT']['scale']
            ),
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
        # if saturated_cooling is True, outlet state method must be calculated
        i = self.inl[0]
        o = self.outl[0]
        # determine whether to constraint the saturated phase state of outlet in hot side
        if (self.supercooling_dT.is_set or o.x.is_set or o.Td_bp.is_set or o.Td_dew.is_set
                or o.target.__class__.__name__ in ['DropletSeparator', 'Drum', 'EvaporateTank']):
            pass
        else:
            self.saturated_cooling.is_set = self.saturated_cooling.val
        # confirm phase state in hot side
        if (self.supercooling_dT.is_set or o.x.is_set or o.Td_dew.is_set or self.saturated_cooling.is_set
                or (i.source.__class__.__name__ in ['DropletSeparator', 'Drum', 'EvaporateTank'] and i.source_id == 'out2')):
            self.two_phase_hot_side = True
        else:
            self.two_phase_hot_side = False
        super().summarize_equations()

    def saturated_cooling_func(self):
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
        This equation is applied in case saturated_cooling is True!
        """
        o = self.outl[0]
        return o.h.val_SI - h_mix_pQ(o.p.val_SI, 0, o.fluid_data)

    def saturated_cooling_variables_columns(self):
        o = self.outl[0]
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [o.h] if data.is_var]  # [o.p, o.h]
        variables_columns1.sort()
        return [variables_columns1]

    def saturated_cooling_take_effect(self):
        pass

    def saturated_cooling_solve_isolated(self):
        o = self.outl[0]
        if o.p.is_var and o.h.is_var:
            return False
        elif o.p.is_var and not o.h.is_var:
            o.p.val_SI = p_mix_hQ(o.h.val_SI, 0, o.fluid_data)
            o.p.is_set = True
            o.p.is_var = False
            self.saturated_cooling.is_set = False
            return True
        elif o.h.is_var and not o.p.is_var:
            o.h.val_SI = h_mix_pQ(o.p.val_SI, 0, o.fluid_data)
            o.h.is_set = True
            o.h.is_var = False
            self.saturated_cooling.is_set = False
            return True
        else:
            self.saturated_cooling.is_set = False
            return True

    def saturated_cooling_func_doc(self, label):
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

    def saturated_cooling_deriv(self, increment_filter, k):
        """
        Calculate partial derivates of saturated_cooling function.

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

    def saturated_cooling_tensor(self, increment_filter, k):
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
        if self.saturated_cooling.is_set:
            i = self.inl[1]
            o = self.outl[0]
            T_i2 = i.calc_T()
            T_o1 = o.calc_T_sat()
            return self.DTL.val_SI - T_o1 + T_i2
        else:
            return super().DTL_func()

    def DTL_variables_columns(self):
        if self.saturated_cooling.is_set:
            variables_colmns1 = [data.J_col for c in [self.inl[1]] for data in [c.h] if data.is_var]  # [c.p, c.h]
            variables_colmns1.sort()
            return [variables_colmns1]
        else:
            return super().DTL_variables_columns()

    def DTL_solve_isolated(self):
        i = self.inl[1]
        o = self.outl[0]
        if i.fluid.is_var or o.fluid.is_var:
            return False
        if self.saturated_cooling.is_set:
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
        if self.saturated_cooling.is_set:
            if self.is_variable(i.h, increment_filter):
                self.network.jacobian[k, i.h.J_col] = dT_mix_pdh(i.p.val_SI, i.h.val_SI, i.fluid_data, i.mixing_rule, T0=i.T.val_SI)
        else:
            super().DTL_deriv(increment_filter, k)

    def DTL_repair_matrix(self, property_):
        i = self.inl[1]
        o = self.outl[0]
        if self.saturated_cooling.is_set or o.x.is_set:
            if property_ == i.h:
                h0 = h_mix_pQ(i.p.val_SI, 0, i.fluid_data)
                h1 = h_mix_pQ(i.p.val_SI, 1, i.fluid_data)
                return abs(self.DTL.val_SI - o.calc_T_sat() + i.calc_T()) / max(i.h.val_SI - h0, h1 - i.h.val_SI)
            else:
                msg = f"variable: {property_.label} is not a valid property in DTL_repair_matrix of {self.__class__.__name__}: {self.label}"
                raise ValueError(msg)
        else:
            return super().DTL_repair_matrix(property_)

    def bounds_p_generate(self):
        o0 = self.outl[0]
        o0.p.max_val = o0.calc_p_critical()

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        super().calc_parameters()
        try:
            i = self.inl[0]
            o = self.outl[1]
            self.DTU_sh.val_SI = i.calc_T_sat() - o.calc_T()
        except ValueError:
            self.DTU_sh.val_SI = np.nan
        try:
            o = self.outl[0]
            self.supercooling_dT.val_SI = T_sat_p(o.p.val_SI, o.fluid_data) - o.calc_T()
        except ValueError:
            self.supercooling_dT.val_SI = np.nan






