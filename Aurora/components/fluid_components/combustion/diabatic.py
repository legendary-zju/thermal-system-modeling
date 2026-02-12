# -*- coding: utf-8

"""Module of class CombustionChamber.
"""

import numpy as np

from Aurora.components import CombustionChamber
from Aurora.components.component import component_registry
from Aurora.tools import logger
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.data_containers import FitCoefficient as dc_fit
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.global_vars import fluid_property_data as fpd
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools.fluid_properties import h_mix_pT, p_mix_hT, dT_mix_pdh


@component_registry
class DiabaticCombustionChamber(CombustionChamber):
    r"""
    The class CombustionChamber is parent class of all combustion components.

    **Mandatory Equations**

    - :py:meth:`AURORA.components.combustion.base.CombustionChamber.mass_flow_func`
    - :py:meth:`AURORA.components.combustion.base.CombustionChamber.combustion_pressure_func`
    - :py:meth:`AURORA.components.combustion.base.CombustionChamber.stoichiometry`

    **Optional Equations**

    - :py:meth:`AURORA.components.combustion.base.CombustionChamber.lambda_func`
    - :py:meth:`AURORA.components.combustion.base.CombustionChamber.ti_func`
    - :py:meth:`AURORA.components.combustion.diabatic.DiabaticCombustionChamber.energy_balance_func`
    - :py:meth:`AURORA.components.combustion.diabatic.DiabaticCombustionChamber.pr_func`

    Available fuels

    - methane, ethane, propane, butane, hydrogen

    Inlets/Outlets

    - in1, in2
    - out1

    .. note::

        The fuel and the air components can be connected to either of the
        inlets. The pressure of inlet 2 is disconnected from the pressure of
        inlet 1. A warning is prompted, if the pressure at inlet 2 is lower than
        the pressure at inlet 1.

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

    lamb : float, dict
        Actual oxygen to stoichiometric oxygen ratio, :math:`\lambda/1`.

    ti : float, dict
        Thermal input, (:math:`{LHV \cdot \dot{m}_f}`), :math:`ti/\text{W}`.

    eta : float, dict
        Combustion thermal efficiency, :math:`\eta`. Heat loss calculation based
        on share of thermal input.

    pr : float, dict
        Pressure ratio of outlet 1 to inlet 1, :math:`pr`.

    """

    @staticmethod
    def component():
        return 'diabatic combustion chamber'

    def get_parameters(self):
        return {
            'lamb': dc_cp(
                min_val=1,
                func=self.lambda_func,
                variables_columns=self.lambda_variables_columns_m,
                solve_isolated=self.lambda_solve_isolated,
                deriv=self.lambda_deriv_m,
                tensor=self.lambda_tensor,
                latex=self.lambda_func_doc,
                num_eq=1,
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
                scale=ps['fluid']['scale'],
                var_scale=ps['fluid']['scale']
            ),
            'ti': dc_cp(
                min_val=0,
                func=self.ti_func,
                variables_columns=self.ti_variables_columns_m,
                solve_isolated=self.ti_solve_isolated,
                deriv=self.ti_deriv_m,
                tensor=self.ti_tensor,
                latex=self.ti_func_doc,
                num_eq=1,
                property_data=cpd['Q'],
                SI_unit=cpd['Q']['SI_unit'],
                scale=ps['fluid']['scale'] * ps['m']['scale'] * ps['h']['scale'],
                var_scale=ps['fluid']['scale'] * ps['m']['scale'] * ps['h']['scale']
            ),
            'T_out': dc_cp(
                min_val=0,
                func=self.T_out_func,
                variables_columns=self.T_out_variables_columns,
                solve_isolated=self.T_out_solve_isolated,
                deriv=self.T_out_deriv,
                tensor=self.T_out_tensor,
                latex=self.T_out_func_doc,
                num_eq=1,
                property_data=fpd['T'],
                SI_unit=fpd['T']['SI_unit'],
                scale=ps['DT']['scale'],
                var_scale=ps['T']['scale']
            ),
            'pr': dc_cp(
                min_val=0,
                func=self.pr_func,
                variables_columns=self.pr_variables_columns,
                solve_isolated=self.pr_solve_isolated,
                deriv=self.pr_deriv,
                tensor=self.pr_tensor,
                latex=self.pr_func_doc,
                num_eq=1,
                property_data=cpd['ratio'],
                SI_unit=cpd['ratio']['SI_unit'],
                scale=ps['p']['scale'],
                var_scale=ps['pr']['scale']
            ),
            'dp': dc_cp(
                min_val=0,
                deriv=self.dp_deriv,
                variables_columns=self.dp_variables_columns,
                solve_isolated=self.dp_solve_isolated,
                func=self.dp_func,
                tensor=self.dp_tensor,
                num_eq=1,
                func_params={"inconn": 0, "outconn": 0, "dp": "dp"},
                property_data=cpd['dp'],
                SI_unit=cpd['dp']['SI_unit'],
                scale=ps['p']['scale'],
                var_scale=ps['p']['scale']
            ),
            'eta': dc_cp(
                max_val=1,
                min_val=0,
                func=self.energy_balance_func,
                variables_columns=self.energy_balance_variables_columns,
                solve_isolated=self.energy_balance_solve_isolated,
                deriv=self.energy_balance_deriv,
                tensor=self.energy_balance_tensor,
                latex=self.energy_balance_func_doc,
                num_eq=1,
                property_data=cpd['eta'],
                SI_unit=cpd['eta']['SI_unit'],
                scale=ps['eff']['scale'] * ps['m']['scale'] * ps['h']['scale'],
                var_scale=ps['eff']['scale']
            ),
            'Q_loss': dc_cp(
                max_val=0,
                is_result=True,
                property_data=cpd['Q'],
                SI_unit=cpd['Q']['SI_unit'],
                scale=ps['eff']['scale'] * ps['m']['scale'] * ps['h']['scale']
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
        }

    def get_mandatory_constraints(self):  # pressure and energy not balance
        return {
            k: v for k, v in super().get_mandatory_constraints().items()
            if k in ["mass_flow_constraints"]  # , "stoichiometry_constraints"
        }

    def simplify_pressure_enthalpy_mass_topology(self, inconni):
        if self.simplify_pressure_enthalpy_mass_topology_check():
            outconn = self.outl[0]
            outconn.target.simplify_pressure_enthalpy_mass_topology(outconn)

    def looking_forward_pressure_values(self, inconni):
        if inconni not in self.network.connections_looking_pressure_container:
            self.network.connections_looking_pressure_container.append(inconni)
            outconn = self.outl[0]
            inconn1 = self.inl[0]
            p_ef = self.get_p_ef_obj(0)
            # start spreading pressure values
            if inconn1.p.is_set and not outconn.p.is_set and p_ef.is_set:
                self.spread_forward_pressure_values(inconn1)
                inconn1.spread_pressure_reference_check()
            elif not inconn1.p.is_set and outconn.p.is_set and p_ef.is_set:
                self.spread_backward_pressure_values(outconn)
                outconn.spread_pressure_reference_check()
            self.inl[1].spread_pressure_reference_check()
            # looking for continue
            outconn.target.looking_forward_pressure_values(outconn)
            for inconn in self.inl:
                if inconn != inconni:
                    inconn.source.looking_backward_pressure_values(inconn)
        return

    def looking_backward_pressure_values(self, outconn):
        if outconn not in self.network.connections_looking_pressure_container:
            self.network.connections_looking_pressure_container.append(outconn)
            inconn1 = self.inl[0]
            p_ef = self.get_p_ef_obj(0)
            # start spreading pressure values
            if inconn1.p.is_set and not outconn.p.is_set and p_ef.is_set:
                self.spread_forward_pressure_values(inconn1)
                inconn1.spread_pressure_reference_check()
            elif not inconn1.p.is_set and outconn.p.is_set and p_ef.is_set:
                self.spread_backward_pressure_values(outconn)
                outconn.spread_pressure_reference_check()
            self.inl[1].spread_pressure_reference_check()
            # looking for continue
            for inconn in self.inl:
                inconn.source.looking_backward_pressure_values(inconn)
        return

    def spread_forward_pressure_values(self, inconni):
        outconn = self.outl[0]
        inconn_idx = self.inl.index(inconni)
        if inconn_idx == 1:
            return
        else:
            if inconni.p.is_set and not outconn.p.is_set and (self.pr.is_set or self.dp.is_set):
                if self.pr.is_set:
                    outconn.p.val_SI = inconni.p.val_SI * self.pr.val_SI
                    outconn.p.is_set = True
                    outconn.p.is_var = False
                    self.pr.is_set = False
                elif self.dp.is_set:
                    outconn.p.val_SI = inconni.p.val_SI - self.dp.val_SI
                    outconn.p.is_set = True
                    outconn.p.is_var = False
                    self.dp.is_set = False
                if outconn not in self.network.connections_spread_pressure_container:
                    self.network.connections_spread_pressure_container.append(outconn)
                    outconn.target.spread_forward_pressure_values(outconn)
                    outconn.spread_pressure_reference_check()
            return

    def spread_backward_pressure_values(self, outconn):
        inconn = self.inl[0]
        if not inconn.p.is_set and outconn.p.is_set and (self.pr.is_set or self.dp.is_set):
            if self.pr.is_set:
                inconn.p.val_SI = outconn.p.val_SI / self.pr.val_SI
                inconn.p.is_set = True
                inconn.p.is_var = False
                self.pr.is_set = False
            elif self.dp.is_set:
                inconn.p.val_SI = outconn.p.val_SI + self.dp.val_SI
                inconn.p.is_set = True
                inconn.p.is_var = False
                self.dp.is_set = False
            if inconn not in self.network.connections_spread_pressure_container:
                self.network.connections_spread_pressure_container.append(inconn)
                inconn.source.spread_backward_pressure_values(inconn)
                inconn.spread_pressure_reference_check()
        return

    def looking_for_pressure_set_boundary(self, inconni):
        outconn = self.outl[0]
        if outconn.p.is_set:
            outconn.target.spread_forward_pressure_initial(outconn)
            self.spread_backward_pressure_initial(outconn)
            return
        for inconn in self.inl:
            if inconn.p.is_set:
                self.spread_forward_pressure_initial(inconn)
                inconn.source.spread_backward_pressure_initial(inconn)
            return
        if inconni not in self.network.connections_pressure_boundary_container:
            self.network.connections_pressure_boundary_container.append(inconni)
            outconn.target.looking_for_pressure_set_boundary(outconn)
        return

    def spread_forward_pressure_initial(self, inconni):
        outconn = self.outl[0]
        inconn_idx = self.inl.index(inconni)
        if inconn_idx == 1:
            return
        else:
            if outconn not in self.network.connections_pressure_initial_container:
                self.network.connections_pressure_initial_container.append(outconn)
                if inconni.p.val_SI and not outconn.p.val_SI:
                    outconn.p.val_SI = inconni.p.val_SI * self.set_pressure_initial_factor()
                outconn.target.spread_forward_pressure_initial(outconn)
            return

    def spread_backward_pressure_initial(self, outconn):
        inconn = self.inl[0]
        if inconn not in self.network.connections_pressure_initial_container:
            self.network.connections_pressure_initial_container.append(inconn)
            if outconn.p.val_SI and not inconn.p.val_SI:
                inconn.p.val_SI = outconn.p.val_SI / self.set_pressure_initial_factor()
            inconn.source.spread_backward_pressure_initial(inconn)
        return

    def set_pressure_initial_factor(self, branch_index=0):
        inconn = self.inl[0]
        outconn = self.outl[0]
        if self.pr.is_set:
            return self.pr.val_SI
        elif self.dp.is_set:
            if inconn.p.is_set:
                return (inconn.p.val_SI - self.dp.val_SI) / inconn.p.val_SI
            elif outconn.p.is_set:
                return outconn.p.val_SI / (outconn.p.val_SI + self.dp.val_SI)
        return 0.98

    def manage_fluid_equations(self):
        self.num_fluid_eqs = 0
        self.fluid_eqs = set(
            [
                f for c in self.inl + self.outl
                for f in c.fluid.val
            ]
        )
        self.fluid_eqs_list = list(self.fluid_eqs)
        fluid_equations = {
            'stoichiometry_constraints': dc_cons(
                func=self.stoichiometry_func,
                variables_columns=self.stoichiometry_variables_columns,
                deriv=self.stoichiometry_deriv,
                tensor=self.stoichiometry_tensor,
                constant_deriv=False,
                latex=self.stoichiometry_func_doc,
                num_eq=len(self.fluid_eqs),
                fluid_composition_list=self.fluid_eqs_list,
                scale=ps['fluid']['scale'] * ps['m']['scale']),
        }
        for key, equation in fluid_equations.items():
            if equation.num_eq > 0:
                equation.label = f"fluid composition equation: {key} of {self.__class__.__name__}: {self.label}"
                self.network.fluid_equations_module_container.append(equation)
                self.num_fluid_eqs += equation.num_eq

    def T_out_func(self):
        return self.outl[0].calc_T() - self.T_out.val_SI

    def T_out_func_doc(self, label):
        pass

    def T_out_variables_columns(self):
        o = self.outl[0]
        variables_columns1 = [data.J_col for data in [o.h] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def T_out_effect(self):
        pass

    def T_out_solve_isolated(self):
        o = self.outl[0]
        if o.fluid.is_var:
            return False
        elif o.p.is_set and not o.h.is_set:
            o.h.val_SI = h_mix_pT(o.p.val_SI, self.T_out.val_SI, o.fluid_data, o.mixing_rule)
            o.h.is_set = True
            o.h.is_var = False
            self.T_out.is_set = False
            return True
        elif o.h.is_set and not o.p.is_set:
            o.p.val_SI = p_mix_hT(o.h.val_SI, self.T_out.val_SI, o.fluid_data, o.mixing_rule)
            o.p.is_set = True
            o.p.is_var = False
            self.T_out.is_set = False
            return True
        elif o.p.is_set and o.h.is_set:
            self.T_out.is_set = False
            return True
        else:
            return False

    def T_out_deriv(self, increment_filter, k):
        o = self.outl[0]
        if self.is_variable(o.h):
            self.network.jacobian[k, o.h.J_col] = dT_mix_pdh(o.p.val_SI, o.h.val_SI, o.fluid_data, o.mixing_rule, T0=self.T_out.val_SI)

    def T_out_repair_matrix(self, property_):
        msg = f"variable: {property_.label} has not a valid repair logic in T_out_repair_matrix of {self.__class__.__name__}: {self.label}"
        raise ValueError(msg)

    def T_out_tensor(self, increment_filter, k):
        pass

    def pr_func(self):
        r"""
        Equation for pressure drop.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                0 = p_\mathrm{in,1} \cdot pr - p_\mathrm{out,1}
        """
        return self.inl[0].p.val_SI * self.pr.val_SI - self.outl[0].p.val_SI

    def pr_variables_columns(self):
        i = self.inl[0]
        o = self.outl[0]
        variables_columns1 = [data.J_col for data in [i.p, o.p] if data.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def pr_take_effect(self):
        pass

    def pr_solve_isolated(self):
        i = self.inl[0]
        o = self.outl[0]
        if not i.p.is_var and not o.p.is_var:
            self.pr.is_set = False
            return True
        elif i.p.is_var and not o.p.is_var:
            i.p.val_SI = o.p.val_SI / self.pr.val_SI
            i.p.is_set = True
            i.p.is_var = False
            self.pr.is_set = False
            return True
        elif not i.p.is_var and o.p.is_var:
            o.p.val_SI = i.p.val_SI * self.pr.val_SI
            o.p.is_set = True
            o.p.is_var = False
            self.pr.is_set = False
            return True
        else:
            return False

    def pr_func_doc(self, label):
        r"""
        Equation for inlet pressure equality.

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
            r'0 = & p_\mathrm{in,1} \cdot pr - p_\mathrm{out,1}\\' + '\n'
            r'\end{split}')
        return generate_latex_eq(self, latex, label)

    def pr_deriv(self, increment_filter, k):
        r"""
        Calculate the partial derivatives for combustion pressure ratio.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of equation in Jacobian matrix.
        """
        i = self.inl[0]
        o = self.outl[0]
        if self.is_variable(i.p):
            self.network.jacobian[k, i.p.J_col] = self.pr.val_SI
        if self.is_variable(o.p):
            self.network.jacobian[k, o.p.J_col] = -1
        if self.pr.is_var:
            self.network.jacobian[k, self.pr.J_col] = i.p.val_SI

    def pr_tensor(self, increment_filter, k):
        i = self.inl[0]
        o = self.outl[0]
        if self.is_variable(i.p) and self.pr.is_var:
            self.network.tensor[i.p.J_col, self.pr.J_col] = 1

    def energy_balance_func(self):
        r"""
        Calculate the energy balance of the diabatic combustion chamber.

        Returns
        -------
        residual : float
            Residual value of equation.

            .. math::

                \begin{split}
                0 = & \sum_i \dot{m}_{in,i} \cdot
                \left( h_{in,i} - h_{in,i,ref} \right)\\
                & -\dot{m}_{out,2}\cdot\left( h_{out,1}-h_{out,1,ref} \right)\\
                & + LHV_{fuel} \cdot\left(\sum_i\dot{m}_{in,i}\cdot
                x_{fuel,in,i}- \dot{m}_{out,1} \cdot x_{fuel} \right)
                \cdot \eta
                \end{split}\\

                \forall i \in \text{inlets}

        Note
        ----
        The temperature for the reference state is set to 25 °C, thus
        the water may be liquid. In order to make sure, the state is
        referring to the lower heating value, the state of the water in the
        flue gas is fored to gaseous.

        - Reference temperature: 298.15 K.
        - Reference pressure: 1 bar.
        """
        T_ref = 298.15
        p_ref = 1e5

        res = 0
        for i in self.inl:
            i.build_fluid_data()
            res += i.m.val_SI * (
                i.h.val_SI
                - h_mix_pT(p_ref, T_ref, i.fluid_data, mixing_rule="forced-gas")
            )

        for o in self.outl:
            o.build_fluid_data()
            res -= o.m.val_SI * (
                o.h.val_SI
                - h_mix_pT(p_ref, T_ref, o.fluid_data, mixing_rule="forced-gas")
            )

        res += self.calc_ti() * self.eta.val_SI
        return res

    def energy_balance_func_doc(self, label):
        r"""
        Calculate the energy balance of the diabatic combustion chamber.

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
            r'0 = & \sum_i \dot{m}_{\mathrm{in,}i} \cdot\left( '
            r'h_{\mathrm{in,}i} - h_{\mathrm{in,}i\mathrm{,ref}} \right) -'
            r'\dot{m}_\mathrm{out,1}\cdot\left( h_\mathrm{out,1}'
            r' - h_\mathrm{out,1,ref}\right)\\' + '\n'
            r'& + LHV_{fuel} \cdot \left(\sum_i \dot{m}_{\mathrm{in,}i} '
            r'\cdot x_{fuel\mathrm{,in,}i} - \dot{m}_\mathrm{out,1} '
            r'\cdot x_{fuel\mathrm{,out,1}} \right) \cdot \eta\\' + '\n'
            r'& \forall i \in \text{inlets}\\'
            r'& T_\mathrm{ref}=\unit[298.15]{K}'
            r'\;p_\mathrm{ref}=\unit[10^5]{Pa}\\'
            '\n' + r'\end{split}'
        )
        return generate_latex_eq(self, latex, label)

    def calc_parameters(self):
        r"""Postprocessing parameter calculation."""
        super().calc_parameters()

        T_ref = 298.15
        p_ref = 1e5

        res = 0
        for i in self.inl:
            i.build_fluid_data()
            res += i.m.val_SI * (
                i.h.val_SI
                - h_mix_pT(p_ref, T_ref, i.fluid_data, mixing_rule="forced-gas")
            )

        for o in self.outl:
            o.build_fluid_data()
            res -= o.m.val_SI * (
                o.h.val_SI
                - h_mix_pT(p_ref, T_ref, o.fluid_data, mixing_rule="forced-gas")
            )

        self.eta.val_SI = -res / self.ti.val_SI
        self.Q_loss.val_SI = -(1 - self.eta.val_SI) * self.ti.val_SI
        self.T_out.val_SI = self.outl[0].calc_T()
        self.pr.val_SI = self.outl[0].p.val_SI / self.inl[0].p.val_SI
        self.dp.val_SI = self.inl[0].p.val_SI - self.outl[0].p.val_SI
        for num, i in enumerate(self.inl):
            if i.p.val_SI < self.outl[0].p.val_SI:
                msg = (
                    f"The pressure at inlet {num + 1} is lower than the "
                    f"pressure at the outlet of component {self.label}."
                )
                logger.warning(msg)

    def exergy_balance(self, T0):

        self.E_P = self.outl[0].Ex_physical - (
            self.inl[0].Ex_physical + self.inl[1].Ex_physical
        )
        self.E_F = (
            self.inl[0].Ex_chemical + self.inl[1].Ex_chemical -
            self.outl[0].Ex_chemical
        )

        self.E_D = self.E_F - self.E_P
        self.epsilon = self._calc_epsilon()
        self.E_bus = {"chemical": np.nan, "physical": np.nan, "massless": np.nan}
