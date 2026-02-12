# -*- coding: utf-8

"""Module of class DeaeratorSimple.
"""

from Aurora.components.component import component_registry
from Aurora.tools import logger
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.document_models import generate_latex_eq
from Aurora.tools.fluid_properties import d2h_mix_d2p_Q
from Aurora.tools.fluid_properties import h_mix_pQ
from Aurora.tools.fluid_properties import p_mix_hQ
from Aurora.components.fluid_components.distributors.merge import Merge


@component_registry
class DeaeratorSimple(Merge):
    r"""
    A simple Deaerator extracts oxygen from vapour.
    The saturated liquid is obtained on the outlet of the deaerator.
    **Mandatory Equations**

    - :py:meth:`Aurora.components.component.Component.fluid_func`
    - :py:meth:`Aurora.components.component.Component.mass_flow_func`
    - :py:meth:`Aurora.components.heat_exchangers.deaerator.Deaerator.energy_balance_func`
    - :py:meth:`Aurora.components.heat_exchangers.deaerator.Deaerator.saturated_liquid_func
    - condensate outlet state, function can be disabled by specifying
      :code:`set_attr(subcooling=True)`
      :py:meth:`Aurora.components.heat_exchangers.deaerator.Deaerator.subcooling_func`

    Inlets/Outlets

    - in1, in2 (index 1: cooled liquid, index 2: extract vapour)
    - out1 (index 1: saturated liquid)

    """

    @staticmethod
    def component():
        return 'deaerator simple'

    @staticmethod
    def initialise_target(c, key):
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
                10^5 & \text{key = 'p'}\\
                5 \cdot 10^5 & \text{key = 'h'}
                \end{cases}
        """
        if key == 'p':
            return 5e5
        elif key == 'h':
            if c.target_id == 'in1':
                return h_mix_pQ(c.p.val_SI, 0, c.fluid_data) * 0.95
            else:
                return h_mix_pQ(c.p.val_SI, 1, c.fluid_data) * 1.05

    @staticmethod
    def initialise_source(c, key):
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
                10^5 & \text{key = 'p'}\\
                h\left(p_{out,1}, x=0 \right) & \text{key = 'h'}
                \end{cases}
        """
        if key == 'p':
            return 2e5
        elif key == 'h':
            return h_mix_pQ(c.p.val_SI, 0, c.fluid_data)

    @staticmethod
    def get_parameters():
        return {'num_in': dc_simple(val=2, is_set=True)}

    def manage_fluid_equations(self):
        self.num_fluid_eqs = 0
        variable_fluids = set(
            [fluid for c in self.inl + self.outl for fluid in c.fluid.is_var]
        )
        num_fluid_eq = len(variable_fluids)
        fluid_equations = {
            'fluid_constraints': dc_cons(
                func=self.fluid_func,
                variables_columns=self.fluid_variables_columns,
                deriv=self.fluid_deriv,
                tensor=self.fluid_tensor,
                constant_deriv=False,
                latex=self.fluid_func_doc,
                num_eq=num_fluid_eq,
                fluid_composition_list=list(variable_fluids),
                scale=ps['fluid']['scale'] * ps['m']['scale']),
        }
        for key, equation in fluid_equations.items():
            if equation.num_eq > 0:
                equation.label = f"fluid composition equation: {key} of {self.__class__.__name__}: {self.label}"
                self.network.fluid_equations_module_container.append(equation)
                self.num_fluid_eqs += equation.num_eq

    def get_mandatory_constraints(self):
        return {
            'mass_flow_constraints': dc_cons(
                func=self.mass_flow_func,
                variables_columns=self.mass_flow_variables_columns,
                solve_isolated=self.mass_flow_solve_isolated,
                deriv=self.mass_flow_deriv,
                tensor=self.mass_flow_tensor,
                constant_deriv=True,
                latex=self.mass_flow_func_doc,
                num_eq=1,  # num_m_eq
                scale=ps['m']['scale']),
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
            'saturated_liquid_constraints': dc_cons(
                func=self.saturated_liquid_func,
                variables_columns=self.saturated_liquid_variables_columns,
                solve_isolated=self.saturated_liquid_solve_isolated,
                deriv=self.saturated_liquid_deriv,
                tensor=self.saturated_liquid_tensor,
                constant_deriv=False,
                latex=self.saturated_liquid_func_doc,
                num_eq=1,
                scale=ps['h']['scale']),
        }

    def saturated_liquid_func(self):
        r"""
        Calculate outlet liquid state.

        Returns
        -------
        residual : float
            Residual value of equation

            .. math::

                0 = h_{out,1} - h\left(p_{out,1}, x=0 \right)
        """
        o = self.outl[0]
        return o.h.val_SI - h_mix_pQ(o.p.val_SI, 0, o.fluid_data)

    def saturated_liquid_variables_columns(self):
        o = self.outl[0]
        variables_columns1 = [data.J_col for data in [o.h] if data.is_var]  # [o.p, o.h]
        variables_columns1.sort()
        return [variables_columns1]

    def saturated_liquid_take_effect(self):
        pass

    def saturated_liquid_solve_isolated(self):
        o = self.outl[0]
        if not o.p.is_var and not o.h.is_var:
            return True
        elif not o.p.is_var and o.h.is_var:
            o.h.val_SI = h_mix_pQ(o.p.val_SI, 0, o.fluid_data)
            o.h.is_set = True
            o.h.is_var = False
            return True
        elif o.p.is_var and not o.h.is_var:
            o.p.val_SI = p_mix_hQ(o.h.val_SI, 0, o.fluid_data)
            o.p.is_set = True
            o.p.is_var = False
            return True
        return False

    def saturated_liquid_func_doc(self, label):
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
        latex = r'0=h_\mathrm{out,1}-h\left(p_\mathrm{out,1}, x=0 \right)'
        return generate_latex_eq(self, latex, label)

    def saturated_liquid_deriv(self, increment_filter, k):
        r"""
        Partial derivatives of saturated liquid at mere outlet function.

        Parameters
        ----------
        increment_filter : ndarray
            Matrix for filtering non-changing variables.

        k : int
            Position of derivatives in Jacobian matrix (k-th equation).
        """
        o = self.outl[0]
        # if self.is_variable(o.p, increment_filter):
        #     self.network.jacobian[k, o.p.J_col] = -dh_mix_dpQ(o.p.val_SI, 0, o.fluid_data)
        if self.is_variable(o.h):
            self.network.jacobian[k, o.h.J_col] = 1

    def saturated_liquid_tensor(self, increment_filter, k):
        o = self.outl[0]
        if self.is_variable(o.p, increment_filter):
            self.network.tensor[o.p.J_col, o.p.J_col, k] = -d2h_mix_d2p_Q(o.p.val_SI, 0, o.fluid_data)

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

    def bounds_p_generate(self):
        o0 = self.outl[0]
        o0.p.max_val = o0.calc_p_critical()

