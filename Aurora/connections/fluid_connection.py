# -*- coding: utf-8
"""Module of class FluidConnection.
"""

import numpy as np

from Aurora.components.component import Component
from Aurora.connections.connection import Connection
from Aurora.tools import fluid_properties as fp
from Aurora.tools import logger
from Aurora.tools.data_containers import DataContainer as dc
from Aurora.tools.data_containers import FluidComposition as dc_flu
from Aurora.tools.data_containers import FluidProperties as dc_prop
from Aurora.tools.data_containers import ReferencedFluidProperties as dc_ref
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
from Aurora.tools.fluid_properties import CoolPropWrapper
from Aurora.tools.fluid_properties import Q_mix_ph

from Aurora.tools.fluid_properties import T_sat_p
from Aurora.tools.fluid_properties import dh_mix_dpQ
from Aurora.tools.fluid_properties import d2h_mix_d2p_Q

from Aurora.tools.fluid_properties import T_mix_ph
from Aurora.tools.fluid_properties import dT_mix_dph
from Aurora.tools.fluid_properties import dT_mix_pdh
from Aurora.tools.fluid_properties import dT_mix_ph_dfluid
from Aurora.tools.fluid_properties import d2T_mix_d2p_h
from Aurora.tools.fluid_properties import d2T_mix_p_d2h
from Aurora.tools.fluid_properties import d2T_mix_ph_d2fluid
from Aurora.tools.fluid_properties import d2T_mix_dpdh
from Aurora.tools.fluid_properties import d2T_mix_dp_h_dfluid
from Aurora.tools.fluid_properties import d2T_mix_p_dh_dfluid
from Aurora.tools.fluid_properties import d2T_mix_ph_dfluid1_dfluid2

from Aurora.tools.fluid_properties import dT_sat_dp
from Aurora.tools.fluid_properties import d2T_sat_d2p

from Aurora.tools.fluid_properties import v_mix_ph
from Aurora.tools.fluid_properties import dv_mix_dph
from Aurora.tools.fluid_properties import dv_mix_pdh
from Aurora.tools.fluid_properties import dv_mix_ph_dfluid
from Aurora.tools.fluid_properties import d2v_mix_d2p_h
from Aurora.tools.fluid_properties import d2v_mix_p_d2h
from Aurora.tools.fluid_properties import d2v_mix_ph_d2fluid
from Aurora.tools.fluid_properties import d2v_mix_dp_dh
from Aurora.tools.fluid_properties import d2v_mix_dp_h_dfluid
from Aurora.tools.fluid_properties import d2v_mix_p_dh_dfluid
from Aurora.tools.fluid_properties import d2v_mix_ph_dfluid1_dfluid2

from Aurora.tools.fluid_properties import h_mix_pQ
from Aurora.tools.fluid_properties import h_mix_pT
from Aurora.tools.fluid_properties import h_mix_pv

from Aurora.tools.fluid_properties import p_mix_hT
from Aurora.tools.fluid_properties import p_mix_hv
from Aurora.tools.fluid_properties import p_mix_hQ

from Aurora.tools.fluid_properties import s_mix_ph
from Aurora.tools.fluid_properties import phase_mix_ph
from Aurora.tools.fluid_properties import p_critical_fluids

from Aurora.tools.fluid_properties import viscosity_mix_ph

from Aurora.tools.fluid_properties.functions import p_sat_T
from Aurora.tools.fluid_properties.helpers import get_mixture_temperature_range
from Aurora.tools.fluid_properties.helpers import get_number_of_fluids
from Aurora.tools.global_vars import ERR
from Aurora.tools.global_vars import min_derive
from Aurora.tools.global_vars import fluid_property_data as fpd
from Aurora.tools.global_vars import electromagnetic_property_data as epd
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.helpers import AURORAConnectionError
from Aurora.tools.helpers import AURORANetworkError
from Aurora.tools.helpers import convert_from_SI


class FluidConnection(Connection):
    r"""
    Class connection is the container for fluid properties between components.

    Parameters
    ----------
    m : float, Aurora.connections.connection.Ref
        Mass flow specification.

    m0 : float
        Starting value specification for mass flow.

    p : float, Aurora.connections.connection.Ref
        Pressure specification.

    p0 : float
        Starting value specification for pressure.

    h : float, Aurora.connections.connection.Ref
        Enthalpy specification.

    h0 : float
        Starting value specification for enthalpy.

    fluid : dict
        Fluid compostition specification.

    fluid0 : dict
        Starting value specification for fluid compostition.

    fluid_balance : boolean
        Fluid balance equation specification.

    x : float
        Gas phase mass fraction specification.

    T : float, Aurora.connections.connection.Ref
        Temperature specification.

    Td_bp : float
        Temperature difference to boiling point at pressure corresponding
        pressure of this connection in K.

    v : float
        Volumetric flow specification.

    'state' : str
        State of the pure fluid on this connection: liquid ('l') or gaseous
        ('g').

    design : list
        List containing design parameters (stated as string).

    offdesign : list
        List containing offdesign parameters (stated as string).

    design_path : str
        Path to individual design case for this connection.

    local_offdesign : boolean
        Treat this connection in offdesign mode in a design calculation.

    local_design : boolean
        Treat this connection in design mode in an offdesign calculation.

    printout : boolean
        Include this connection in the network's results printout.

    label : str
        Label of the connection. The default value is:
        :code:`'source:source_id_target:target_id'`.

    Note
    ----
    - The fluid balance parameter applies a balancing of the fluid vector on
      the specified conntion to 100 %. For example, you have four fluid
      components (a, b, c and d) in your vector, you set two of them
      (a and b) and want the other two (components c and d) to be a result of
      your calculation. If you set this parameter to True, the equation
      (0 = 1 - a - b - c - d) will be applied.

    - The specification of values for design and/or offdesign is used for
      automatic switch from design to offdesign calculation: All parameters
      given in 'design', e.g. :code:`design=['T', 'p']`, are unset in any
      offdesign calculation, parameters given in 'offdesign' are set for
      offdesign calculation.

    - The property state is applied on pure fluids only. If you specify the
      desired state of the fluid at a connection the convergence check will
      adjust the enthalpy values of that connection for the first
      iterations in order to meet the state requirement.

    """

    def __init__(self, source, outlet_id, target, inlet_id, label=None, **kwargs):
        super().__init__(source, outlet_id, target, inlet_id, label, **kwargs)
        self.state = dc_simple()  # delicate the fluid state, vapour or liquid
        self.phase = dc_simple()
        self.mixing_rule = None  #

    def connection_type(self):
        return 'fluid'

    def get_parameters(self):
        return {
            "m": dc_prop(
                is_var=True,
                property_data=fpd['m'],
                SI_unit=fpd['m']['SI_unit'],
                scale=ps['m']['scale']),
            "p": dc_prop(
                is_var=True,
                property_data=fpd['p'],
                SI_unit=fpd['p']['SI_unit'],
                scale=ps['p']['scale']),
            "h": dc_prop(
                is_var=True,
                property_data=fpd['h'],
                SI_unit=fpd['h']['SI_unit'],
                scale=ps['h']['scale']),
            "vol": dc_prop(
                property_data=fpd['vol'],
                SI_unit=fpd['vol']['SI_unit'],
                scale=ps['vol']['scale']),  # specific volume
            "s": dc_prop(
                property_data=fpd['s'],
                SI_unit=fpd['s']['SI_unit'],
                scale=ps['s']['scale']),
            "fluid": dc_flu(scale=ps['fluid']['scale']),
            "T": dc_prop(
                func=self.T_func,
                variables_columns=self.T_variables_columns,
                solve_isolated=self.T_solve_isolated,
                deriv=self.T_deriv,
                repair_matrix=self.T_repair_matrix,
                tensor=self.T_tensor,
                num_eq=1,
                val_SI=300,
                property_data=fpd['T'],
                SI_unit=fpd['T']['SI_unit'],
                scale=ps['T']['scale']),
            "v": dc_prop(
                func=self.v_func,
                variables_columns=self.v_variables_columns,
                solve_isolated=self.v_solve_isolated,
                deriv=self.v_deriv,
                tensor=self.v_tensor,
                num_eq=1,
                property_data=fpd['v'],
                SI_unit=fpd['v']['SI_unit'],
                scale=ps['v']['scale']),  # volumetric flow
            "x": dc_prop(
                func=self.x_func,
                variables_columns=self.x_variables_columns,
                solve_isolated=self.x_solve_isolated,
                deriv=self.x_deriv,
                tensor=self.x_tensor,
                num_eq=1,
                property_data=fpd['x'],
                SI_unit=fpd['x']['SI_unit'],
                scale=ps['h']['scale']),
            "Td_bp": dc_prop(
                func=self.Td_bp_func,
                variables_columns=self.Td_bp_variables_columns,
                solve_isolated=self.Td_bp_solve_isolated,
                deriv=self.Td_bp_deriv,
                repair_matrix=self.Td_bp_repair_matrix,
                tensor=self.Td_bp_tensor,
                min_val=0,
                num_eq=1,
                property_data=fpd['Td_bp'],
                SI_unit=fpd['Td_bp']['SI_unit'],
                scale=ps['DT']['scale']),  # T difference to boiling point
            'Td_dew': dc_prop(
                func=self.Td_dew_func,
                variables_columns=self.Td_dew_variables_columns,
                solve_isolated=self.Td_dew_solve_isolated,
                deriv=self.Td_dew_deriv,
                repair_matrix=self.Td_dew_repair_matrix,
                tensor=self.Td_dew_tensor,
                min_val=0,
                num_eq=1,
                property_data=fpd['Td_dew'],
                SI_unit=fpd['Td_dew']['SI_unit'],
                scale=ps['DT']['scale']
            ),
            "m_ref": dc_ref(
                func=self.primary_ref_func,
                variables_columns=self.primary_ref_variables_columns,
                solve_isolated=self.primary_ref_solve_isolated,
                deriv=self.primary_ref_deriv,
                tensor=self.primary_ref_tensor,
                num_eq=1,
                func_params={"variable": "m"},
                SI_unit=fpd['m']['SI_unit'],
                scale=ps['m']['scale']
            ),
            "p_ref": dc_ref(
                func=self.primary_ref_func,
                variables_columns=self.primary_ref_variables_columns,
                solve_isolated=self.primary_ref_solve_isolated,
                deriv=self.primary_ref_deriv,
                tensor=self.primary_ref_tensor,
                num_eq=1,
                func_params={"variable": "p"},
                SI_unit=fpd['p']['SI_unit'],
                scale=ps['p']['scale']
            ),
            "h_ref": dc_ref(
                func=self.primary_ref_func,
                variables_columns=self.primary_ref_variables_columns,
                solve_isolated=self.primary_ref_solve_isolated,
                deriv=self.primary_ref_deriv,
                tensor=self.primary_ref_tensor,
                num_eq=1,
                func_params={"variable": "h"},
                SI_unit=fpd['h']['SI_unit'],
                scale=ps['h']['scale']
            ),
            "T_ref": dc_ref(
                func=self.T_ref_func,
                variables_columns=self.T_ref_variables_columns,
                solve_isolated=self.T_ref_solve_isolated,
                deriv=self.T_ref_deriv,
                repair_matrix=self.T_ref_repair_matrix,
                tensor=self.T_ref_tensor,
                num_eq=1,
                SI_unit=fpd['T']['SI_unit'],
                scale=ps['T']['scale']
            ),
            "v_ref": dc_ref(
                func=self.v_ref_func,
                variables_columns=self.v_ref_variables_columns,
                solve_isolated=self.v_ref_solve_isolated,
                deriv=self.v_ref_deriv,
                tensor=self.v_ref_tensor,
                num_eq=1,
                SI_unit=fpd['v']['SI_unit'],
                scale=ps['v']['scale']
            ),
        }

    def set_attr(self, **kwargs):
        r"""
        Set, reset or unset attributes of a connection.

        Parameters
        ----------
        m : float, Aurora.connections.connection.Ref
            Mass flow specification.

        m0 : float
            Starting value specification for mass flow.

        p : float, Aurora.connections.connection.Ref
            Pressure specification.

        p0 : float
            Starting value specification for pressure.

        h : float, Aurora.connections.connection.Ref
            Enthalpy specification.

        h0 : float
            Starting value specification for enthalpy.

        fluid : dict
            Fluid composition specification.

        fluid0 : dict
            Starting value specification for fluid composition.

        fluid_balance : boolean
            Fluid balance equation specification.

        x : float
            Gas phase mass fraction specification.

        T : float, Aurora.connections.connection.Ref
            Temperature specification.

        Td_bp : float
            Temperature difference to boiling point at pressure corresponding
            pressure of this connection in K.

        v : float
            Volumetric flow specification.

        state : str
            State of the pure fluid on this connection: liquid ('l') or gaseous
            ('g').

        design : list
            List containing design parameters (stated as string).

        offdesign : list
            List containing offdesign parameters (stated as string).

        design_path : str
            Path to individual design case for this connection.

        local_offdesign : boolean
            Treat this connection in offdesign mode in a design calculation.

        local_design : boolean
            Treat this connection in design mode in an offdesign calculation.

        printout : boolean
            Include this connection in the network's results printout.

        Note
        ----
        - The fluid balance parameter applies a balancing of the fluid vector
          on the specified connection to 100 %. For example, you have four
          fluid components (a, b, c and d) in your vector, you set two of them
          (a and b) and want the other two (components c and d) to be a result
          of your calculation. If you set this parameter to True, the equation
          (0 = 1 - a - b - c - d) will be applied.
        - The specification of values for design and/or offdesign is used for
          automatic switch from design to offdesign calculation: All parameters
          given in 'design', e.g. :code:`design=['T', 'p']`, are unset in any
          offdesign calculation, parameters given in 'offdesign' are set for
          offdesign calculation.
        - The property state is applied on pure fluids only. If you specify the
          desired state of the fluid at a connection the convergence check will
          adjust the enthalpy values of that connection for the first
          iterations in order to meet the state requirement.
        """
        # set specified values
        for key in kwargs:
            if 'fluid' in key:
                self._fluid_specification(key, kwargs[key])
            elif key == 'state':
                if kwargs[key] in ['l', 'g']:
                    self.state.set_attr(val=kwargs[key], is_set=True)
                elif kwargs[key] is None:
                    self.state.set_attr(is_set=False)
                else:
                    msg = (
                        'Keyword argument "state" must either be '
                        '"l" or "g" or be None.'
                    )
                    logger.error(msg)
                    raise TypeError(msg)
            elif key == "mixing_rule":  # no checking
                self.mixing_rule = kwargs[key]
            # invalid keyword
            else:
                super().set_attr(**{key: kwargs[key]})

    def _fluid_specification(self, key, value):
        self._check_fluid_datatypes(key, value)  # check the value type of fluid
        if key == "fluid":
            for fluid, fraction in value.items():
                if "::" in fluid:
                    back_end, fluid = fluid.split("::")
                else:
                    back_end = None
                if fraction is None:  # unset the known fluid composition
                    if fluid in self.fluid.is_set:
                        self.fluid.is_set.remove(fluid)  # unset the known_variables_state of the fluid composition
                    self.fluid.is_var.add(fluid)  # set the fluid composition is var
                else:
                    self.fluid.val[fluid] = fraction  # generate the fluid composition dict
                    self.fluid.is_set.add(fluid)
                    if fluid in self.fluid.is_var:  # unset the fluid composition is var
                        self.fluid.is_var.remove(fluid)
                    self.fluid.back_end[fluid] = back_end  # generate the fluid back_end dict
        elif key == "fluid0":
            self.fluid.val0.update(value)  # the value_type of fluid0 is dict
        elif key == "fluid_engines":  # no checking ????
            self.fluid.engine = value
        else:
            msg = f"Connections do not have an attribute named {key}"
            logger.error(msg)
            raise KeyError(msg)

    def _check_fluid_datatypes(self, key, value):  # check the value type of variables about fluid
        if not isinstance(value, dict):
            msg = "Datatype for fluid vector specification must be dict."
            logger.error(msg)
            raise TypeError(msg)

    @staticmethod
    def _serializable():
        return [
            "source_id", "target_id",
            "design_path", "design", "offdesign", "local_design", "local_offdesign",
            "printout", "mixing_rule", "state"
        ]

    def _create_fluid_wrapper(self):  # generate fluid wrapper for all conns in initialise in network
        for fluid in self.fluid.val:
            if fluid in self.fluid.wrapper:
                continue
            if fluid not in self.fluid.engine:
                self.fluid.engine[fluid] = CoolPropWrapper

            back_end = None
            if fluid in self.fluid.back_end:
                back_end = self.fluid.back_end[fluid]
            else:
                self.fluid.back_end[fluid] = None
            # add the fluid calculating back information to fluid wrapper
            self.fluid.wrapper[fluid] = self.fluid.engine[fluid](fluid, back_end)

    def build_fluid_data(self):  # serve for fluid property calculation
        self.fluid_data = {
            fluid: {
                "wrapper": self.fluid.wrapper[fluid],
                "mass_fraction": self.fluid.val[fluid]
            } for fluid in self.fluid.val
        }

    def simplify_specifications(self):  # cited in the init_design and init_offdesign in network
        """simplify the initialization of the connection, delete redundant variables and equations.'"""
        systemvar_specs = []
        nonsystemvar_specs = []
        for name, container in self.property_data.items():
            if container.is_set:
                if name in ["m", "p", "h"]:  # iterated properties
                    systemvar_specs += [name]
                elif name in ["T", "x", "Td_bp", "v"]:
                    nonsystemvar_specs += [name]
        specs = set(systemvar_specs + nonsystemvar_specs)
        num_specs = len(specs)
        if num_specs > 3:
            inputs = ", ".join(specs)
            msg = (
                "You have specified more than 3 parameters for the connection "
                f"{self.label} with a known fluid compoistion: {inputs}. This "
                "overdetermines the state of the fluid."
            )
            raise AURORANetworkError(msg)
        if not self.h.is_set and self.p.is_set:  # !!!!
            if self.T.is_set:
                self.h.val_SI = h_mix_pT(self.p.val_SI, self.T.val_SI, self.fluid_data, self.mixing_rule)
                # self.h._solved = True
                self.h.is_set = True
                self.h.is_var = False
                self.T.is_set = False
            elif self.Td_bp.is_set:
                T_sat = T_sat_p(self.p.val_SI, self.fluid_data)
                self.h.val_SI = h_mix_pT(self.p.val_SI, T_sat + self.Td_bp.val_SI, self.fluid_data)
                # self.h._solved = True
                self.h.is_set = True
                self.h.is_var = False
                self.Td_bp.is_set = False
            elif self.x.is_set:
                self.h.val_SI = h_mix_pQ(self.p.val_SI, self.x.val_SI, self.fluid_data)
                # self.h._solved = True
                self.h.is_set = True
                self.h.is_var = False
                self.x.is_set = False
        elif not self.h.is_set and not self.p.is_set:
            if self.T.is_set and self.x.is_set:
                self.p.val_SI = p_sat_T(self.T.val_SI, self.fluid_data)  # !!!
                self.h.val_SI = h_mix_pQ(self.p.val_SI, self.x.val_SI, self.fluid_data)
                self.T.is_set = False
                self.x.is_set = False
                # self.p._solved = True
                self.p.is_set = True
                self.p.is_var = False
                # self.h._solved = True
                self.h.is_set = True
                self.h.is_var = False

    def calc_T(self, T0=None):  # calculate temperature due to p,h
        if T0 is None:
            T0 = self.T.val_SI
        return T_mix_ph(self.p.val_SI, self.h.val_SI, self.fluid_data, self.mixing_rule, T0=T0)

    def T_func(self, **kwargs):  # T.val_SI is the value specified as known property
        return self.calc_T() - self.T.val_SI

    def T_variables_columns(self, **kwargs):
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [self.h] if data.is_var]  # [self.p, self.h]
        # variables_columns1 += [self.fluid.J_col[fluid] for fluid in self.fluid.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def T_take_effect(self):
        pass

    def T_solve_isolated(self, **kwargs):
        if self.fluid.is_var:
            return False
        if self.h.is_var and not self.p.is_var:
            self.h.val_SI = h_mix_pT(self.p.val_SI, self.T.val_SI, self.fluid_data, self.mixing_rule)
            self.h.is_set = True
            self.h.is_var = False
            self.T.is_set = False
            return True
        elif not self.h.is_var and self.p.is_var:
            self.p.val_SI = p_mix_hT(self.h.val_SI, self.T.val_SI, self.fluid_data, self.mixing_rule)
            self.p.is_set = True
            self.p.is_var = False
            self.T.is_set = False
            return True
        elif self.h.is_var and self.p.is_var:
            return False
        else:
            self.T.is_set = False
            return True

    def T_repair_matrix(self, property_, **kwargs):
        if property_ == self.h:
            h0 = h_mix_pQ(self.p.val_SI, 0, self.fluid_data)
            h1 = h_mix_pQ(self.p.val_SI, 1, self.fluid_data)
            return abs(self.calc_T() - self.T.val_SI) / max(self.h.val_SI - h0, h1 - self.h.val_SI)
        else:
            msg = f"variable: {property_.label} is not a valid property in T_repair_matrix of Connection: {self.label}"
            raise ValueError(msg)

    def T_deriv(self, increment_filter, k, **kwargs):  # m,p,h are the variables iterated merely
        # if self.p.is_var:
        #     self.network.jacobian[k, self.p.J_col] = (
        #         dT_mix_dph(self.p.val_SI, self.h.val_SI, self.fluid_data, self.mixing_rule, self.T.val_SI)
        #     )
        if self.h.is_var:
            self.network.jacobian[k, self.h.J_col] = dT_mix_pdh(self.p.val_SI, self.h.val_SI, self.fluid_data, self.mixing_rule, self.T.val_SI)

    def T_tensor(self, increment_filter, k, **kwargs):
        # if self.p.is_var:
        #     self.network.tensor[self.p.J_col, self.p.J_col, k] = (
        #         d2T_mix_d2p_h(self.p.val_SI, self.h.val_SI, self.fluid_data, self.mixing_rule, self.T.val_SI))
        if self.h.is_var:
            self.network.tensor[self.h.J_col, self.h.J_col, k] = (
                d2T_mix_p_d2h(self.p.val_SI, self.h.val_SI, self.fluid_data, self.mixing_rule, self.T.val_SI))
        # if self.p.is_var and self.h.is_var:
        #     tensor_2 = d2T_mix_dpdh(self.p.val_SI, self.h.val_SI, self.fluid_data, self.mixing_rule, self.T.val_SI)
        #     self.network.tensor[self.p.J_col, self.h.J_col, k] = tensor_2
        #     self.network.tensor[self.h.J_col, self.p.J_col, k] = tensor_2

    def T_ref_func(self, **kwargs):
        ref = self.T_ref.ref
        return self.calc_T() - (ref.obj.calc_T() * ref.factor + ref.delta_SI)

    def T_ref_variables_columns(self, **kwargs):
        variables_columns = self.T_variables_columns(**kwargs)
        ref = self.T_ref.ref
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [ref.obj.h] if data.is_var]  # [ref.obj.p, ref.obj.h]
        # variables_columns1 += [ref.obj.fluid.J_col[fluid] for fluid in ref.obj.fluid.is_var]
        variables_columns[0] = sorted(list(set(variables_columns[0] + variables_columns1)))
        return variables_columns

    def T_ref_take_effect(self, **kwargs):
        pass

    def T_ref_solve_isolated(self, **kwargs):
        ref = self.T_ref.ref
        if self.fluid.is_var or ref.obj.fluid.is_var:
            return False
        if not self.p.is_var and not self.h.is_var:
            if not ref.obj.p.is_var and ref.obj.h.is_var:
                self.T.val_SI = self.calc_T()
                ref.obj.T.val_SI = (self.T.val_SI - ref.delta_SI) / ref.factor
                ref.obj.h.val_SI = h_mix_pT(ref.obj.p.val_SI, ref.obj.T.val_SI, ref.obj.fluid_data, ref.obj.mixing_rule)
                ref.obj.h.is_set = True
                ref.obj.h.is_var = False
                self.T_ref.is_set = False
                return True
            elif ref.obj.p.is_var and not ref.obj.h.is_var:
                self.T.val_SI = self.calc_T()
                ref.obj.T.val_SI = (self.T.val_SI - ref.delta_SI) / ref.factor
                ref.obj.p.val_SI = p_mix_hT(ref.obj.h.val_SI, ref.obj.T.val_SI, ref.obj.fluid_data, ref.obj.mixing_rule)
                ref.obj.p.is_set = True
                ref.obj.p.is_var = False
                self.T_ref.is_set = False
                return True
            elif ref.obj.p.is_var and ref.obj.h.is_var:
                return False
            else:
                self.T_ref.is_set = False
                return True
        elif not ref.obj.p.is_var and not ref.obj.h.is_var:
            if not self.p.is_var and self.h.is_var:
                ref.obj.T.val_SI = ref.obj.calc_T()
                self.T.val_SI = ref.obj.T.val_SI * ref.factor + ref.delta_SI
                self.h.val_SI = h_mix_pT(self.p.val_SI, self.T.val_SI, self.fluid_data, self.mixing_rule)
                self.h.is_set = True
                self.h.is_var = False
                self.T_ref.is_set = False
                return True
            elif self.p.is_var and not self.h.is_var:
                ref.obj.T.val_SI = ref.obj.calc_T()
                self.T.val_SI = ref.obj.T.val_SI * ref.factor + ref.delta_SI
                self.p.val_SI = p_mix_hT(self.h.val_SI, self.T.val_SI, self.fluid_data, self.mixing_rule)
                self.p.is_set = True
                self.p.is_var = False
                self.T_ref.is_set = False
                return True
            elif self.p.is_var and self.h.is_var:
                return False
            else:
                self.T_ref.is_set = False
                return True
        return False

    def T_ref_deriv(self, increment_filter, k, **kwargs):
        # first part of sum is identical to direct temperature specification
        self.T_deriv(increment_filter, k, **kwargs)
        ref = self.T_ref.ref
        # if ref.obj.p.is_var:
        #     self.network.jacobian[k, ref.obj.p.J_col] = -(
        #         dT_mix_dph(ref.obj.p.val_SI, ref.obj.h.val_SI, ref.obj.fluid_data, ref.obj.mixing_rule)
        #     ) * ref.factor
        if ref.obj.h.is_var:
            self.network.jacobian[k, ref.obj.h.J_col] = -(
                dT_mix_pdh(ref.obj.p.val_SI, ref.obj.h.val_SI, ref.obj.fluid_data, ref.obj.mixing_rule)
            ) * ref.factor

    def T_ref_repair_matrix(self, property_, **kwargs):
        ref = self.T_ref.ref
        if property_ == self.h:
            h0 = h_mix_pQ(self.p.val_SI, 0, self.fluid_data)
            h1 = h_mix_pQ(self.p.val_SI, 1, self.fluid_data)
            return abs(self.calc_T() - (ref.obj.calc_T() * ref.factor + ref.delta_SI)) / max(self.h.val_SI - h0, h1 - self.h.val_SI)
        elif property_ == ref.obj.h:
            h0 = h_mix_pQ(ref.obj.p.val_SI, 0, ref.obj.fluid_data)
            h1 = h_mix_pQ(ref.obj.p.val_SI, 1, ref.obj.fluid_data)
            return -ref.factor * abs(self.calc_T() - (ref.obj.calc_T() * ref.factor + ref.delta_SI)) / max(ref.obj.h.val_SI - h0, h1 - ref.obj.h.val_SI)
        else:
            msg = f"variable: {property_.label} is not a valid property in T_ref_repair_matrix of Connection: {self.label}"
            raise ValueError(msg)

    def T_ref_tensor(self, increment_filter, k, **kwargs):
        self.T_tensor(increment_filter, k, **kwargs)
        ref = self.T_ref.ref
        # if ref.obj.p.is_var:
        #     self.network.tensor[ref.obj.p.J_col, ref.obj.p.J_col, k] = (
        #         -d2T_mix_d2p_h(ref.obj.p.val_SI, ref.obj.h.val_SI, ref.obj.fluid_data, ref.obj.mixing_rule) * ref.factor)
        if ref.obj.h.is_var:
            self.network.tensor[ref.obj.h.J_col, ref.obj.h.J_col, k] = (
                -d2T_mix_p_d2h(ref.obj.p.val_SI, ref.obj.h.val_SI, ref.obj.fluid_data, ref.obj.mixing_rule) * ref.factor)
        # if ref.obj.p.is_var and ref.obj.h.is_var:
        #     tensor_2 = -d2T_mix_dpdh(ref.obj.p.val_SI, ref.obj.h.val_SI, ref.obj.fluid_data, ref.obj.mixing_rule) * ref.factor
        #     self.network.tensor[ref.obj.p.J_col, ref.obj.h.J_col, k] = tensor_2
        #     self.network.tensor[ref.obj.h.J_col, ref.obj.p.J_col, k] = tensor_2

    def calc_viscosity(self, T0=None):  # viscosity 粘度
        try:
            return viscosity_mix_ph(self.p.val_SI, self.h.val_SI, self.fluid_data, self.mixing_rule, T0=T0)
        except NotImplementedError:
            return np.nan

    def calc_vol(self, T0=None):  # volume
        try:
            return v_mix_ph(self.p.val_SI, self.h.val_SI, self.fluid_data, self.mixing_rule, T0=T0)
        except NotImplementedError:
            return np.nan

    def v_func(self, **kwargs):
        return self.calc_vol(T0=self.T.val_SI) * self.m.val_SI - self.v.val_SI

    def v_variables_columns(self, **kwargs):
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [self.p, self.h, self.m] if data.is_var]
        # variables_columns1 += [self.fluid.J_col[fluid] for fluid in self.fluid.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def v_take_effect(self, **kwargs):
        pass

    def v_solve_isolated(self, **kwargs):
        if self.fluid.is_var:
            return False
        if not self.m.is_var and not self.p.is_var and not self.h.is_var:
            self.v.is_set = False
            return True
        elif self.m.is_var and not self.p.is_var and not self.h.is_var:
            v = self.calc_vol(T0=self.T.val_SI)
            self.m.val_SI = self.v.val_SI / v
            self.m.is_set = True
            self.m.is_var = False
            self.v.is_set = False
            return True
        elif not self.m.is_var and self.p.is_var and not self.h.is_var:
            v = self.v.val_SI / self.m.val_SI
            self.p.val_SI = p_mix_hv(self.h.val_SI, v, self.fluid_data, self.mixing_rule)
            self.p.is_set = True
            self.p.is_var = False
            self.v.is_set = False
            return True
        elif not self.m.is_var and not self.p.is_var and self.h.is_var:
            v = self.v.val_SI / self.m.val_SI
            self.h.val_SI = h_mix_pv(self.p.val_SI, v, self.fluid_data, self.mixing_rule)
            self.h.is_set = True
            self.h.is_var = False
            self.v.is_set = False
            return True
        return False

    def v_deriv(self, increment_filter, k, **kwargs):  # m,p,h are the variables iterated merely
        if self.m.is_var:
            self.network.jacobian[k, self.m.J_col] = self.calc_vol(T0=self.T.val_SI)
        if self.p.is_var:
            self.network.jacobian[k, self.p.J_col] = (
                    dv_mix_dph(self.p.val_SI, self.h.val_SI, self.fluid_data, self.mixing_rule) * self.m.val_SI)
        if self.h.is_var:
            self.network.jacobian[k, self.h.J_col] = (
                    dv_mix_pdh(self.p.val_SI, self.h.val_SI, self.fluid_data, self.mixing_rule) * self.m.val_SI)
        # for fluid in self.fluid.is_var:
        #     self.network.jacobian[k, self.fluid.J_col[fluid]] = dv_mix_ph_dfluid(
        #         self.p.val_SI, self.h.val_SI, fluid, self.fluid_data, self.mixing_rule
        #     ) * self.m.val_SI

    def v_tensor(self, increment_filter, k, **kwargs):
        if self.p.is_var:
            self.network.tensor[self.p.J_col, self.p.J_col, k] = (
                    d2v_mix_d2p_h(self.p.val_SI, self.h.val_SI, self.fluid_data, self.mixing_rule) * self.m.val_SI)
        if self.h.is_var:
            self.network.tensor[self.h.J_col, self.h.J_col, k] = (
                    d2v_mix_p_d2h(self.p.val_SI, self.h.val_SI, self.fluid_data, self.mixing_rule) * self.m.val_SI)
        if self.p.is_var and self.h.is_var:
            tensor_2 = d2v_mix_dp_dh(self.p.val_SI, self.h.val_SI, self.fluid_data, self.mixing_rule) * self.m.val_SI
            self.network.tensor[self.p.J_col, self.h.J_col, k] = tensor_2
            self.network.tensor[self.h.J_col, self.p.J_col, k] = tensor_2

    def v_ref_func(self, **kwargs):
        ref = self.v_ref.ref
        return (
            self.calc_vol(T0=self.T.val_SI) * self.m.val_SI
            - (ref.obj.calc_vol(T0=ref.obj.T.val_SI) * ref.obj.m.val_SI * ref.factor + ref.delta_SI)
        )

    def v_ref_variables_columns(self, **kwargs):
        variables_columns = self.v_variables_columns(**kwargs)
        ref = self.v_ref.ref
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [ref.obj.p, ref.obj.h, ref.obj.m] if data.is_var]
        # variables_columns1 += [ref.obj.fluid.J_col[fluid] for fluid in ref.obj.fluid.is_var]
        variables_columns[0] = sorted(list(set(variables_columns[0] + variables_columns1)))
        return variables_columns

    def v_ref_take_effect(self, **kwargs):
        pass

    def v_ref_solve_isolated(self, **kwargs):
        # !!!!!!!!!!!
        return False

    def v_ref_deriv(self, increment_filter, k, **kwargs):
        # first part of sum is identical to direct volumetric flow specification
        self.v_deriv(increment_filter, k, **kwargs)  #
        ref = self.v_ref.ref
        if ref.obj.m.is_var:
            self.network.jacobian[k, ref.obj.m.J_col] = -(
                ref.obj.calc_vol(T0=ref.obj.T.val_SI) * ref.factor
            )
        if ref.obj.p.is_var:
            self.network.jacobian[k, ref.obj.p.J_col] = -(
                dv_mix_dph(ref.obj.p.val_SI, ref.obj.h.val_SI, ref.obj.fluid_data, ref.obj.mixing_rule)
                * ref.obj.m.val_SI * ref.factor
            )
        if ref.obj.h.is_var:
            self.network.jacobian[k, ref.obj.h.J_col] = -(
                dv_mix_pdh(ref.obj.p.val_SI, ref.obj.h.val_SI, ref.obj.fluid_data, ref.obj.mixing_rule)
                * ref.obj.m.val_SI * ref.factor
            )
        # for fluid in ref.obj.fluid.is_var:
        #     if not increment_filter[ref.obj.fluid.J_col[fluid]]:  # !!!!!!!!
        #         self.network.jacobian[k, ref.obj.fluid.J_col[fluid]] += -dv_mix_ph_dfluid(
        #             ref.obj.p.val_SI, ref.obj.h.val_SI, fluid, ref.obj.fluid_data, ref.obj.mixing_rule
        #         ) * ref.obj.m.val_SI * ref.factor

    def v_ref_tensor(self, increment_filter, k, **kwargs):
        self.v_tensor(increment_filter, k, **kwargs)
        ref = self.v_ref.ref
        if ref.obj.p.is_var:
            self.network.tensor[ref.obj.p.J_col, ref.obj.p.J_col, k] = (
                    -d2v_mix_d2p_h(ref.obj.p.val_SI, ref.obj.h.val_SI, ref.obj.fluid_data, ref.obj.mixing_rule)
                    * ref.obj.m.val_SI * ref.factor)
        if ref.obj.h.is_var:
            self.network.tensor[ref.obj.h.J_col, ref.obj.h.J_col, k] = (
                    -d2v_mix_p_d2h(ref.obj.p.val_SI, ref.obj.h.val_SI, ref.obj.fluid_data, ref.obj.mixing_rule)
                    * ref.obj.m.val_SI * ref.factor)
        if ref.obj.p.is_var and ref.obj.h.is_var:
            tensor_2 = (-d2v_mix_dp_dh(ref.obj.p.val_SI, ref.obj.h.val_SI, ref.obj.fluid_data, ref.obj.mixing_rule)
                        * ref.obj.m.val_SI * ref.factor)
            self.network.tensor[ref.obj.p.J_col, ref.obj.h.J_col, k] = tensor_2
            self.network.tensor[ref.obj.h.J_col, ref.obj.p.J_col, k] = tensor_2

    def calc_x(self):
        try:
            return Q_mix_ph(self.p.val_SI, self.h.val_SI, self.fluid_data)
        except NotImplementedError:
            return np.nan

    def x_func(self, **kwargs):
        # saturated steam fraction
        return self.h.val_SI - h_mix_pQ(self.p.val_SI, self.x.val_SI, self.fluid_data)

    def x_variables_columns(self, **kwargs):
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [self.h] if data.is_var]  # [self.p, self.h]
        variables_columns1.sort()
        return [variables_columns1]

    def x_take_effect(self, **kwargs):
        pass

    def x_solve_isolated(self, **kwargs):
        if self.fluid.is_var:
            logger.error(f"phase information error for multiple fluid composition")
            return False
        if self.h.is_var and not self.p.is_var:
            self.h.val_SI = h_mix_pQ(self.p.val_SI, self.x.val_SI, self.fluid_data)
            self.h.is_set = True
            self.h.is_var = False
            self.x.is_set = False
            return True
        elif not self.h.is_var and self.p.is_var:
            self.p.val_SI = p_mix_hQ(self.h.val_SI, self.x.val_SI, self.fluid_data)
            self.p.is_set = True
            self.p.is_var = False
            self.x.is_set = False
            return True
        elif not self.h.is_var and not self.p.is_var:
            self.x.is_set = False
            return True
        return False

    def x_deriv(self, increment_filter, k, **kwargs):  # m,p,h are the variables iterated merely
        # if self.p.is_var:
        #     self.network.jacobian[k, self.p.J_col] = -dh_mix_dpQ(self.p.val_SI, self.x.val_SI, self.fluid_data)
        if self.h.is_var:
            self.network.jacobian[k, self.h.J_col] = 1

    def x_tensor(self, increment_filter, k, **kwargs):
        if self.p.is_var:
            self.network.tensor[self.p.J_col, self.p.J_col, k] = -d2h_mix_d2p_Q(self.p.val_SI, self.x.val_SI, self.fluid_data)

    def calc_T_sat(self):  # set boiling or condensing point temperature
        try:
            return T_sat_p(self.p.val_SI, self.fluid_data)
        except NotImplementedError:
            return np.nan

    def calc_Td_bp(self):
        try:
            return self.calc_T() - T_sat_p(self.p.val_SI, self.fluid_data)
        except NotImplementedError:
            return np.nan

    def calc_Td_dew(self):
        try:
            return T_sat_p(self.p.val_SI, self.fluid_data) - self.calc_T()
        except NotImplementedError:
            return np.nan

    def Td_bp_func(self, **kwargs):
        """
        Calculate temperature difference above boiling point

        :param kwargs:
        :return:
        """
        return self.calc_Td_bp() - self.Td_bp.val_SI

    def Td_bp_variables_columns(self, **kwargs):
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [self.h] if data.is_var]  # [self.p, self.h]
        variables_columns1.sort()
        return [variables_columns1]

    def Td_bp_take_effect(self, **kwargs):
        pass

    def Td_bp_solve_isolated(self, **kwargs):
        if self.fluid.is_var:
            logger.error(f"phase information error for multiple fluid composition in Td_bp equation of Connection: {self.label}")
            return False
        if self.h.is_var and not self.p.is_var:
            self.T.val_SI = T_sat_p(self.p.val_SI, self.fluid_data) + self.Td_bp.val_SI
            self.h.val_SI = h_mix_pT(self.p.val_SI, self.T.val_SI, self.fluid_data, self.mixing_rule)
            self.h.is_set = True
            self.h.is_var = False
            self.Td_bp.is_set = False
            return True
        elif not self.h.is_var and self.p.is_var:  # !!!!!
            return False
        elif not self.h.is_var and not self.p.is_var:
            self.Td_bp.is_set = False
            return True
        return False

    def Td_bp_deriv(self, increment_filter, k, **kwargs):  # m,p,h are the variables iterated merely
        # if self.p.is_var:
        #     self.network.jacobian[k, self.p.J_col] = (
        #         dT_mix_dph(self.p.val_SI, self.h.val_SI, self.fluid_data)
        #         - dT_sat_dp(self.p.val_SI, self.fluid_data)
        #     )
        if self.h.is_var:
            self.network.jacobian[k, self.h.J_col] = dT_mix_pdh(
                self.p.val_SI, self.h.val_SI, self.fluid_data
            )

    def Td_bp_repair_matrix(self, property_, **kwargs):
        if property_ == self.h:
            h0 = h_mix_pQ(self.p.val_SI, 0, self.fluid_data)
            h1 = h_mix_pQ(self.p.val_SI, 1, self.fluid_data)
            return abs(self.calc_Td_bp() - self.Td_bp.val_SI) / max(self.h.val_SI - h0, h1 - self.h.val_SI)
        else:
            msg = f"variable: {property_.label} is not a valid property in Td_bp_repair_matrix of Connection: {self.label}"
            raise ValueError(msg)

    def Td_bp_tensor(self, increment_filter, k, **kwargs):
        # if self.p.is_var:
        #     self.network.tensor[self.p.J_col, self.p.J_col, k] = (d2T_mix_d2p_h(self.p.val_SI, self.h.val_SI, self.fluid_data)
        #                                                           - d2T_sat_d2p(self.p.val_SI, self.fluid_data))
        if self.h.is_var:
            self.network.tensor[self.h.J_col, self.h.J_col, k] = d2T_mix_p_d2h(self.p.val_SI, self.h.val_SI, self.fluid_data)
        # if self.p.is_var and self.h.is_var:
        #     tensor_2 = d2T_mix_dpdh(self.p.val_SI, self.h.val_SI, self.fluid_data)
        #     self.network.tensor[self.p.J_col, self.h.J_col, k] = tensor_2
        #     self.network.tensor[self.h.J_col, self.p.J_col, k] = tensor_2

    def Td_dew_func(self, **kwargs):
        """
        Calculate temperature difference under boiling point

        :param kwargs:
        :return:
        """
        return self.calc_Td_dew() - self.Td_dew.val_SI

    def Td_dew_variables_columns(self, **kwargs):
        variables_columns1 = []
        variables_columns1 += [data.J_col for data in [self.h] if data.is_var]  # [self.p, self.h]
        variables_columns1.sort()
        return [variables_columns1]

    def Td_dew_take_effect(self, **kwargs):
        pass

    def Td_dew_solve_isolated(self, **kwargs):
        if self.fluid.is_var:
            logger.error(f"phase information error for multiple fluid composition in Td_dew equation of Connection: {self.label}")
            return False
        if self.h.is_var and not self.p.is_var:
            self.T.val_SI = T_sat_p(self.p.val_SI, self.fluid_data) - self.Td_dew.val_SI
            self.h.val_SI = h_mix_pT(self.p.val_SI, self.T.val_SI, self.fluid_data, self.mixing_rule)
            self.h.is_set = True
            self.h.is_var = False
            self.Td_dew.is_set = False
            return True
        elif not self.h.is_var and self.p.is_var:  # !!!!!
            return False
        elif not self.h.is_var and not self.p.is_var:
            self.Td_dew.is_set = False
            return True
        return False

    def Td_dew_deriv(self, increment_filter, k, **kwargs):
        if self.h.is_var:
            self.network.jacobian[k, self.h.J_col] = -dT_mix_pdh(
                self.p.val_SI, self.h.val_SI, self.fluid_data
            )

    def Td_dew_repair_matrix(self, property_, **kwargs):
        if property_ == self.h:
            h0 = h_mix_pQ(self.p.val_SI, 0, self.fluid_data)
            h1 = h_mix_pQ(self.p.val_SI, 1, self.fluid_data)
            return -abs(self.calc_Td_dew() - self.Td_dew.val_SI) / max(self.h.val_SI - h0, h1 - self.h.val_SI)
        else:
            msg = f"variable: {property_.label} is not a valid property in Td_dew_repair_matrix of Connection: {self.label}"
            raise ValueError(msg)

    def Td_dew_tensor(self, increment_filter, k, **kwargs):
        pass

    def fluid_balance_func(self, **kwargs):
        residual = 1 - sum(self.fluid.val[f] for f in self.fluid.is_set)
        residual -= sum(self.fluid.val[f] for f in self.fluid.is_var)
        return residual

    def fluid_balance_variables_columns(self, **kwargs):
        variables_columns1 = []
        variables_columns1 += [self.fluid.J_col[fluid] for fluid in self.fluid.is_var]
        variables_columns1.sort()
        return [variables_columns1]

    def fluid_balance_take_effect(self, **kwargs):
        pass

    def fluid_balance_solve_isolated(self, **kwargs):

        return False

    def fluid_balance_deriv(self, increment_filter, k, **kwargs):
        for f in self.fluid.is_var:
            self.network.fluid_jacobian[k, self.fluid.J_col[f]] = -1

    def fluid_balance_tensor(self, increment_filter, k, **kwargs):
        pass

    def calc_s(self):
        try:
            return s_mix_ph(self.p.val_SI, self.h.val_SI, self.fluid_data, self.mixing_rule, T0=self.T.val_SI)
        except NotImplementedError:
            return np.nan

    def calc_Q(self):  # vapour mass fraction
        return Q_mix_ph(self.p.val_SI, self.h.val_SI, self.fluid_data)

    def calc_phase(self):
        try:
            return phase_mix_ph(self.p.val_SI, self.h.val_SI, self.fluid_data)
        except NotImplementedError:
            return np.nan

    def calc_p_critical(self):
        return p_critical_fluids(self, self.fluid_data)

    def spread_pressure_reference_check(self):
        if np.isin(self, self.network.reference_pressure_object_container):
            indices = np.where(self.network.reference_pressure_object_container == self)
            coordinates = list(zip(indices[0], indices[1]))
            for index in coordinates:
                row_index, column_index = index[0], index[1]
                if column_index == 0:
                    refer_conn = self.network.reference_pressure_object_container[row_index][1]
                    if self.p.is_set and refer_conn.p.is_set:
                        self.p_ref.is_set = False
                    elif self.p.is_set and not refer_conn.p.is_set:
                        refer_conn.p.val_SI = ((self.p.val_SI - self.p_ref.ref.delta_SI) / self.p_ref.ref.factor)
                        refer_conn.p.is_set = True
                        refer_conn.p.is_var = False
                        self.p_ref.is_set = False
                        self.network.reference_pressure_object_container = np.delete(
                            self.network.reference_pressure_object_container,
                            row_index,
                            axis=0)
                        refer_conn.spread_pressure_referent_values()
                elif column_index == 1:
                    refered_conn = self.network.reference_pressure_object_container[row_index][0]
                    if self.p.is_set and refered_conn.p.is_set:
                        refered_conn.p_ref.is_set = False
                    elif self.p.is_set and not refered_conn.p.is_set:
                        refered_conn.p.val_SI = self.p.val_SI * refered_conn.p_ref.ref.factor + refered_conn.p_ref.ref.delta_SI
                        refered_conn.p.is_set = True
                        refered_conn.p.is_var = False
                        refered_conn.p_ref.is_set = False
                        self.network.reference_pressure_object_container = np.delete(
                            self.network.reference_pressure_object_container,
                            row_index,
                            axis=0)
                        refered_conn.spread_pressure_referent_values()
        else:
            return

    def spread_pressure_referent_values(self):
        if self not in self.network.connections_spread_pressure_container:
            self.network.connections_spread_pressure_container.append(self)
            self.target.spread_forward_pressure_values(self)
            self.source.spread_backward_pressure_values(self)
            self.spread_pressure_reference_check()
        return

    def calc_results(self):
        self.T.val_SI = self.calc_T()
        number_fluids = get_number_of_fluids(self.fluid_data)
        _converged = True
        if number_fluids > 1:
            h_from_T = h_mix_pT(self.p.val_SI, self.T.val_SI, self.fluid_data, self.mixing_rule)
            if abs(h_from_T - self.h.val_SI) > ERR ** .5 and abs((h_from_T - self.h.val_SI) / self.h.val_SI) > ERR ** .5:
                self.T.val_SI = np.nan
                self.vol.val_SI = np.nan
                self.v.val_SI = np.nan
                self.s.val_SI = np.nan
                msg = (
                    "Could not find a feasible value for mixture temperature at "
                    f"connection {self.label}. The values for temperature, "
                    "specific volume, volumetric flow and entropy are set to nan."
                )
                logger.error(msg)
                _converged = False
            else:
                _, Tmax = get_mixture_temperature_range(self.fluid_data)
                if self.T.val_SI > Tmax:
                    msg = (
                        "The temperature value of the mixture is above the "
                        "upper temperature limit of a mixture component. The "
                        "resulting temperature may have larger deviations "
                        "compared to the tolerance specified in the "
                        "corresponding substance property library."
                    )
                    logger.warning(msg)
        else:
            try:
                if not self.x.is_set:
                    self.x.val_SI = self.calc_x()
            except ValueError:
                self.x.val_SI = np.nan

            try:
                self.phase.val = self.calc_phase()
            except ValueError:
                self.phase.val = np.nan

            try:
                if not self.Td_bp.is_set:
                    self.Td_bp.val_SI = self.calc_Td_bp()
            except ValueError:
                self.Td_bp.val_SI = np.nan

            try:
                if not self.Td_dew.is_set:
                    self.Td_dew.val_SI = self.calc_Td_dew()
            except ValueError:
                self.Td_dew.val_SI = np.nan

        if _converged:
            self.vol.val_SI = self.calc_vol()
            self.v.val_SI = self.vol.val_SI * self.m.val_SI
            self.s.val_SI = self.calc_s()

        for prop in fpd.keys():
            self.get_attr(prop).val = convert_from_SI(
                self.get_attr(prop).property_data, self.get_attr(prop).val_SI, self.get_attr(prop).unit
            )

        self.m.val0 = self.m.val
        self.p.val0 = self.p.val
        self.h.val0 = self.h.val
        self.fluid.val0 = self.fluid.val.copy()

    def bounds_T_generate(self):
        if self.h.is_var:
            Tmin = max(
                [w._T_min for f, w in self.fluid.wrapper.items() if self.fluid.val[f] > ERR]
            ) * 1.01
            Tmax = min(
                [w._T_max for f, w in self.fluid.wrapper.items() if self.fluid.val[f] > ERR]
            ) * 0.99
            self.T.min_val, self.T.max_val = Tmin, Tmax
        if len(self.fluid.val) > 1:
            self.T.min_val = 276

    def bounds_ph_generate(self):
        self.p.min_val, self.p.max_val = self.network.p_range_SI
        self.h.min_val, self.h.max_val = self.network.h_range_SI
        if not self.p.is_var:
            # h range for sure p
            if self.h.is_var:
                if len(self.fluid.val) <=1:
                    Tmin, Tmax = self.T.min_val, self.T.max_val
                    while True:
                        try:
                            hmin = h_mix_pT(self.p.val_SI, Tmin + 1e0, self.fluid_data, self.mixing_rule)
                            break
                        except ValueError as e:
                            Tmin *= 1.05
                            if Tmin > Tmax:
                                raise ValueError(e) from e
                    while True:
                        try:
                            hmax = h_mix_pT(self.p.val_SI, Tmax - 1e0, self.fluid_data, self.mixing_rule)
                            break
                        except ValueError as e:
                            Tmax *= 0.99
                            if Tmax < Tmin:
                                raise ValueError(e) from e
                    self.h.min_val, self.h.max_val = hmin, hmax
                else:
                    self.h.min_val, self.h.max_val = self.network.h_range_SI[0], self.network.h_range_SI[1]
                    # if "H2O" in self.fluid.val and "CO2" in self.fluid.val:
                    #     Tmin = max(
                    #         [w._T_min for f, w in self.fluid.wrapper.items()]
                    #     ) * 1.01
                    #     Tmax = min(
                    #         [w._T_max for f, w in self.fluid.wrapper.items()]
                    #     ) * 0.99
                    #     Tmin = min(Tmin, 300)
                    #     while True:
                    #         try:
                    #             hmin = h_mix_pT(self.p.val_SI, Tmin + 1e0, self.fluid_data, self.mixing_rule)
                    #             break
                    #         except ValueError as e:
                    #             Tmin *= 1.05
                    #             if Tmin > Tmax:
                    #                 raise ValueError(e) from e
                    #     print(f"the {self.label}.h: {h_mix_pT(self.p.val_SI, 90.7941, self.fluid_data, 'ideal-cond')}---{hmin}"
                    #           f"--T: {T_mix_ph(self.p.val_SI, 188000, self.fluid_data, 'ideal-cond')}--{Tmin}   {self.mixing_rule}")
                    #     self.h.min_val = hmin
        elif self.p.is_var:
            p_min = min(max([w._p_min for f, w in self.fluid.wrapper.items()]) * 1.001,
                        self.network.p_range_SI[0])
            p_max = min([w._p_max for f, w in self.fluid.wrapper.items()]) * 0.999
            self.p.min_val, self.p.max_val = p_min, p_max

    def calc_min_enthalpy(self, p):
        Tmin, Tmax = self.T.min_val, self.T.max_val
        while True:
            try:
                hmin = h_mix_pT(p, Tmin + 1e0, self.fluid_data, self.mixing_rule)
                break
            except ValueError as e:
                Tmin *= 1.05
                if Tmin > Tmax:
                    raise ValueError(e) from e
        # print(f'T range: [{Tmin}, {Tmax}], h_min: {hmin}')
        return hmin

    def calc_max_enthalpy(self, p):
        Tmin, Tmax = self.T.min_val, self.T.max_val
        while True:
            try:
                hmax = h_mix_pT(p, Tmax - 1e0, self.fluid_data, self.mixing_rule)
                break
            except ValueError as e:
                Tmax *= 0.99
                if Tmax < Tmin:
                    raise ValueError(e) from e
        # print(f'T range: [{Tmin}, {Tmax}], h_max: {hmax}')
        return hmax

    def get_physical_exergy(self, pamb, Tamb):  # analysis the
        r"""
        Get the value of a connection's specific physical exergy.

        ex_therm: thermal exergy
        ex_mech: mechanical exergy

        Parameters
        ----------
        pamb : float
            Ambient pressure p0 / Pa.

        Tamb : float
            Ambient temperature T0 / K.

        Note
        ----
            .. math::

                e^\mathrm{PH} = e^\mathrm{T} + e^\mathrm{M}\\
                E^\mathrm{T} = \dot{m} \cdot e^\mathrm{T}\\
                E^\mathrm{M} = \dot{m} \cdot e^\mathrm{M}\\
                E^\mathrm{PH} = \dot{m} \cdot e^\mathrm{PH}
        """
        self.ex_therm, self.ex_mech = fp.functions.calc_physical_exergy(
            self.h.val_SI, self.s.val_SI, self.p.val_SI, pamb, Tamb,
            self.fluid_data, self.mixing_rule, self.T.val_SI
        )
        self.Ex_therm = self.ex_therm * self.m.val_SI
        self.Ex_mech = self.ex_mech * self.m.val_SI

        self.ex_physical = self.ex_therm + self.ex_mech
        self.Ex_physical = self.m.val_SI * self.ex_physical

    def get_chemical_exergy(self, pamb, Tamb, Chem_Ex):
        r"""
        Get the value of a connection's specific chemical exergy.

        ex_chemical: chemical exergy

        Parameters
        ----------
        p0 : float
            Ambient pressure p0 / Pa.

        T0 : float
            Ambient temperature T0 / K.

        Chem_Ex : dict
            Lookup table for standard specific chemical exergy.

        Note
        ----
            .. math::

                E^\mathrm{CH} = \dot{m} \cdot e^\mathrm{CH}
        """
        if Chem_Ex is None:
            self.ex_chemical = 0
        else:
            self.ex_chemical = fp.functions.calc_chemical_exergy(
                pamb, Tamb, self.fluid_data, Chem_Ex, self.mixing_rule,
                self.T.val_SI
            )

        self.Ex_chemical = self.m.val_SI * self.ex_chemical








