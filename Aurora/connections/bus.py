# -*- coding: utf-8

"""Module of class Bus.
"""

import numpy as np
import pandas as pd

from Aurora.components.component import Component
from Aurora.tools import logger
from Aurora.tools.characteristics import CharLine
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.global_vars import property_scale as ps
from Aurora.tools.global_vars import component_property_data as cpd


class Bus:
    r"""
    A bus is used to connect different energy flows.

    Parameters
    ----------
    label : str
        Label for the bus.

    P :
        Total power/heat flow specification for bus, :math:`P \text{/W}`.

    printout : boolean
        Print the results of this bus to prompt with the
        :py:meth:`tespy.networks.network.Network.print_results` method.
        Standard value is :code:`True`.

    Example
    -------
    Busses are used to connect energy flow of different components. They can
    also be used to introduce efficiencies of energy conversion, e.g. in
    motors, generator or boilers. This example takes the combustion engine
    example at
    :py:class:`tespy.components.combustion.engine.CombustionEngine`
    and adds a flue gas cooler and a circulation pump for the cooling water.
    Then busses for heat output, thermal input and electricity output are
    implemented.
    """
    def __init__(self, label, **kwargs):
        dtypes = {
            "param": str,
            "P_ref": float,
            "char": object,
            "efficiency": float,
            "base": str,
            # "unit": str,
        }
        self.comps = pd.DataFrame(
            columns=list(dtypes.keys())
        ).astype(dtypes)
        self.label = label
        self.P = dc_cp(val=np.nan,
                           is_set=False,
                           property_data=cpd['P'],
                           SI_unit=cpd['P']['SI_unit'],
                           scale=ps['m']['scale'] * ps['h']['scale'],
                           var_scale=ps['m']['scale'] * ps['h']['scale'])
        self.char = CharLine(x=np.array([0, 3]), y=np.array([1, 1]))
        self.printout = True
        self.set_attr(**kwargs)
        msg = f"Created bus {self.label}."
        logger.debug(msg)

    def set_attr(self, **kwargs):
        r"""
        Set, reset or unset attributes of a bus object.

        Parameters
        ----------
        label : str
            Label for the bus.

        P : float
            Total power/heat flow specification for bus, :math:`P \text{/W}`.

        printout : boolean
            Print the results of this bus to prompt with the
            :py:meth:`Aurora.networks.network.Network.print_results` method.
            Standard value is :code:`True`.

        Note
        ----
        Specify :math:`P=\text{nan}`, if you want to unset the value of P.
        """
        for key in kwargs:
            try:
                float(kwargs[key])
                is_numeric = True
            except (TypeError, ValueError):
                is_numeric = False
            if key == 'P':
                if is_numeric:
                    if np.isnan(kwargs[key]):
                        self.P.set_attr(is_set=False)
                    else:
                        self.P.set_attr(val=kwargs[key], is_set=True)
                elif kwargs[key] is None:
                    self.P.set_attr(is_set=False)
                else:
                    msg = f"Keyword argument {key} must be numeric."
                    logger.error(msg)
                    raise TypeError(msg)
            elif key == 'P_unit':
                if kwargs[key] in self.P.property_data['units']:
                    self.P.unit = kwargs[key]
                else:
                    msg = f'The unit {kwargs[key]} does not exist in Bus {self.label}.'
                    logger.error(msg)
                    raise AttributeError(msg)
            elif key == 'printout':
                if not isinstance(kwargs[key], bool):
                    msg = f"Please provide the {key} as boolean."
                    logger.error(msg)
                    raise TypeError(msg)
                else:
                    self.__dict__.update({key: kwargs[key]})
            # invalid keyword
            else:
                msg = f"A bus has no attribute {key}."
                logger.error(msg)
                raise KeyError(msg)

    def get_attr(self, key):
        r"""
        Get the value of a busses attribute.

        Parameters
        ----------
        key : str
            The attribute you want to retrieve.

        Returns
        -------
        out :
            Specified attribute.
        """
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            msg = f"Bus {self.label} has no attribute {key}."
            logger.error(msg)
            raise KeyError(msg)

    def add_comps(self, *args):
        r"""
        Add components to a bus.

        Parameters
        ----------
        *args : dict
            Dictionaries containing the component information to be added to
            the bus. The information are described below.

        Note
        ----
        **Required Key**

        - comp (Aurora.components.component.Component): Component you want to
          add to the bus.

        **Optional Keys**

        - param (str): Bus parameter, optional.

            - You do not need to provide a parameter, if the component only has
              one option for the bus (turbomachines, heat exchangers,
              combustion chamber).
            - For instance, you do neet do provide a parameter, if you want to
              add a combustion engine ('Q', 'Q1', 'Q2', 'TI', 'P', 'Qloss').

        - char (float/Aurora.components.characteristics.characteristics):
          Characteristic function for this components share to the bus value,
          optional.

            - If you do not provide a characteristic line at all, TESPy assumes
              a constant factor of 1.
            - If you provide a numeric value instead of a characteristic line,
              TESPy takes this numeric value as a constant factor.
            - Provide a :py:class:`Aurora.tools.characteristics.CharLine`, if
              you want the factor to follow a characteristic line.

        - P_ref (float): Energy flow specification for reference case,
          :math:`P \text{/W}`, optional.
        - base (str): Base value for characteristic line and efficiency
          calculation. The base can either be :code:`'component'` (default) or
          :code:`'bus'`.

            - In case you choose :code:`'component'`, the characteristic line
              input will follow the value of the component's bus function and
              the efficiency definition is
              :math:`\eta=\frac{P_\mathrm{bus}}{P_\mathrm{component}}`.
            - In case you choose :code:`'bus'`, the characteristic line
              input will follow the bus value of the component and the
              efficiency definition is
              :math:`\eta=\frac{P_\mathrm{component}}{P_\mathrm{bus}}`.
        """
        for c in args:
            if isinstance(c, dict):
                if 'comp' in c:
                    comp = c['comp']
                    # default values
                    if isinstance(comp, Component):
                        self.comps.loc[comp] = [
                            None, np.nan, self.char, np.nan, 'component'
                        ]
                    else:
                        msg = f'Keyword "comp" must hold a component in Bus: {self.label}.'
                        logger.error(msg)
                        raise TypeError(msg)
                else:
                    msg = f'You must provide the component "comp" in Bus: {self.label}.'
                    logger.error(msg)
                    raise TypeError(msg)
                # component information
                for k, v in c.items():
                    if k == 'param':
                        if isinstance(v, str) or v is None:
                            self.comps.loc[comp, 'param'] = v
                        else:
                            msg = (
                                f"The bus parameter selection of Bus: {self.label} must be a string "
                                f"at bus {self.label}.")
                            logger.error(msg)
                            raise TypeError(msg)
                    elif k == 'char':
                        try:
                            float(v)
                            is_numeric = True
                        except (TypeError, ValueError):
                            is_numeric = False
                        if isinstance(v, CharLine):
                            self.comps.loc[comp, 'char'] = v
                        elif is_numeric:  # default stable char line
                            x = np.array([0, 3])
                            y = np.array([1, 1]) * v
                            self.comps.loc[comp, 'char'] = (
                                    CharLine(x=x, y=y))
                        else:
                            msg = (
                                'Char must be a number or a '
                                f'characteristics char line in Bus: {self.label}.')
                            logger.error(msg)
                            raise TypeError(msg)
                    elif k == 'P_ref':
                        try:
                            float(v)
                            is_numeric = True
                        except (TypeError, ValueError):
                            is_numeric = False
                        if v is None or is_numeric:
                            self.comps.loc[comp, 'P_ref'] = v
                        else:
                            msg = f'Reference value must be numeric in Bus: {self.label}.'
                            logger.error(msg)
                            raise TypeError(msg)
                    elif k == 'base':
                        if v in ['bus', 'component']:
                            self.comps.loc[comp, 'base'] = v
                        else:
                            msg = (
                                f'The base value must be "bus" or "component" in Bus: {self.label}.')
                            logger.error(msg)
                            raise ValueError(msg)
                    # elif k == 'unit':
                    #     if v in self.P.property_data['units']:
                    #         self.comps.loc[comp, 'unit'] = v
                    #     else:
                    #         msg = f'The unit must in {list(self.P.property_data["units"].keys())}.'
                    #         logger.error(msg)
                    #         raise ValueError(msg)
            msg = f"Added component {comp.label} to bus {self.label}."
            logger.debug(msg)

    def _serialize(self):
        export = {}
        export["P"] = self.P._serialize()
        for cp in self.comps.index:
            export[cp.label] = {}
            export[cp.label]["param"] = self.comps.loc[cp, "param"]
            export[cp.label]["base"] = self.comps.loc[cp, "base"]
            export[cp.label]["char"] = self.comps.loc[cp, "char"]._serialize()

        return {self.label: export}

    def power_bus_func(self):
        residual = self.P.val_SI
        for cp in self.comps.index:
            residual -= cp.calc_bus_value(self)
        return residual

    def power_bus_variables_columns(self):
        variables_columns = [[]]
        for cp in self.comps.index:
            variables_columns[0] += cp.bus_variables_columns(self)[0]
        variables_columns[0] = sorted(list(set(variables_columns[0])))
        return variables_columns

    def power_bus_deriv(self, increment_filter, k):
        for cp in self.comps.index:
            cp.bus_deriv(self, increment_filter, k)

    def power_bus_tensor(self):
        pass

    def summarize_equations(self):
        equation_container = dc_cons(func=self.power_bus_func,
                                     variables_columns=self.power_bus_variables_columns,
                                     deriv=self.power_bus_deriv,
                                     tensor=self.power_bus_tensor,
                                     constant_deriv=False,
                                     num_eq=1,
                                     scale=self.P.scale)
        equation_container.label = f'<Bus energy> of {self.label}'
        if self.P.is_set and equation_container.take_effect():
            self.network.sorted_equations_module_container.append(equation_container)


