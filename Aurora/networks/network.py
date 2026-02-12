# -*- coding: utf-8

"""Module for Aurora network class.
"""

import importlib
import json
import math
import os
import itertools
from time import time

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.linalg import null_space
from tabulate import tabulate

from Aurora.components.component import component_registry
from Aurora.connections import Bus
from Aurora.connections import Connection
from Aurora.connections import Ref

from Aurora.tools import fluid_properties as fp
from Aurora.tools import helpers as hlp
from Aurora.tools import logger

from Aurora.tools.characteristics import CharLine
from Aurora.tools.characteristics import CharMap

from Aurora.tools.data_containers import ComponentCharacteristicMaps as dc_cm
from Aurora.tools.data_containers import ComponentCharacteristics as dc_cc
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import FluidComposition as dc_flu
from Aurora.tools.data_containers import FluidProperties as dc_prop
from Aurora.tools.data_containers import GroupedComponentCharacteristics as dc_gcc
from Aurora.tools.data_containers import GroupedComponentProperties as dc_gcp
from Aurora.tools.data_containers import DataContainer as dc
from Aurora.tools.fluid_properties.wrappers import wrapper_registry
from Aurora.tools.global_vars import ERR
from Aurora.tools.global_vars import fluid_property_data as fpd
from Aurora.tools.global_vars import electromagnetic_property_data as epd
from Aurora.tools.global_vars import component_property_data as cpd
from Aurora.tools.global_vars import space_time_property_data as stpd
from Aurora.tools.global_vars import mathematical_property_data as mapd
from Aurora.tools.helpers import convert_from_SI

# Only require cupy if Cuda shall be used
try:
    import cupy as cu
except ModuleNotFoundError:
    cu = None


class Network:
    r"""
    Class component is the base class of all AURORA components.

    Parameters
    ----------
    h_range : list
        List with minimum and maximum values for enthalpy value range.

    h_unit : str
        Specify the unit for enthalpy: 'J / kg', 'kJ / kg', 'MJ / kg'.

    iterinfo : boolean
        Print convergence progress to console.

    m_range : list
        List with minimum and maximum values for mass flow value range.

    m_unit : str
        Specify the unit for mass flow: 'kg / s', 't / h'.

    p_range : list
        List with minimum and maximum values for pressure value range.

    p_unit : str
        Specify the unit for pressure: 'Pa', 'psi', 'bar', 'MPa'.

    s_unit : str
        Specify the unit for specific entropy: 'J / kgK', 'kJ / kgK',
        'MJ / kgK'.

    T_unit : str
        Specify the unit for temperature: 'K', 'C', 'F', 'R'.

    v_unit : str
        Specify the unit for volumetric flow: 'm3 / s', 'm3 / h', 'l / s',
        'l / h'.

    vol_unit : str
        Specify the unit for specific volume: 'm3 / kg', 'l / kg'.

    x_unit : str
        Specify the unit for steam mass fraction: '-', '%'.

    Note
    ----
    Unit specification is optional: If not specified the SI unit (first
    element in above lists) will be applied!

    Range specification is optional, too. The value range is used to stabilize
    the newton algorithm. For more information see the "getting started"
    section in the online-documentation.
    """

    def __init__(self, **kwargs):
        self.set_defaults()
        self.set_attr(**kwargs)

    def _serialize(self):
        return {
            "m_unit": self.properties_unit_citation['m_unit'],
            "m_range": list(self.m_range),
            "p_unit": self.properties_unit_citation['p_unit'],
            "p_range": list(self.p_range),
            "h_unit": self.properties_unit_citation['h_unit'],
            "h_range": list(self.h_range),
            "T_unit": self.properties_unit_citation['T_unit'],
            "x_unit": self.properties_unit_citation['x_unit'],
            "v_unit": self.properties_unit_citation['v_unit'],
            "s_unit": self.properties_unit_citation['s_unit'],
        }

    def set_defaults(self):
        """Set default network properties."""
        # connection dataframe
        dtypes = {
            "object": object,
            "source": object,
            "source_id": str,
            "target": object,
            "target_id": str,
            "conn_type": str,
        }
        self.conns = pd.DataFrame(
            columns=list(dtypes.keys())
        ).astype(dtypes)
        self.all_fluids = set()
        # component dataframe
        dtypes = {
            "comp_type": str,
            "object": object,
        }
        self.comps = pd.DataFrame(
            columns=list(dtypes.keys())
        ).astype(dtypes)
        # combines dataframe
        dtypes = {
            'combine_type': str,
            'object': object,
            'combined_components': object,
        }
        self.combs = pd.DataFrame(
            columns=list(dtypes.keys())
        ).astype(dtypes)
        # user defined function dictionary for fast access
        self.user_defined_eq = {}
        # bus dictionary
        self.busses = {}
        # results and specification dictionary
        self.results = {}
        self.specifications = {}  # ????
        # contain units information
        self.properties_unit_citation = {}

        # lookup: the detailed information
        self.specifications['lookup'] = {
            'properties': 'prop_specifications',  # value set dict
            'chars': 'char_specifications',  # car line/map set dict
            'variables': 'var_specifications',  # var set dict
            'groups': 'group_specifications'  # group data set dict
        }

        # in case of a design calculation after an offdesign calculation
        self.redesign = False  # delicate the switch on the calculating mode
        self.checked = False  # topologic construction check
        self.design_path = None
        self.iterinfo = True
        # units of fluid_property_data、electromagnetic_property_data、space_time_property_data
        msg = 'Default unit specifications:\n'
        for prop, data in itertools.chain(fpd.items(), epd.items(), stpd.items()):  # units of connections or nerwork
            # standard unit set, the property of network
            self.properties_unit_citation.update({prop + '_unit': data['SI_unit']})
            msg += data['text'] + ': ' + data['SI_unit'] + '\n'
        # don't need the last newline
        logger.debug(msg[:-1])
        # property variables information of connections
        self.variables_properties_connections_summar = {'fluid': ['m', 'p', 'h'],
                                                        'electric': ['U', 'I', 'f'],  # {'U': 2, 'I': 2, 'f': 1}
                                                        'magnetic': [],
                                                        'mechanical': []}
        self.connections_properties_data = {
            'fluid': list(fpd.keys()),
            'electric': list(epd.keys()),
        }
        # iterated value range
        self.m_range_SI = [1e-5, 1e12]  # !!!!
        self.p_range_SI = [1e5, 300e5]  # [2e2, 300e5]
        self.h_range_SI = [1e3, 7e6]  # [1e3, 7e6]
        self.T_range_SI = [2e2, 2e3]
        self.U_range_SI = [-1e12, 1e12]
        self.I_range_SI = [-1e12, 1e12]
        self.f_range_SI = [1e-5, 1e12]
        #
        for prop in list(itertools.chain(*self.variables_properties_connections_summar.values())):  #
            limits = self.get_attr(prop + '_range_SI')
            msg = (
                f"Default {dict(itertools.chain(fpd.items(), epd.items(), stpd.items()))[prop]['text']} limits\n"
                f"min: {limits[0]} {self.properties_unit_citation[prop + '_unit']}\n"  # default unit: SI_unit
                f"max: {limits[1]} {self.properties_unit_citation[prop + '_unit']}"
            )
            logger.debug(msg)

    def set_attr(self, **kwargs):
        r"""
        Set, resets or unsets attributes of a network.

        Parameters
        ----------
        h_range : list
            List with minimum and maximum values for enthalpy value range.

        h_unit : str
            Specify the unit for enthalpy: 'J / kg', 'kJ / kg', 'MJ / kg'.

        iterinfo : boolean
            Print convergence progress to console.

        m_range : list
            List with minimum and maximum values for mass flow value range.

        m_unit : str
            Specify the unit for mass flow: 'kg / s', 't / h'.

        p_range : list
            List with minimum and maximum values for pressure value range.

        p_unit : str
            Specify the unit for pressure: 'Pa', 'psi', 'bar', 'MPa'.

        s_unit : str
            Specify the unit for specific entropy: 'J / kgK', 'kJ / kgK',
            'MJ / kgK'.

        T_unit : str
            Specify the unit for temperature: 'K', 'C', 'F', 'R'.

        v_unit : str
            Specify the unit for volumetric flow: 'm3 / s', 'm3 / h', 'l / s',
            'l / h'.

        vol_unit : str
            Specify the unit for specific volume: 'm3 / kg', 'l / kg'.
        """
        # unit set
        for key, value in kwargs.items():
            if 'unit' in key:
                prop = key.split('_unit')[0]
                if prop in fpd.keys():
                    if value in fpd[prop]['units']:
                        self.properties_unit_citation.update({f'{prop}_unit': value})
                        msg = f'Setting fluid property {fpd[prop]["text"]} unit: {value}.'
                        logger.debug(msg)
                    else:
                        keys = ', '.join(fpd[prop]['units'].keys())
                        msg = (f"The unit: {value} not in units of fluid property: {fpd[prop]['text']}"
                               f'Allowed units for {fpd[prop]["text"]} are: {keys}')
                        logger.error(msg)
                        raise NotImplementedError(msg)
                elif prop in epd.keys():
                    if value in epd[prop]['units']:
                        self.properties_unit_citation.update({f'{prop}_unit': value})
                        msg = f'Setting electric property {epd[prop]["text"]} unit: {value}.'
                        logger.debug(msg)
                    else:
                        keys = ', '.join(epd[prop]['units'].keys())
                        msg = (f"The unit: {value} not in units of electromagnetic property: {epd[prop]['text']}"
                               f'Allowed units for {epd[prop]["text"]} are: {keys}')
                        logger.error(msg)
                        raise NotImplementedError(msg)
                elif prop in cpd.keys():
                    if value in cpd[prop]['units']:
                        self.properties_unit_citation.update({f'{prop}_unit': value})
                        msg = f'Setting component property {cpd[prop]["text"]} unit: {value}.'
                        logger.debug(msg)
                    else:
                        keys = ', '.join(cpd[prop]['units'].keys())
                        msg = (f"The unit: {value} not in units of component property: {cpd[prop]['text']}"
                               f'Allowed units for {cpd[prop]["text"]} are: {keys}')
                        logger.error(msg)
                        raise NotImplementedError(msg)
                elif prop in stpd.keys():
                    if value in stpd[prop]['units']:
                        self.properties_unit_citation.update({f'{prop}_unit': value})
                        msg = f'Setting space time property {stpd[prop]["text"]} unit: {value}.'
                        logger.debug(msg)
                    else:
                        keys = ', '.join(stpd[prop]['units'].keys())
                        msg = (f"The unit: {value} not in units of space or time property: {stpd[prop]['text']}"
                               f'Allowed units for {stpd[prop]["text"]} are: {keys}')
                        logger.error(msg)
                        raise NotImplementedError(msg)
                else:
                    msg = f"The {key} is not a valid property unit for present model."
                    logger.error(msg)
                    raise NotImplementedError(msg)
        # set the property range converted to SI second
        combined_properties_dict = dict(itertools.chain(fpd.items(), epd.items(), stpd.items()))
        for prop in list(itertools.chain(*self.variables_properties_connections_summar.values())):  #
            if f'{prop}_range' in kwargs:
                if isinstance(kwargs[f'{prop}_range'], list):
                    self.__dict__.update({
                        f'{prop}_range_SI': [hlp.convert_to_SI(
                            combined_properties_dict[prop], value,
                            self.properties_unit_citation[f'{prop}_unit']
                        ) for value in kwargs[f'{prop}_range']]
                    })
                else:
                    msg = f'Specify the range as list: [{prop}_min, {prop}_max]'
                    logger.error(msg)
                    raise TypeError(msg)
                limits = self.get_attr(f'{prop}_range_SI')
                msg = (
                    f'Setting {combined_properties_dict[prop]["text"]} limits\n'
                    f'min: {limits[0]} {combined_properties_dict[prop]["SI_unit"]}\n'
                    f'max: {limits[1]} {combined_properties_dict[prop]["SI_unit"]}'
                )
                logger.debug(msg)
        # update un_SI property ranges
        for prop in list(itertools.chain(*self.variables_properties_connections_summar.values())):  # ['m', 'p', 'h']
            SI_range = self.get_attr(f'{prop}_range_SI')
            self.__dict__.update({
                f'{prop}_range': [hlp.convert_from_SI(
                    combined_properties_dict[prop], SI_value,
                    self.properties_unit_citation[f'{prop}_unit']
                ) for SI_value in SI_range]
            })
        # whether to print iteration information
        self.iterinfo = kwargs.get('iterinfo', self.iterinfo)

    def get_attr(self, key):
        r"""
        Get the value of a network attribute.

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
            msg = f"Network has no attribute '{key}'."
            logger.error(msg)
            raise KeyError(msg)

    def add_subsys(self, *args):  # adding subsystems is adding connections of subsystems to network
        r"""
        Add one or more subsystems to the network.

        Parameters
        ----------
        c : Aurora.components.subsystem.Subsystem
            The subsystem to be added to the network, subsystem objects si
            :code:`network.add_subsys(s1, s2, s3, ...)`.
        """
        for subsys in args:
            for c in subsys.conns.values():
                self.add_conns(c)

    def get_conn(self, label):
        r"""
        Get Connection via label.

        Parameters
        ----------
        label : str
            Label of the Connection object.

        Returns
        -------
        c : Aurora.connections.connection.Connection
            Connection object with specified label, None if no Connection of
            the network has this label.
        """
        try:  # label: row ;  'object': column
            return self.conns.loc[label, 'object']
        except KeyError:
            logger.warning(f"Connection with label {label} not found.")
            return None

    def get_comp(self, label):
        r"""
        Get Component via label.

        Parameters
        ----------
        label : str
            Label of the Component object.

        Returns
        -------
        c : Aurora.components.component.Component
            Component object with specified label, None if no Component of
            the network has this label.
        """
        try:
            return self.comps.loc[label, 'object']
        except KeyError:
            logger.warning(f"Component with label {label} not found.")
            return None

    def add_conns(self, *args):
        r"""
        Add one or more connections to the network.

        Parameters
        ----------
        c : Aurora.connections.connection.Connection
            The connection to be added to the network, connections objects ci
            :code:`add_conns(c1, c2, c3, ...)`.
        """
        for c in args:
            if not isinstance(c, Connection):
                msg = (
                    f'Must provide objects that inherit from {Connection.__module__}.{Connection.__name__}, '
                    f'got {type(c).__module__}.{type(c).__name__} instead.'
                )
                logger.error(msg)
                raise TypeError(msg)
            elif c.label in self.conns.index:
                msg = (
                    'There is already a connection with the label '
                    f'{c.label}. The connection labels must be unique!'
                )
                logger.error(msg)
                raise ValueError(msg)
            c.good_starting_values = False  # delicate whether initialise the calculation with values from the previous calculation
            self.conns.loc[c.label] = [
                c, c.source, c.source_id, c.target, c.target_id, c.__class__.__name__
            ]
            msg = f'Added connection {c.label} to network.'
            logger.debug(msg)
            # set status "checked" to false, if connection is added to network.
            self.checked = False  # topologic construction need to be checked again
        self.conns = self.conns.sort_index()
        self._add_comps(*args)  # add comps belong to the added connections

    def del_conns(self, *args):
        """
        Remove one or more connections from the network.

        Parameters
        ----------
        c : Aurora.connections.connection.Connection
            The connection to be removed from the network, connections objects
            ci :code:`del_conns(c1, c2, c3, ...)`.
        """
        comps = list({cp for c in args for cp in [c.source, c.target]})
        for c in args:
            self.conns.drop(c.label, inplace=True)
            if c.__class__.__name__ in self.results:
                self.results[c.__class__.__name__].drop(
                    c.label, inplace=True, errors="ignore"
                )
            msg = f'Deleted connection {c.label} from network.'
            logger.debug(msg)
        self._del_comps(comps)
        # set status "checked" to false, if connection is deleted from network.
        self.checked = False  # topologic construction need to be checked again

    def _add_comps(self, *args):
        r"""
        Add to network's component DataFrame from added connections.

        Parameters
        ----------
        c : Aurora.connections.connection.Connection
            The connections, which have been added to the network. The
            components are extracted from these information.
        """
        # get unique components in new connections
        comps = list({cp for c in args for cp in [c.source, c.target]})
        # add to the dataframe of components
        for comp in comps:
            if comp.label in self.comps.index:
                if self.comps.loc[comp.label, 'object'] == comp:
                    continue
                else:
                    comp_type = comp.__class__.__name__
                    other_obj = self.comps.loc[comp.label, "object"]
                    other_comp_type = other_obj.__class__.__name__
                    msg = (
                        f"The component with the label {comp.label} of type "
                        f"{comp_type} cannot be added to the network as a "
                        f"different component of type {other_comp_type} with "
                        "the same label has already been added. All "
                        "components must have unique values!"
                    )
                    raise hlp.AURORANetworkError(msg)

            comp_type = comp.__class__.__name__
            self.comps.loc[comp.label, 'comp_type'] = comp_type
            self.comps.loc[comp.label, 'object'] = comp
        self.comps = self.comps.sort_index()

    def _del_comps(self, comps):
        r"""
        Delete from network's component DataFrame from deleted connections.

        For every component it is checked, if it is still part of other
        connections, which have not been deleted. The component is only
        removed if it cannot be found int the remaining connections.

        Parameters
        ----------
        comps : list
            List of components to potentially be deleted.
        """
        for comp in comps:
            if (
                comp not in self.conns["source"].values and
                comp not in self.conns["target"].values
            ):
                self.comps.drop(comp.label, inplace=True)
                if comp.__class__.__name__ in self.results:
                    self.results[comp.__class__.__name__].drop(
                        comp.label, inplace=True, errors="ignore"
                    )
                msg = f"Deleted component {comp.label} from network."
                logger.debug(msg)

    def add_ude(self, *args):  # add self defined equation to network
        r"""
        Add a user defined function to the network.

        Parameters
        ----------
        c : Aurora.tools.helpers.UserDefinedEquation
            The objects to be added to the network, UserDefinedEquation objects
            ci :code:`add_ude(c1, c2, c3, ...)`.
        """
        for c in args:
            if not isinstance(c, hlp.UserDefinedEquation):
                msg = (
                    'Must provide Aurora.tools.helpers.UserDefinedEquation '
                    'objects as parameters.'
                )
                logger.error(msg)
                raise TypeError(msg)
            elif c.label in self.user_defined_eq:
                msg = (
                    'There is already a UserDefinedEquation with the label '
                    f'{c.label} . The UserDefinedEquation labels must be '
                    'unique within a network'
                )
                logger.error(msg)
                raise ValueError(msg)
            self.user_defined_eq[c.label] = c
            msg = f"Added UserDefinedEquation {c.label} to network."
            logger.debug(msg)

    def del_ude(self, *args):  # delete self defined equation
        """
        Remove a user defined function from the network.

        Parameters
        ----------
        c : Aurora.tools.helpers.UserDefinedEquation
            The objects to be deleted from the network,
            UserDefinedEquation objects ci :code:`del_ude(c1, c2, c3, ...)`.
        """
        for c in args:
            del self.user_defined_eq[c.label]
            msg = f"Deleted UserDefinedEquation {c.label} from network."
            logger.debug(msg)

    def add_busses(self, *args):
        r"""
        Add one or more busses to the network.

        Parameters
        ----------
        b : Aurora.connections.bus.Bus
            The bus to be added to the network, bus objects bi
            :code:`add_busses(b1, b2, b3, ...)`.
        """
        for b in args:
            if self.check_busses(b):
                self.busses[b.label] = b
                msg = f"Added bus {b.label} to network."
                logger.debug(msg)
                dtypes = {
                    'component value': 'float64',
                    'bus value': 'float64',
                    'efficiency': 'float64',
                    'design value': 'float64',
                    'unit': str,
                }
                self.results[b.label] = pd.DataFrame(
                    columns=list(dtypes.keys()),
                ).astype(dtypes)

    def del_busses(self, *args):
        r"""
        Remove one or more busses from the network.

        Parameters
        ----------
        b : Aurora.connections.bus.Bus
            The bus to be removed from the network, bus objects bi
            :code:`add_busses(b1, b2, b3, ...)`.
        """
        for b in args:
            if b in self.busses.values():
                del self.busses[b.label]
                msg = f"Deleted bus {b.label} from network."
                logger.debug(msg)
                del self.results[b.label]

    def check_busses(self, b):
        r"""
        Checksthe busses to be added for type, duplicates and identical labels.

        Parameters
        ----------
        b : Aurora.connections.bus.Bus
            The bus to be checked.
        """
        if isinstance(b, Bus):
            if len(self.busses) > 0:
                if b in self.busses.values():
                    msg = f"The network contains the bus {b.label} already."
                    logger.error(msg)
                    raise hlp.AURORANetworkError(msg)
                elif b.label in self.busses:
                    msg = f"The network already has a bus labeled {b.label}."
                    logger.error(msg)
                    raise hlp.AURORANetworkError(msg)
                else:
                    return True
            else:
                return True
        else:
            msg = 'Only objects of type bus are allowed in *args.'
            logger.error(msg)
            raise TypeError(msg)

    def add_combines(self):
        pass

    def del_combines(self):
        pass

    def check_combines(self, b):
        pass

    def check_network(self):
        r"""
        Check if components are connected properly within the network.
        Initialize the topological constructure around the components.
        Add network objective to components、connections、buses、controls.
        Simplify the topological construction.
        """
        if len(self.conns) == 0:
            msg = (
                'No connections have been added to the network, please make '
                'sure to add your connections with the .add_conns() method.'
            )
            logger.error(msg)
            raise hlp.AURORANetworkError(msg)
        # system topological constructure management
        self.check_conns()  # check the multiple connect
        self.init_components()  # generate specifications: {comp_type: dataframe}
        self.check_components()  # check the outlet and inlet of comp
        self.add_network_()  # add net work objective for all components、connections、buses、equations
        self.generate_nodes_constructure_()
        self.classify_combine_system_()
        self.collect_start_components_()  # determine starts points
        self.combined_system_topological_constructure_simplify_()  # manage fluid-electric system
        # initialise the property_SI、specification dataframe of connections
        self.init_connections()
        # initialize nodes results of components
        self.init_nodes_results()
        # initialize unit of properties of components/connections
        self.init_unit_set()
        # network checked
        self.checked = True
        msg = '########----Network check successful----########'
        logger.info(msg)

    def check_conns(self):
        r"""
        Check connections for multiple usage of inlets or outlets.
        """
        # check outlets of components
        dub = self.conns.loc[self.conns.duplicated(["source", "source_id"])]
        for c in dub['object']:
            targets = []
            mask = (
                (self.conns["source"].values == c.source)
                & (self.conns["source_id"].values == c.source_id)
            )
            for conns in self.conns.loc[mask, "object"]:
                targets += [f"\"{conns.target.label}\" ({conns.target_id})"]
            targets = ", ".join(targets)

            msg = (
                f"The source \"{c.source.label}\" ({c.source_id}) is attached "
                f"to more than one component on the target side: {targets}. "
                "Please check your network configuration."
            )
            logger.error(msg)
            raise hlp.AURORANetworkError(msg)
        # check inlets of components
        dub = self.conns.loc[self.conns.duplicated(['target', 'target_id'])]
        for c in dub['object']:
            sources = []
            mask = (
                (self.conns["target"].values == c.target)
                & (self.conns["target_id"].values == c.target_id)
            )
            for conns in self.conns.loc[mask, "object"]:
                sources += [f"\"{conns.source.label}\" ({conns.source_id})"]
            sources = ", ".join(sources)
            msg = (
                f"The target \"{c.target.label}\" ({c.target_id}) is attached "
                f"to more than one component on the source side: {sources}. "
                "Please check your network configuration."
            )
            logger.error(msg)
            raise hlp.AURORANetworkError(msg)

    def init_components(self):
        r"""
        Initialize necessary component information.
        Generate dataframe constructure of components in results.
        """
        for comp in self.comps["object"]:
            # get incoming and outgoing connections of a component
            sources = self.conns[self.conns['source'] == comp]  # outlet connections
            sources = sources['source_id'].sort_values().index.tolist()  # sort due to source_id, get index converted to list
            targets = self.conns[self.conns['target'] == comp]  # inlet connections
            targets = targets['target_id'].sort_values().index.tolist()
            # save the incoming and outgoing as well as the number of
            # connections as component attribute
            comp.inl = self.conns.loc[targets, 'object'].tolist()  # add inlet connection as property to the comp
            comp.outl = self.conns.loc[sources, 'object'].tolist()  # add outlet connection as property to the comp
            comp.num_i = len(comp.inlets())
            comp.num_o = len(comp.outlets())
            # set up restults and specification dataframes
            comp_type = comp.__class__.__name__
            if comp_type not in self.results:
                dtypes = {}
                for key, data in comp.parameters.items():
                    if isinstance(data, dc_cp):
                        if data.dimension == 1:
                            dtypes[key] = 'float64'
                        else:
                            dtypes[key] = 'object'
                        dtypes[f'{key}_unit'] = str
                self.results[comp_type] = pd.DataFrame(columns=list(dtypes.keys())).astype(dtypes)
            if comp_type not in self.specifications:  # generate the comp_type item of specifications dict
                cols, groups, chars = [], [], []  # properties, group properties, char lines
                for col, data in comp.parameters.items():
                    if isinstance(data, dc_cp):
                        cols += [col]
                    elif isinstance(data, dc_gcp) or isinstance(data, dc_gcc):
                        groups += [col]
                    elif isinstance(data, dc_cc) or isinstance(data, dc_cm):
                        chars += [col]
                self.specifications[comp_type] = {
                    'groups': pd.DataFrame(columns=groups, dtype='bool'),
                    'chars': pd.DataFrame(columns=chars, dtype='object'),
                    'variables': pd.DataFrame(columns=cols, dtype='bool'),
                    'properties': pd.DataFrame(columns=cols, dtype='bool')
                }

    def check_components(self):
        """ count number of incoming and outgoing connections and compare to expected values
        """
        for comp in self.comps['object']:
            counts = (self.conns[['source', 'target']] == comp).sum()
            if counts["source"] != comp.num_o:
                msg = (
                    f"The component {comp.label} is missing "
                    f"{comp.num_o - counts['source']} outgoing connections. "
                    "Make sure all outlets are connected and all connections "
                    "have been added to the network."
                )
                logger.error(msg)
                # raise an error in case network check is unsuccessful
                raise hlp.AURORANetworkError(msg)
            elif counts["target"] != comp.num_i:
                msg = (
                    f"The component {comp.label} is missing "
                    f"{comp.num_i - counts['target']} incoming connections. "
                    "Make sure all inlets are connected and all connections "
                    "have been added to the network."
                )
                logger.error(msg)
                # raise an error in case network check is unsuccesful
                raise hlp.AURORANetworkError(msg)

    def add_network_(self):
        """
        Add network objective for all components、connections、buses、equations、controls.
        """
        for comp in self.comps['object']:
            comp.network = self
        for conn in self.conns['object']:
            conn.network = self
        for bus in self.busses.values():
            bus.network = self
        for ude in self.user_defined_eq.values():
            ude.network = self

    def generate_nodes_constructure_(self):
        mask = self.comps['object'].apply(lambda comp: comp.is_differential_component() and comp.nodes_num > 0)
        self.differ_comps = self.comps.loc[mask, :]
        for comp in self.differ_comps['object']:
            comp.generate_nodes_constructure()

    def classify_combine_system_(self):
        """
        Classify fluid-electric components and connections.
        """
        # fluid section
        fluid_comps_mask = self.comps["object"].apply(lambda c: 'fluid' in c.component_type())
        self.fluid_comps = self.comps.loc[fluid_comps_mask, :]
        fluid_conns_mask = self.conns["object"].apply(lambda c: c.connection_type() == 'fluid')
        self.fluid_conns = self.conns.loc[fluid_conns_mask, :]
        # electric section
        electric_comps_mask = self.comps["object"].apply(lambda c: 'electric' in c.component_type())
        self.electric_comps = self.comps.loc[electric_comps_mask, :]
        electric_conns_mask = self.conns["object"].apply(lambda c: c.connection_type() == 'electric')
        self.electric_conns = self.conns.loc[electric_conns_mask, :]

    def collect_start_components_(self):
        """
        Collect start components of fluid or electric system.
        """
        # fluid section
        self.fluid_components_number = len(self.fluid_comps["object"])
        mask_f = self.fluid_comps["object"].apply(lambda c: c.is_branch_source())  # chose the start components
        self.fluid_start_components = self.fluid_comps["object"].loc[mask_f]  # fluid_start_components: list of fluid start comp object of branches
        if self.fluid_components_number > 0 and len(self.fluid_start_components) == 0:
            msg = (
                "You cannot build a fluid system without at least one CycleCloser or "
                "a Source and Sink."
            )
            raise hlp.AURORANetworkError(msg)
        # electric section
        self.electric_components_number = len(self.electric_comps["object"])
        mask_e = self.electric_comps["object"].apply(lambda c: c.is_branch_source())
        self.electric_start_components = self.electric_comps["object"].loc[mask_e]
        if self.electric_components_number > 0 and len(self.electric_start_components) == 0:
            msg = f'You cannot build a electric system without at least one CycleCloser or a Source and Sink.'
            raise hlp.AURORANetworkError(msg)
        # combines section between fluid and electric section
        if self.fluid_components_number > 0 and self.electric_components_number > 0:
            if self.combs.empty:
                msg = f'No combines be generated to combine fluid and electric systems.'
                raise hlp.AURORANetworkError(msg)

    def combined_system_topological_constructure_simplify_(self):
        """
        Simplify combined fluid-electric system topological construction.
        """
        self.branches_components = []
        if self.fluid_components_number > 0:
            self.create_massflow_and_fluid_branches_()  # multi_edge branches, plot constructions, non_edge various fluid branches
            self.create_fluid_wrapper_branches_()  #
            ##### simplify topological constructure
            # fluid.val, mixing_rule, engine, back_end, wrapper
            self.propagate_fluid_wrappers_()
            # set the shared massflow property on each massflow branch
            self.presolve_massflow_topology_()
            self.get_mass_flow_range()
            # set fluid composition J_col
            self.presolve_fluid_topology_()  # set the shared fluid property(val, is_set, is_var, J_col) on each fluid branch
            self.presolve_pressure_enthalpy_mass_topology_()
            msg = f'########----Simplify Fluid System Successful----########'
            logger.info(msg)
        if self.electric_components_number > 0:
            self.create_electricity_branches_()
            self.create_frequency_branches_()
            self.presolve_electricity_branches_()
            self.presolve_frequency_branches_()
            self.presolve_voltage_topology_()
            msg = f'########----Simplify Electric System Successful----########'
            logger.info(msg)
        # simplify the topological constructure of combined properties between various system
        self.presolve_combined_property_topology_()
        # simplify nodes constructure topology
        self.presolve_nodes_topology_()

    def create_massflow_and_fluid_branches_(self):  # generate mass flow branches
        """
        Create main branches of fluid system.
        """
        self.branches_fluid_system = {}
        for start in self.fluid_start_components:
            self.branches_fluid_system.update(start.start_branch())  # collect all branches, using iteration to get branch
        self.massflow_branches = hlp.get_all_subdictionaries(self.branches_fluid_system)  # plot construction list of all mass flow branch dicts
        self.fluid_branches = {}  # fluid property branches, simplify the iteration
        for branch_name, branch_data in self.branches_fluid_system.items():  # branch_data: branch dict
            subbranches = hlp.get_all_subdictionaries(branch_data["subbranches"])
            main = {k: v for k, v in branch_data.items() if k != "subbranches"}
            self.fluid_branches[branch_name] = [main] + subbranches  # {branch_name: dict_list}, dict_list: [{components:[], connections: []}]

    def create_fluid_wrapper_branches_(self):
        """
        Create fluid wrapper branches of fluid system.
        """
        self.fluid_wrapper_branches = {}
        mask = self.fluid_comps["object"].apply(lambda c: c.is_wrapper_branch_source())  # ["Source", "CycleCloser", "WaterElectrolyzer", "FuelCell"]
        start_components = self.fluid_comps["object"].loc[mask]  # mask: Boolean Series
        for start in start_components:
            self.fluid_wrapper_branches.update(start.start_fluid_wrapper_branch())
        merged = self.fluid_wrapper_branches.copy()
        for branch_name, branch_data in self.fluid_wrapper_branches.items():
            if branch_name not in merged:
                continue
            for ob_name, ob_data in self.fluid_wrapper_branches.items():
                if ob_name != branch_name:
                    common_connections = list(
                        set(branch_data["connections"])
                        & set(ob_data["connections"])
                    )
                    if len(common_connections) > 0 and ob_name in merged:
                        # delete duplicate conns and comps
                        # merge the connected branches
                        merged[branch_name]["connections"] = list(
                            set(branch_data["connections"] + ob_data["connections"])
                        )
                        merged[branch_name]["components"] = list(
                            set(branch_data["components"] + ob_data["components"])
                        )
                        del merged[ob_name]
                        break
        # corrected wrapper branches
        self.fluid_wrapper_branches = merged

    def propagate_fluid_wrappers_(self):
        """
        Allocate fluid wrapper of fluid connections connected together.
        """
        connections_in_wrapper_branches = []
        for branch_data in self.fluid_wrapper_branches.values():
            all_connections = [c for c in branch_data["connections"]]
            any_fluids_set = []
            engines = {}
            back_ends = {}
            for c in all_connections:
                for f in c.fluid.is_set:  # is_set: set, fluid composition, no mass fraction
                    any_fluids_set += [f]
                    if f in c.fluid.engine:
                        engines[f] = c.fluid.engine[f]
                    if f in c.fluid.back_end:
                        back_ends[f] = c.fluid.back_end[f]
            # mixing rule for multiple fluid composition
            mixing_rules = [
                c.mixing_rule for c in all_connections
                if c.mixing_rule is not None
            ]
            mixing_rule = set(mixing_rules)
            if len(mixing_rule) > 1:
                msg = "You have provided more than one mixing rule."
                raise hlp.AURORANetworkError(msg)
            elif len(mixing_rule) == 0:  # default mixing rule
                mixing_rule = set(["ideal-cond"])  # default mixing rule for mixture fluid
            if not any_fluids_set:
                msg = "You are missing fluid specifications."
            any_fluids = [f for c in all_connections for f in c.fluid.val]
            any_fluids0 = [f for c in all_connections for f in c.fluid.val0]  # val0 !!!!
            potential_fluids = set(any_fluids_set + any_fluids + any_fluids0)
            num_potential_fluids = len(potential_fluids)
            if num_potential_fluids == 0:
                msg = (
                    "The follwing connections of your network are missing any "
                    "kind of fluid composition information:"
                    + ", ".join([c.label for c in all_connections]) + "."
                )
                raise hlp.AURORANetworkError(msg)
            # set fluid composition for all branches
            for c in all_connections:
                c.mixing_rule = list(mixing_rule)[0]
                c._potential_fluids = potential_fluids  # set !!!!
                if num_potential_fluids == 1:
                    f = list(potential_fluids)[0]
                    c.fluid.val[f] = 1
                    # c.fluid.is_set = {f}
                    # c.fluid.is_var = set()
                else:  # set initial fluid val
                    for f in potential_fluids:
                        if (f not in c.fluid.is_set and f not in c.fluid.val and f not in c.fluid.val0):
                            c.fluid.val[f] = 1 / len(potential_fluids)
                        elif f not in c.fluid.is_set and f not in c.fluid.val and f in c.fluid.val0:
                            c.fluid.val[f] = c.fluid.val0[f]
                #
                for f, engine in engines.items():
                    c.fluid.engine[f] = engine
                for f, back_end in back_ends.items():
                    c.fluid.back_end[f] = back_end
                # generate fluid wrapper for all conns
                c._create_fluid_wrapper()
            connections_in_wrapper_branches += all_connections
        missing_wrappers = (
                set(self.fluid_conns["object"].tolist())
                - set(connections_in_wrapper_branches)
        )
        if len(missing_wrappers) > 0:
            msg = (
                f"The fluid information propagation for the connections "
                f"{', '.join([c.label for c in missing_wrappers])} failed. "
                "The reason for this is likely, that these connections do not "
                "have any Sources or a CycleCloser attached to them."
            )
            logger.error(msg)
            raise hlp.AURORANetworkError(msg)

    def presolve_massflow_topology_(self):
        """
        Simplify the topological constructure of mass flow objective.
        """
        # mass flow is a single variable in each sub branch
        # fluid composition is a single variable in each main branch
        for branch in self.massflow_branches:  # massflow_branches: plot construction list of all branch dict
            num_massflow_specs = 0
            # statistical analysis massflow set
            for c in branch["connections"]:
                # number of specifications cannot exceed 1
                num_massflow_specs += c.m.is_set  #
                if c.m.is_set:
                    main_conn = c
                # self reference is not allowed
                if c.m_ref.is_set:
                    if c.m_ref.ref.obj in branch["connections"]:
                        msg = (
                            "You cannot reference a mass flow in the same "
                            f"linear branch. The connection {c.label} "
                            "references the connection "
                            f"{c.m_ref.ref.obj.label}."
                        )
                        raise hlp.AURORANetworkError(msg)
            # sharing massflow objective
            if num_massflow_specs == 1:
                # set every mass flow in branch to the specified value
                for c in branch["connections"]:
                    # map all connection's mass flow data containers to first
                    # branch element
                    c._m_tmp = c.m  # save origin m property tentatively
                    c.m = main_conn.m
                    if c not in c.m.shared_connection:
                        c.m.shared_connection.append(c)
                branch_self = [conn.label for conn in branch["connections"]]
                msg = (
                    "Removing "
                    f"{len(branch['connections']) - num_massflow_specs} "
                    f"mass flow variables from {branch_self} from system variables."
                )
                logger.debug(msg)
            elif num_massflow_specs > 1:  # erupted solve will lead to it
                msg = (
                    "You cannot specify two or more values for mass flow in "
                    "the same linear branch (starting at "
                    f"{branch['components'][0].label} and ending at "
                    f"{branch['components'][-1].label})."
                )
                raise hlp.AURORANetworkError(msg)
            else:  # no mass flow objective been set
                main_conn = branch["connections"][0]
                for c in branch["connections"][1:]:
                    # map all connection's mass flow data containers to first
                    # branch element
                    c._m_tmp = c.m
                    c.m = main_conn.m
                    if c not in c.m.shared_connection:
                        c.m.shared_connection.append(c)

    def get_mass_flow_range(self):
        for branch_data in self.fluid_wrapper_branches.values():
            all_connections = [c for c in branch_data["connections"]]
            mass_obj_contained = []
            mass_flow_contained0 = []
            for c in all_connections:
                if c.m.is_set and c.m not in mass_obj_contained:
                    mass_obj_contained.append(c.m)
                    mass_flow_contained0.append(c.m.val)
            mass_flow_contained0.extend([1, 10])  # !!!!
            branch_data["massflow"] = set(mass_flow_contained0)

    def presolve_fluid_topology_(self):
        """
        Simplify the topological constructure of fluid composition objective.
        """
        # set fluid objective due to branches constructure
        for branch_name, branch in self.fluid_branches.items():  # branch: plot construction list of all branch dicts
            all_connections = [c for b in branch for c in b["connections"]]  # all connections connected together of branch groups
            main_conn = all_connections[0]
            fluid_specs = [f for c in all_connections for f in c.fluid.is_set]
            # not fluid is set in the branch ################################
            if len(fluid_specs) == 0:
                main_conn._fluid_tmp = dc_flu()
                main_conn._fluid_tmp.val = main_conn.fluid.val.copy()
                main_conn._fluid_tmp.is_set = main_conn.fluid.is_set.copy()
                main_conn._fluid_tmp.is_var = main_conn.fluid.is_var.copy()
                main_conn._fluid_tmp.wrapper = main_conn.fluid.wrapper.copy()
                main_conn._fluid_tmp.engine = main_conn.fluid.engine.copy()
                main_conn._fluid_tmp.back_end = main_conn.fluid.back_end.copy()
                # share fluid objective
                for c in all_connections[1:]:  # main_conn is not included
                    c._fluid_tmp = c.fluid
                    c.fluid = main_conn.fluid
                # multipal fluids, all are vars
                if len(main_conn._potential_fluids) > 1:
                    main_conn.fluid.is_var = {f for f in main_conn.fluid.val}  #
                else:
                    main_conn.fluid.val[list(main_conn._potential_fluids)[0]] = 1  # single fluid, fluid.is_var is none
                    main_conn.fluid.is_set = {list(main_conn._potential_fluids)[0]}
                    main_conn.fluid.is_var = set()
            #
            elif len(fluid_specs) != len(set(fluid_specs)):
                single_branch = [c.label for c in all_connections]
                msg = (
                    f"The mass fraction:{fluid_specs} of a single fluid branch:{single_branch} cannot be specified "
                    "twice within a branch."
                )
                raise hlp.AURORANetworkError(msg)
            # some fluid is set ####################################
            else:
                fixed_fractions = {
                    f: c.fluid.val[f]  # fluid composition set mass fraction dict
                    for c in all_connections
                    for f in fluid_specs
                    if f in c.fluid.is_set
                }
                mass_fraction_sum = sum(fixed_fractions.values())
                if mass_fraction_sum > 1 + ERR:
                    msg = "Total mass fractions within a branch cannot exceed 1"
                    raise ValueError(msg)
                elif mass_fraction_sum < 1 - ERR:
                    # set the fluids with specified mass fraction
                    # remaining fluids are variable, create wrappers for them
                    all_fluids = main_conn.fluid.val.keys()
                    num_remaining_fluids = len(all_fluids) - len(fixed_fractions)
                    if num_remaining_fluids == 1:  # if various composition is one, calculate immediately
                        missing_fluid = list(
                            main_conn.fluid.val.keys() - fixed_fractions.keys()
                        )[0]
                        fixed_fractions[missing_fluid] = 1 - mass_fraction_sum
                        variable = set()
                    else:
                        missing_fluids = (
                            main_conn.fluid.val.keys() - fixed_fractions.keys()
                        )
                        variable = {f for f in missing_fluids}
                # sum of fluids set is enough: = 1
                else:
                    # fluid mass fraction is 100 %, all other fluids are 0 %
                    all_fluids = main_conn.fluid.val.keys()
                    remaining_fluids = (
                        main_conn.fluid.val.keys() - fixed_fractions.keys()
                    )
                    for f in remaining_fluids:
                        fixed_fractions[f] = 0
                    variable = set()
                # normalize all fluid objective of single branc
                main_conn._fluid_tmp = dc_flu()
                main_conn._fluid_tmp.val = main_conn.fluid.val.copy()
                main_conn._fluid_tmp.is_set = main_conn.fluid.is_set.copy()
                main_conn._fluid_tmp.is_var = main_conn.fluid.is_var.copy()
                main_conn._fluid_tmp.wrapper = main_conn.fluid.wrapper.copy()
                main_conn._fluid_tmp.engine = main_conn.fluid.engine.copy()
                main_conn._fluid_tmp.back_end = main_conn.fluid.back_end.copy()
                # sharing fluid objective
                for c in all_connections[1:]:
                    c._fluid_tmp = c.fluid
                    c.fluid = main_conn.fluid
                # set shared fluid objective
                main_conn.fluid.val.update(fixed_fractions)
                main_conn.fluid.is_set = {f for f in fixed_fractions}
                main_conn.fluid.is_var = variable
                # this seems to be a problem in some cases, e.g. the basic
                # gas turbine tutorial !!!!!!!
                # num_var = len(variable)
                # for f in variable:
                #     main_conn.fluid.val[f]: (1 - mass_fraction_sum) / num_var  # set the original value for iteration ?
            #
            set_fluid_dict = {f: main_conn.fluid.val[f] for f in main_conn.fluid.is_set}
            all_set_fraction = sum(set_fluid_dict.values())
            initial_fraction_list = []
            for f in main_conn.fluid.val0:
                if f not in main_conn.fluid.is_set and f in main_conn.fluid.is_var:
                    main_conn.fluid.val[f] = main_conn.fluid.val0[f]
                    initial_fraction_list += [f]
            initial_fluid_dict = {f: main_conn.fluid.val[f] for f in initial_fraction_list}
            initial_set_fraction = sum(initial_fluid_dict.values())
            # if set(initial_fraction_list) == set(main_conn.fluid.is_var) and all_set_fraction + initial_set_fraction != 1:
            #     msg = f'Total mass fraction within a branch:{main_conn.label} cannot exceed 1'
            #     raise hlp.AURORANetworkError(msg)
            for f in main_conn.fluid.is_var:
                if f not in initial_fraction_list:
                    main_conn.fluid.val[f] = ((1 - all_set_fraction - initial_set_fraction) /
                                              max(len(set(main_conn.fluid.is_var) - set(initial_fraction_list)), 1))
            # generate fluid data of connections
            [c.build_fluid_data() for c in all_connections]

    def presolve_pressure_enthalpy_mass_topology_(self):
        """
        Combine the pressure、enthalpy、mass flow objective of connections.
        The components located in critical nodes will be contained at self.branches_components
        to prevent duplicate topological construction simplification.

        Returns
        -------

        """
        mask = self.fluid_comps["object"].apply(lambda c: c.is_simplify_topology_start())  # ["Source", "CycleCloser", "WaterElectrolyzer", "FuelCell"]
        start_components = self.fluid_comps["object"].loc[mask]  # mask: Boolean Series
        for start in start_components:
            start.simplify_pressure_enthalpy_mass_topology_start()

    def create_electricity_branches_(self):
        """
        Create electricity branches.
        """
        self.branches_electric_system = {}
        for start in self.electric_start_components:
            self.branches_electric_system.update(start.start_branch())

    def create_frequency_branches_(self):
        """
        Create electric frequency branches.
        """
        pass

    def presolve_electricity_branches_(self):
        pass

    def presolve_frequency_branches_(self):
        pass

    def presolve_voltage_topology_(self):
        mask = self.electric_comps["object"].apply(lambda c: c.is_simplify_topology_start())
        start_components = self.electric_comps["object"].loc[mask]  # mask: Boolean Series
        for start in start_components:
            start.simplify_voltage_topology_start()

    def presolve_combined_property_topology_(self):
        for comb in self.combs['object']:
            pass

    def presolve_nodes_topology_(self):
        for comp in self.differ_comps['object']:
            comp.simplify_nodes_topology()

    def init_connections(self):
        """
        Specification of SI values for user set values.
        Generate dataframe constructure of connections in results、specifications.
        """
        self.all_fluids = []
        # fluid property values
        for c in self.fluid_conns['object']:
            self.all_fluids += c.fluid.val.keys()
            if not self.init_previous:  # Initialise the calculation with values from the previous calculation
                c.good_starting_values = False  # exists in add_conns, init_properties, solve module
        if len(self.all_fluids) == 0 and len(self.fluid_conns['object']) > 0:
            msg = (
                'Network has no fluids, please specify a list with fluids on '
                'network creation.'
            )
            logger.error(msg)
            raise hlp.AURORANetworkError(msg)
        self.all_fluids = set(self.all_fluids)
        # set up results dataframe for connections
        # this should be done based on the connections
        # properties information of connection type for results
        self.results_properties_connections_summar = {
            'fluid': dict(itertools.chain(
                dict([(item[0], item[1]) for key in fpd.keys() for item in [(key, 'float64'), (f'{key}_unit', str)]]).items(),
                        {'phase': str}.items(),
                        dict([(item[0], item[1]) for fluid in self.all_fluids for item in [(fluid, str)]]).items(), )),
            'electric': {k: v for key in epd.keys() for k, v in [(key, 'object'), (f'{key}_unit', str)]},
            # 'magnetic': [],
            # 'mechanical': []
        }
        # working media properties connections
        for system_type_ in self.results_properties_connections_summar.keys():
            conns_dataframe = self.get_attr(f'{system_type_}_conns')
            if not conns_dataframe.empty:
                conn_class_type = conns_dataframe['object'].iat[0].__class__.__name__
                results_properties_dict = self.results_properties_connections_summar[system_type_]
                if conn_class_type not in self.results:
                    self.results[conn_class_type] = pd.DataFrame(columns=list(results_properties_dict.keys())).astype(results_properties_dict)
                if conn_class_type not in self.specifications:
                    self.specifications[conn_class_type] = pd.DataFrame(columns=list(results_properties_dict.keys()), dtype=bool)
                if f'{conn_class_type}_Ref' not in self.specifications:
                    cols = [key for key in conns_dataframe['object'].iat[0].property_data.keys() if '_ref' in key]
                    self.specifications[f'{conn_class_type}_Ref'] = pd.DataFrame(columns=list(cols), dtype=bool)
        msg = (
            "Updated fluid property SI values and fluid mass fraction for user "
            "specified connection parameters."
        )
        logger.debug(msg)

    def init_nodes_results(self):
        self.nodes_results = {}
        for comp in self.differ_comps['object']:
            self.nodes_results[f'{comp.__class__.__name__}: {comp.label}'] = pd.DataFrame(columns=list(range(comp.nodes_num)))

    def init_unit_set(self):
        """
        Initialize the unit set of properties.
        """
        # properties of connections
        for c in self.conns['object']:
            for key in c.property_data:
                data = c.get_attr(key)
                if hasattr(data, 'property_data'):
                    if data.unit:
                        pass
                    elif f'{key}_unit' in self.properties_unit_citation:
                        data.unit = self.properties_unit_citation[f'{key}_unit']
                    elif f'{data.property_data["equal"]}_unit' in self.properties_unit_citation:
                        data.unit = self.properties_unit_citation[f'{data.property_data["equal"]}_unit']
                    else:
                        data.unit = data.SI_unit
        # properties of components
        for comp in self.comps['object']:
            comp_class = comp.__class__.__name__
            for key, val in comp.parameters.items():  # component property unit initialization
                data = comp.get_attr(key)
                if isinstance(data, dc_cp) and hasattr(data, 'property_data'):
                    if data.unit:
                        pass
                    elif f'{key}_unit' in self.properties_unit_citation:
                        data.unit = self.properties_unit_citation[f'{key}_unit']
                    elif f'{data.property_data["equal"]}_unit' in self.properties_unit_citation:
                        data.unit = self.properties_unit_citation[f'{data.property_data["equal"]}_unit']
                    else:
                        data.unit = data.SI_unit
        # bus
        for b in self.busses.values():
            # sum power property
            if b.P.unit:
                pass
            elif 'P_unit' in self.properties_unit_citation:
                b.P.unit = self.properties_unit_citation['P_unit']
            elif f'{b.P.property_data["equal"]}_unit' in self.properties_unit_citation:
                b.P.unit = self.properties_unit_citation[f'{b.P.property_data["equal"]}_unit']
            else:
                b.P.unit = b.P.SI_unit
        # combines
        for comb in self.combs['object']:
            pass
        # units of nodes
        for comp in self.differ_comps['object']:
            comp.init_nodes_properties_units_()

    def reset_topology_reduction_specifications(self):
        """
        Topological constructure reversion.   need to be rewritten later!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        for c in self.fluid_conns["object"]:
            if hasattr(c, "_m_tmp"):
                value = c.m.val_SI
                unit = c.m.unit
                c.m = c._m_tmp
                c.m.val_SI = value
                c.m.unit = unit
                del c._m_tmp
            if hasattr(c, "_fluid_tmp"):
                val = c.fluid.val
                c.fluid = c._fluid_tmp
                c.fluid.val = val
                del c._fluid_tmp
        for branches_component in self.branches_components:
            for conn in branches_component.inl + branches_component.outl:
                if hasattr(conn, "_p_tmp"):
                    value = conn.p.val_SI
                    unit = conn.p.unit
                    conn.p = conn._p_tmp
                    conn.p.val_SI = value
                    conn.p.unit = unit
                    del conn._p_tmp
                if hasattr(conn, "_h_tmp"):
                    value = conn.h.val_SI
                    unit = conn.h.unit
                    conn.h = conn._h_tmp
                    conn.h.val_SI = value
                    conn.h.unit = unit
                    del conn._h_tmp

    def initialise(self):
        r"""
        Initilialise the network depending on calclation mode.

        Design

        - Generic fluid composition and fluid property initialisation.
        - Starting values from initialisation path if provided.

        Offdesign

        - Check offdesign path specification.
        - Set component and connection design point properties.
        - Switch from design/offdesign parameter specification.
        """
        # keep track of the number of bus, component and connection equations
        # as well as number of component variables
        self.num_bus_eqs = 0  # sum of P set in all buses
        self.num_comp_eqs = 0  # number of all equations including constrains and parameter equation chosen
        self.num_conn_eqs = 0  # sum of no_primary to primary property equations set in all connections
        self.num_eqs = 0
        self.num_vars = 0
        self.num_comp_vars = 0  # index of the parameters chosen of components
        self.num_conn_vars = 0  # index of the primary variables of connections
        # fluid variables section
        self.num_conn_mass_flow_vars = 0  # mass flow variable
        self.num_conn_pressure_vars = 0  # pressure variable
        self.num_conn_enthalpy_vars = 0  # entalpy variable
        # electric variables section
        self.num_conn_voltage_vars = 0  # electric voltage variable
        self.num_conn_electricity_vars = 0
        self.num_conn_frequency_vars = 0  # electric frequency variable
        # fluid groups
        self.num_fluid_vars = 0
        self.num_fluid_eqs = 0

        # switch on/off design/offdesign properties
        if self.mode == 'offdesign':
            self.redesign = True  # delicate: the mode has been switched
            if self.design_path is None:
                # must provide design_path
                msg = "Please provide a design_path for offdesign mode."
                logger.error(msg)
                raise hlp.AURORANetworkError(msg)
            # load design case
            if self.new_design:  # the reference mode is another design mode of network
                # set design property values
                self.init_offdesign_params()
            # switch on offdesign properties, switch off offdesign properties
            self.init_offdesign()
        else:
            # reset any preceding offdesign calculation
            self.init_design()  # network design mode initialization

        ##### presolve variables values #####
        # convert set properties value to standard val_SI
        self.convert_set_property_criterion_()
        if self.fluid_components_number > 0:
            # presolve properties of node due to state of nodes
            self.decrease_fluid_system_variables_dimension_()
            # summarize connections connected by pressure reference objective
            self.list_pressure_reference_nodes_()
            # initialize pressure value for connections
            self.spread_pressure_values_()
            # init pressure val0 through spreading along branches
            self.init_fluid_properties_diffusively_()  # !!!!
        if self.electric_components_number > 0:
            self.init_electric_properties_diffusively_()
        # convert properties value to standard val_SI
        self.convert_variables_criterion_()
        # pack connections properties
        self.pack_connections_properties_()
        # property initialisation from json file
        self.init_properties_from_file_()  # count no_primary parameters set in all connections and set initial value of primary properties
        # initialize the unset properties without json file
        self.init_properties_logically()  # !!!!
        # initialize the reaction parameters of combustion components
        self.init_reaction_parameters()
        # generate bounds of variables, using for presolving equations
        self.initialize_variables_bounds()
        # generate node equation system to contain functions unsolved after calculating isolated equations in advance
        self.generate_node_equation_system()  #
        msg = '########----Network initialised----########'
        logger.info(msg)

    def init_design(self):
        r"""
        Initialise a design calculation.

        Offdesign parameters are unset, design parameters are set. If
        :code:`local_offdesign` is :code:`True` for connections or components,
        the design point information are read from the .csv-files in the
        respective :code:`design_path`. In this case, the design values are
        unset, the offdesign values set.
        """
        _local_designs = {}  # {conn.design_path: dataframe}
        ###### connections ######
        for c in self.conns['object']:
            # read design point information of connections with
            # local_offdesign activated from their respective design path
            if c.local_offdesign:  # the mode of single connection
                path = c.design_path
                if path is None:
                    msg = (
                        "The parameter local_offdesign is True for the "
                        f"connection {c.label}, an individual design_path must "
                        "be specified in this case!"
                    )
                    logger.error(msg)
                    raise hlp.AURORANetworkError(msg)
                # unset design parameters
                for var in c.design:
                    c.get_attr(var).is_set = False
                # set offdesign parameters
                for var in c.offdesign:
                    c.get_attr(var).is_set = True
                # read design point information
                msg = (
                    "Reading individual design point information for "
                    f"connection {c.label} from {c.design_path}."
                )
                logger.debug(msg)
                if path not in _local_designs:
                    _local_designs[path] = self._load_network_state(path)
                df = _local_designs[c.design_path][c.__class__.__name__]  # df: dataframe, the design value of conns
                # write data to connections
                self.init_conn_design_params(c, df)  # set the design value for each conn based on design unit in file
            else:  # conn not local_offdesign
                # unset all design values
                # design value is not necessary for design mode
                self.unset_conn_design_values_(c)
                c.new_design = True  # the another design mode of single connection
                # switch connections to design mode
                for var in c.design:
                    c.get_attr(var).is_set = True
                for var in c.offdesign:
                    c.get_attr(var).is_set = False
        ###############################
        # unset design values for busses, count bus equations and
        # reindex bus dictionary
        for b in self.busses.values():
            b.comps['P_ref'] = np.nan
        ###############################
        series = pd.Series(dtype='float64')
        for cp in self.comps['object']:
            cpcl = cp.__class__.__name__
            # read design point information of components with
            # local_offdesign activated from their respective design path
            if cp.local_offdesign:
                path = cp.design_path
                if path is None:
                    msg = (
                        "The parameter local_offdesign is True for the "
                        f"component {cp.label}, an individual design_path must "
                        "be specified in this case!"
                    )
                    logger.error(msg)
                    raise hlp.AURORANetworkError(msg)
                if path not in _local_designs:
                    _local_designs[path] = self._load_network_state(path)
                dfs = _local_designs[path][cpcl]
                # write data
                self.init_comp_design_params(cp, dfs)  # set design value of property_data of comp
                # unset design parameters
                for var in cp.design:
                    cp.get_attr(var).is_set = False
                # set offdesign parameters
                switched = False
                msg = 'Set component attributes '
                for var in cp.offdesign:
                    # set variables provided in .offdesign attribute
                    data = cp.get_attr(var)
                    data.is_set = True
                    # take nominal values from design point
                    # the design point as the original iterated value of offdesign variables
                    if isinstance(data, dc_cp):
                        # the offdesign parameter is constant property
                        data.val = hlp.convert_from_SI(data.property_data, data.design, data.unit)
                        switched = True
                        msg += var + ", "
                if switched:
                    msg = f"{msg[:-2]} to design value at component {cp.label}."
                    logger.debug(msg)
                cp.new_design = False
            else:  # comp not local_offdesign
                # switch connections to design mode
                for var in cp.design:
                    cp.get_attr(var).is_set = True
                for var in cp.offdesign:
                    cp.get_attr(var).is_set = False
                cp.new_design = True  # reference another design mode
                for key, dc in cp.parameters.items():
                    if isinstance(dc, dc_cp):
                        # offdesign mode of the component
                        if (
                                (self.mode == 'offdesign' and not cp.local_design) or
                                (self.mode == 'design' and cp.local_offdesign) and
                                (series[key] is not None)
                        ):
                            cp.get_attr(key).design = float(series[key])
                        # design mode of the component
                        else:
                            if cp.get_attr(key).dimension == 1:
                                cp.get_attr(key).design = np.nan  # set the design value of comp property is nan
                            else:
                                cp.get_attr(key).design = []

    def init_offdesign_params(self):
        r"""
        Read design point information from specified :code:`design_path`.

        If a :code:`design_path` has been specified individually for components
        or connections, the data will be read from the specified individual
        path instead.

        Note
        ----
        The methods
        :py:meth:`Aurora.networks.network.Network.init_comp_design_params`
        (components) and the
        :py:meth:`Aurora.networks.network.Network.init_conn_design_params`
        (connections) handle the parameter specification.
        """
        # components with offdesign parameters
        components_with_parameters = [
            cp.label for cp in self.comps["object"] if len(cp.parameters) > 0
        ]
        # fetch all components, reindex with label
        df_comps = self.comps.loc[components_with_parameters].copy()  # dataframe of components
        # iter through unique types of components (class names)
        dfs = self._load_network_state(self.design_path)  # dicts of dataframe from design path
        # iter through all components of this type and set data
        ind_designs = {}
        ###### components design set ######
        for label, row in df_comps.iterrows():
            df = dfs[row["comp_type"]]  # properties dataframe of component class
            comp = row["object"]  # component objective
            path = comp.design_path
            # read data of components with individual design_path
            if path is not None:
                if path not in ind_designs:
                    ind_designs[path] = self._load_network_state(path)
                data = ind_designs[path][row["comp_type"]]
            else:
                data = df
            # write data to components
            self.init_comp_design_params(comp, data)
        msg = 'Done reading design point information for components.'
        logger.debug(msg)
        ###### buses design set ######
        if len(self.busses) > 0:
            for b, bus in self.busses.items():  # b: bus.label
                # the bus design data are stored in dfs[b][0] (column is not named)
                if len(bus.comps) > 0:
                    bus.comps.loc[self.get_comp(dfs[b].index), "P_ref"] = [hlp.convert_to_SI(bus.P.property_data,
                                                                                             dfs[b]['component value'].loc[cp],
                                                                                             dfs[b]['unit'].loc(cp))
                                                                           for cp in dfs[b].index]
        ###### connections design set ######
        # read connection design point information
        # iter through connections
        for c in self.conns['object']:
            # read data of connections with individual design_path
            df = dfs[c.__class__.__name__]
            path = c.design_path
            if path is not None:
                if path not in ind_designs:
                    ind_designs[path] = self._load_network_state(path)
                data = ind_designs[path][c.__class__.__name__]
            else:
                data = df
            self.init_conn_design_params(c, data)
        msg = 'Done reading design point information for connections.'
        logger.debug(msg)

    def init_comp_design_params(self, cp, df):
        r"""
        Write design point information to components.

        Parameters
        ----------
        cp : Aurora.components.component.Component
            Write design point information to this component.
        df : pandas.core.series.Series, pandas.core.frame.DataFrame
            Design point information.
        """
        if cp.label not in df.index:
            # no matches in the connections of the network and the design files
            msg = (
                f"Could not find component '{cp.label}' in design case file. "
                "This is is critical only to components, which need to load "
                "design values from this case."
            )
            logger.debug(msg)
            return
        # write component design data
        data = df.loc[cp.label]
        # set design value
        if self.mode == 'design' or cp.local_design:
            # rewrite all design information to prevent remaining of other design information before
            cp.new_design = True  # reference another design mode
        for key, dc in cp.parameters.items():
            if isinstance(dc, dc_cp):
                # offdesign mode of the component
                if (
                        (self.mode == 'offdesign' and not cp.local_design) or
                        (self.mode == 'design' and cp.local_offdesign)
                ):
                    valid_value = True
                    if str(data[f'{key}_unit']).lower() == 'nan':
                        valid_value = False
                    if valid_value:
                        cp.get_attr(key).design = hlp.convert_to_SI(cp.get_attr(key).property_data, data[key], str(data[f'{key}_unit']))
                    else:
                        cp.get_attr(key).design = np.nan
                # design mode of the component
                else:
                    if cp.get_attr(key).dimension == 1:
                        cp.get_attr(key).design = np.nan  # set the design value of comp property is nan
                    else:
                        cp.get_attr(key).design = []

    def init_conn_design_params(self, c, df):
        r"""
        Write design point information to connections.

        Parameters
        ----------
        c : Aurora.connections.connection.Connection
            Write design point information to this connection.
        df : pandas.core.frame.DataFrame
            Dataframe containing design point information.
        """
        # match connection (source, source_id, target, target_id) on
        # connection objects of design file
        if c.label not in df.index:
            # no matches in the connections of the network and the design files
            msg = (
                f"Could not find connection '{c.label}' in design case. Please "
                "make sure no connections have been modified or components "
                "have been relabeled for your offdesign calculation."
            )
            logger.exception(msg)
            raise hlp.AURORANetworkError(msg)
        #
        conn = df.loc[c.label]
        for var in self.connections_properties_data[c.connection_type()]:
            c.get_attr(var).design = hlp.convert_to_SI(
                c.get_attr(var).property_data, conn[var], conn[f"{var}_unit"]
            )
        if c.connection_type() == 'fluid':
            if c.m.design != 0.0:
                c.vol.design = c.v.design / c.m.design
            else:
                c.vol.design = math.inf
            for fluid in c.fluid.val:
                c.fluid.design[fluid] = float(conn[fluid])

    def unset_conn_design_values_(self, c):
        for var in self.variables_properties_connections_summar[c.connection_type()]:
            data = c.get_attr(var)
            if data.dimension == 1:
                data.design = np.nan
            else:
                data.design = []
        if c.connection_type() == 'fluid':
            c.fluid.design = {}

    def init_offdesign(self):
        r"""
        Switch components and connections from design to offdesign mode.

        Note
        ----
        **components**

        All parameters stated in the component's attribute :code:`cp.design`
        will be unset and all parameters stated in the component's attribute
        :code:`cp.offdesign` will be set instead.

        Additionally, all component parameters specified as variables are
        unset and the values from design point are set.

        **connections**

        All parameters given in the connection's attribute :code:`c.design`
        will be unset and all parameters stated in the connections's attribute
        :code:`cp.offdesign` will be set instead. This does also affect
        referenced values!
        """
        ###### connections ######
        for c in self.conns['object']:
            if not c.local_design:
                # switch connections to offdesign mode
                for var in c.design:
                    param = c.get_attr(var)
                    param.is_set = False
                    param.is_var = True
                    if f"{var}_ref" in c.property_data:  #
                        c.get_attr(f"{var}_ref").is_set = False
                for var in c.offdesign:
                    param = c.get_attr(var)
                    param.is_set = True
                    param.is_var = False
                c.new_design = False
        msg = 'Switched connections from design to offdesign.'
        logger.debug(msg)
        ###### components ######
        for cp in self.comps['object']:
            if not cp.local_design:
                # unset variables provided in .design attribute
                for var in cp.design:
                    cp.get_attr(var).is_set = False
                switched = False
                msg = 'Set component attributes '
                for var in cp.offdesign:
                    # set variables provided in .offdesign attribute
                    data = cp.get_attr(var)
                    data.is_set = True
                    # take nominal values from design point
                    if isinstance(data, dc_cp):
                        # like zeta ...
                        data.val = hlp.convert_from_SI(data.property_data, data.design, data.unit)
                        switched = True
                        msg += var + ', '
                if switched:
                    msg = f"{msg[:-2]} to design value at component {cp.label}."
                    logger.debug(msg)
            cp.new_design = False
        msg = 'Switched components from design to offdesign.'
        logger.debug(msg)

    def convert_set_property_criterion_(self):
        """
        Convert properties set in connections to criteria values.
        """
        for c in self.conns['object']:
            # properties has set in connections
            for key in c.property_data:
                prop = key.split("_ref")[0]
                # convert properties to SI values
                if c.get_attr(key).is_set:
                    if "ref" in key:  # reference property object
                        if prop == 'T':
                            c.get_attr(key).ref.delta_SI = hlp.convert_to_SI(
                                cpd['DT'], c.get_attr(key).ref.delta,
                                c.get_attr(prop).unit
                            )
                        else:  # m p h v reference
                            c.get_attr(key).ref.delta_SI = hlp.convert_to_SI(
                                c.get_attr(prop).property_data, c.get_attr(key).ref.delta,
                                c.get_attr(prop).unit
                            )
                    elif key in self.connections_properties_data[c.connection_type()]:  # property object m p h v T
                        try:
                            c.get_attr(key).val_SI = hlp.convert_to_SI(
                                c.get_attr(key).property_data, c.get_attr(key).val, c.get_attr(key).unit
                            )
                        except KeyError as e:
                            raise KeyError(f"{key}--{c.get_attr(key).unit}: {e}")
        # convert properties in components
        for cp in self.comps['object']:
            cp.convert_set_property_to_criterion()

    def decrease_fluid_system_variables_dimension_(self):
        """
        Pre solve simple properties in fluid connections.
        """
        # decrease the dimension of fluid connections
        for c in self.fluid_conns['object']:
            if not c.fluid.is_var:  #
                # simplify the initialization of val_SI of variables to ignore needless primary equation
                c.simplify_specifications()
            # primary properties reference objective
            for key in ['h_ref', 'p_ref', 'm_ref']:  # h_ref, m_ref, P_ref
                prop = key.split("_ref")[0]  # h, p, m
                if (c.get_attr(key).is_set and not c.get_attr(prop).is_set
                        and c.get_attr(key).ref.obj.get_attr(prop).is_set):
                    c.get_attr(prop).val_SI = (
                            c.get_attr(key).ref.obj.get_attr(prop).val_SI
                            * c.get_attr(key).ref.factor
                            + c.get_attr(key).ref.delta_SI)
                    c.get_attr(prop).is_set = True  # properties object
                    c.get_attr(prop).is_var = False
                    c.get_attr(key).is_set = False  # ref equation object
                if (c.get_attr(key).is_set and c.get_attr(prop).is_set
                        and not c.get_attr(key).ref.obj.get_attr(prop).is_set):
                    c.get_attr(key).ref.obj.get_attr(prop).val_SI = (
                            (c.get_attr(prop).val_SI - c.get_attr(key).ref.delta_SI) / c.get_attr(key).ref.factor
                    )
                    c.get_attr(key).ref.obj.get_attr(prop).is_set = True
                    c.get_attr(key).ref.obj.get_attr(prop).is_var = False
                    c.get_attr(key).is_set = False

    def list_pressure_reference_nodes_(self):
        """
        Contain the reference pressure objective of fluid connections.
        """
        self.reference_pressure_object_container = []
        for c in self.fluid_conns['object']:
            if c.p_ref.is_set:
                self.reference_pressure_object_container.append([c, c.p_ref.ref.obj])
        self.reference_pressure_object_container = np.array(self.reference_pressure_object_container, dtype=object)

    def spread_pressure_values_(self):
        """
        Looking for the pressure set value and spreading values along branches after combining pressure objective.

        Returns
        -------

        """
        self.connections_looking_pressure_container = []
        self.connections_spread_pressure_container = []
        mask = self.fluid_comps["object"].apply(lambda c: c.is_spread_pressure_values_start())
        start_components = self.fluid_comps["object"].loc[mask]  # mask: Boolean Series
        for start in start_components:
            start.spread_pressure_values_start()

    def init_fluid_properties_diffusively_(self):
        """
        Initialise the variable fluid properties(unset initial value based on file or others)
        on connection of the network.

        Returns
        -------

        """
        # spread pressure initial value
        self.connections_pressure_boundary_container = []
        self.connections_pressure_initial_container = []
        mask = self.fluid_comps["object"].apply(lambda c: c.is_spread_pressure_initial_start())
        start_components = self.fluid_comps["object"].loc[mask]  # mask: Boolean Series
        for start in start_components:
            start.spread_pressure_initial_start()
        # spread enthalpy initial value
        for c in self.fluid_conns['object']:
            data = c.get_attr('h')
            if data.is_var and not data.initialized:
                self.init_val0(c, 'h')
        # spread mass flow initial value based on branches structure
        mass_flow_ = []
        for c in self.fluid_conns['object']:
            data = c.get_attr('m')
            if data in mass_flow_:
                continue
            mass_flow_.append(data)
            if data.is_var and not data.initialized:
                for branch_data in self.fluid_wrapper_branches.values():
                    if c in branch_data["connections"]:
                        c.get_attr('m').val0 = float(np.random.uniform(min(branch_data["massflow"]),
                                                                       max(branch_data["massflow"])))

    def init_val0(self, c, key):
        r"""
        Set starting values for fluid properties.

        The component classes provide generic starting values for their inlets
        and outlets.

        Parameters
        ----------
        c : Aurora.connections.connection.Connection
            Connection to initialise.
        """
        # retrieve starting values from component information
        val_s = c.source.initialise_source(c, key)
        val_t = c.target.initialise_target(c, key)
        if val_s == 0 and val_t == 0:
            if key == 'p':
                c.get_attr(key).val0 = 1e5
            elif key == 'h':
                c.get_attr(key).val0 = 1e6
        elif val_s == 0:
            c.get_attr(key).val0 = val_t
        elif val_t == 0:
            c.get_attr(key).val0 = val_s
        else:
            c.get_attr(key).val0 = (val_s + val_t) / 2
        # change value according to specified unit system
        c.get_attr(key).val0 = hlp.convert_from_SI(c.get_attr(key).property_data,
                                                   c.get_attr(key).val0,
                                                   c.get_attr(key).unit)

    def init_electric_properties_diffusively_(self):
        """
        Initialise the variable fluid properties(unset initial value based on file or others) in connection of the network.
        """
        for c in self.electric_conns['object']:
            pass
        pass

    def convert_variables_criterion_(self):
        """
        Convert unit of properties of connections to critical units, as well as specifying set_val/initial_val0.

        Returns
        -------

        """
        for c in self.conns['object']:
            # variables of properties in connections
            for key in self.variables_properties_connections_summar[c.connection_type()]:  #
                if not c.get_attr(key).is_set:
                    c.get_attr(key).is_var = True
                    if c.get_attr(key).val0 and not c.get_attr(key).val_SI:
                        c.get_attr(key).val_SI = hlp.convert_to_SI(
                            c.get_attr(key).property_data, c.get_attr(key).val0, c.get_attr(key).unit
                        )
                    elif not c.get_attr(key).val0 and not c.get_attr(key).val_SI:
                        msg = f"Has not initialized {key} in {c.__class__.__name__}: {c.label}"
                        logger.debug(msg)
                    else:
                        pass

    def pack_connections_properties_(self):
        """
        Specify main properties of connections.

        Returns
        -------

        """
        self.node_fluid_objective_container = []
        self.node_pressure_objective_container = []
        self.node_enthalpy_objective_container = []
        self.node_mass_flow_objective_container = []
        self.node_voltage_objective_container = []
        self.node_electricity_objective_container = []
        self.node_frequency_objective_container = []
        self.fluid_objective_list = []
        self.massflow_objective_list = []
        # fluid main properties specify
        for c in self.fluid_conns['object']:
            if c.fluid not in self.node_fluid_objective_container:
                self.node_fluid_objective_container.append(c.fluid)
                c.fluid.label = f"<fluid objective> of connection: {c.label}"
                if c.fluid.is_var:
                    set_fluid_dict = {f: c.fluid.val[f] for f in c.fluid.is_set}
                    all_set_fraction = sum(set_fluid_dict.values())
                    self.fluid_objective_list += [
                        {'obj': c.fluid,
                         'fraction': all_set_fraction,
                         'info': f"{c.label}"}]
            if c.p not in self.node_pressure_objective_container:
                self.node_pressure_objective_container.append(c.p)
                c.p.label = f"<pressure objective> of connection: {c.label}"
            if c.h not in self.node_enthalpy_objective_container:
                self.node_enthalpy_objective_container.append(c.h)
                c.h.label = f"<enthalpy objective> of connection: {c.label}"
            if c.m not in self.node_mass_flow_objective_container:
                self.node_mass_flow_objective_container.append(c.m)
                c.m.label = f"<mass flow objective> of connection: {c.label}"
                if c.m.is_var:
                    self.massflow_objective_list += [{'obj': c.m}]
        # electric main properties specify
        for c in self.electric_conns['object']:
            pass

    def init_properties_from_file_(self):
        """
        Initialise the fluid properties on every connection of the network.

        - Set generic starting values for mass flow, enthalpy and pressure if
          not user specified, read from :code:`init_path` or available from
          previous calculation.
        - For generic starting values precalculate enthalpy value at points of
          given temperature, vapor mass fraction, temperature difference to
          boiling point or fluid state.

        Write parameter information from init_path to connections.
        """
        if self.init_path is not None:
            dfs = self._load_network_state(self.init_path)
            for c in self.conns['object']:
                # set the initial value of primary property of connection in iteration
                df = dfs[c.__class__.__name__]
                if c.label not in df.index:
                    if c.init_path:
                        df = self._load_network_state(c.init_path)[c.__class__.__name__]
                    else:
                        # no matches in the connections of the network and the design files
                        msg = f"Could not find connection {c.label} in init path file."
                        logger.debug(msg)
                        continue
                df.index = df.index.astype(str)
                conn = df.loc[c.label]  # single row of dataframe
                for prop in self.variables_properties_connections_summar[c.connection_type()]:
                    data = c.get_attr(prop)
                    if data.is_var:
                        data.val_SI = hlp.convert_to_SI(data.property_data, conn[prop], conn[prop + '_unit'])
                        data.initialized = True
                if c.connection_type() == 'fluid':
                    for fluid in c.fluid.is_var:
                        c.fluid.val[fluid] = float(conn[fluid])
                        c.fluid.val0[fluid] = float(c.fluid.val[fluid])
                        # c.fluid.initialized.update(fluid)
                c.good_starting_values = True
        for c in self.fluid_conns['object']:
            if sum(c.fluid.val.values()) == 0:
                msg = (
                    'The starting value for the fluid composition of the '
                    'connection ' + c.label + ' is empty. This might lead to '
                    'issues in the initialisation and solving process as '
                    'fluid property functions can not be called. Make sure '
                    'you specified a fluid composition in all parts of the '
                    'network.')
                logger.warning(msg)
        msg = 'Generic fluid property specification complete in the final of initialization.'
        logger.debug(msg)

    def init_properties_logically(self):
        """
        Further initialize the medium properties of connections through logical calculation.

        Returns
        -------

        """
        ###### pre initialize property values ######
        for c in self.fluid_conns['object']:
            if not c.good_starting_values:
                if c.h.is_var and not c.h.initialized:
                    if c.x.is_set:  #
                        try:
                            c.h.val_SI = fp.h_mix_pQ(c.p.val_SI, c.x.val_SI, c.fluid_data, c.mixing_rule)
                            c.h.initialized = True
                        except ValueError:
                            pass
                    if c.T.is_set:
                        try:
                            c.h.val_SI = fp.h_mix_pT(c.p.val_SI, c.T.val_SI, c.fluid_data, c.mixing_rule)
                            c.h.initialized = True
                        except ValueError:
                            pass
                # improved starting values for referenced connections.
                for key in ['m_ref', 'p_ref', 'h_ref']:
                    prop = key.split("_ref")[0]
                    if (c.get_attr(key).is_set
                            and not c.get_attr(prop).is_set
                            and not c.get_attr(prop).initialized
                            and c.get_attr(key).ref.obj.get_attr(prop).initialized):
                        c.get_attr(prop).initialized = True
                        c.get_attr(prop).val_SI = (
                                c.get_attr(key).ref.obj.get_attr(prop).val_SI
                                * c.get_attr(key).ref.factor
                                + c.get_attr(key).ref.delta_SI
                                )
                    if (c.get_attr(key).is_set
                            and not c.get_attr(key).ref.obj.get_attr(prop).is_set
                            and not c.get_attr(key).ref.obj.get_attr(prop).initialized
                            and c.get_attr(prop).initialized):
                        c.get_attr(key).ref.obj.get_attr(prop).initialized = True
                        c.get_attr(key).ref.obj.get_attr(prop).val_SI = (
                                (c.get_attr(prop).val_SI -
                                 c.get_attr(key).ref.delta_SI) /
                                c.get_attr(key).ref.factor
                                )
            # starting values for specified subcooling/overheating
            # and state specification. These should be recalculated even with
            # good starting values, for example, when one exchanges enthalpy
            # with boiling point temperature difference.
            # specified vapour content values, temperature values as well as subccooling/overheating and state specification
            if (c.Td_bp.is_set or c.state.is_set) and c.h.is_var:  #
                if ((c.Td_bp.is_set and c.Td_bp.val_SI > 0) or
                        (c.state.is_set and c.state.val == 'g')):
                    h = fp.h_mix_pQ(c.p.val_SI, 1, c.fluid_data)  # enthalpy of saturated vapour
                    if c.h.val_SI < h:
                        c.h.val_SI = h * 1.001
                        c.h.initialized = True
                elif ((c.Td_bp.is_set and c.Td_bp.val_SI < 0) or
                        (c.state.is_set and c.state.val == 'l')):
                    h = fp.h_mix_pQ(c.p.val_SI, 0, c.fluid_data)  # enthalpy of saturated liquid
                    if c.h.val_SI > h:
                        c.h.val_SI = h * 0.999
                        c.h.initialized = True
        ###### initialize enthalpy value based on enthalpy inequation due to pressure value ######
        ###### initialize mass flow value based on energy constraints ######
        ###### initialize fluid values due to mass flow values of flow branches ######
        pass

    def init_reaction_parameters(self):
        r"""Initialize reaction (gas name aliases and LHV)."""
        mask = self.fluid_comps["object"].apply(lambda c: c.is_reaction_component())
        combustion_components = self.fluid_comps["object"].loc[mask]  # mask: Boolean Series
        for combustion in combustion_components:
            combustion.setup_reaction_parameters()

    @staticmethod
    def _load_network_state(json_path):
        r"""
        Read network state from given file.

        Parameters
        ----------
        json_path : str
            Path to network information.
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        # reversion dataframe
        dfs = {}
        for key, value in data["Connection"].items():  # key: connection type
            dfs[key] = pd.DataFrame.from_dict(value, orient="index").fillna(np.nan)
            dfs[key].index = dfs[key].index.astype(str)
        for key, value in data["Component"].items():  # key: component type
            dfs[key] = pd.DataFrame.from_dict(value, orient="index").fillna(np.nan)
            dfs[key].index = dfs[key].index.astype(str)
        for key, value in data["Bus"].items():  # key: bus.label
            dfs[key] = pd.DataFrame.from_dict(value, orient="index").fillna(np.nan)
            dfs[key].index = dfs[key].index.astype(str)
        return dfs

    def initialize_variables_bounds(self):
        """
        Generate bounds of variables.
        Returns:

        """
        msg = f"########-----Initializing variables bounds-----########"
        logger.info(msg)
        # fluid properties
        # m bounds of mass branches
        for item in self.massflow_objective_list:
            item['obj'].min_val, item['obj'].max_val = self.m_range_SI
        # p、h bounds of connections
        for c in self.fluid_conns['object']:
            c.build_fluid_data()
            c.bounds_T_generate()
            c.bounds_ph_generate()
        # p bounds of connections around components due to phase state request
        for cp in self.fluid_comps['object']:
            cp.bounds_p_generate()
        for cp in self.fluid_comps['object']:
            cp.bounds_h_generate()
        # electric properties
        for c in self.electric_conns['object']:
            pass
        for cp in self.electric_comps['object']:
            pass

    def generate_node_equation_system(self):
        """
        Generate node equation system after simplifying the topological structure.

        Returns
        -------

        """
        self.sorted_variables_module_container = []
        self.sorted_equations_module_container = []
        ######------ summarize equations ------######
        # fluid composition equations, contain fluid equations
        self.fluid_equations_module_container = []
        mask = self.fluid_comps["object"].apply(lambda c: c.is_fluid_composition_component())
        composition_components = self.fluid_comps["object"].loc[mask]  # mask: Boolean Series
        for cp in composition_components:
            cp.manage_fluid_equations()
            self.num_fluid_eqs += cp.num_fluid_eqs
        # equations of node connections
        for c in self.conns['object']:
            c.summarize_equations()  # connection initialization: count self_set no_primary variables/equations
            # get number of equations
            self.num_conn_eqs += c.num_eq
        # equations of components
        for cp in self.comps['object']:
            cp.summarize_equations()
            self.num_comp_eqs += cp.num_eq  # number of all equations including constrains and parameter equation chosen
        # equations of buses
        for b in self.busses.values():
            b.summarize_equations()
            self.num_bus_eqs += b.P.is_set * 1
        ######------ components preprocess ------######
        for cp in self.comps['object']:
            # component initialisation
            cls = cp.__class__.__name__
            cp.preprocess()
            for spec in self.specifications[cls].keys():  # groups, chars, variables, properties
                if len(cp.get_attr(self.specifications['lookup'][spec])) > 0:  # comp.spec_specifications
                    self.specifications[cls][spec].loc[cp.label] = (
                        cp.get_attr(self.specifications['lookup'][spec]))
        ######------ solve isolated equations in advance ------######
        self.num_presolved_eqs = 0
        presolve_equation_continue = True
        presolve_equations_info = []
        presolve_batch = 0
        while presolve_equation_continue:
            presolve_batch += 1
            logger.debug(f'------presolve_batch: {presolve_batch}------')
            presolve_equation_continue = False
            for equation in self.sorted_equations_module_container:
                if hasattr(equation, 'func_params'):
                    kwargs = equation.func_params.copy()
                else:
                    kwargs = {}
                try:
                    if equation.solve_isolated(**kwargs):
                        presolve_equation_continue = True
                        self.num_presolved_eqs += equation.num_eq
                        presolve_equations_info.append(equation.label)
                        self.sorted_equations_module_container.remove(equation)
                        msg = f'Has pre_solved equation: {equation.label}'
                        logger.debug(msg)
                except ValueError as e:
                    msg = f"Has something wrong with pre_solving equation: {equation.label}: {e}"
                    logger.error(msg)
        msg = f'Has removed {self.num_presolved_eqs} pre_solved equations totally.'
        logger.debug(msg)
        ######------ summarize variables ------######
        # variables of properties in connections
        # fluid connections properties
        # fluid variables
        self.fluid_variables_module_container = []
        for property in self.node_fluid_objective_container:
            for fluid_var in property.is_var:
                property.J_col[fluid_var] = self.num_fluid_vars
                self.fluid_variables_module_container.append({'object': property, 'type': 'fluid', 'fluid': fluid_var})
                self.num_fluid_vars += 1
        # other variables summarization
        # mass flow
        for property in self.node_mass_flow_objective_container:
            if not property.is_set:
                property.is_var = True
                property.J_col = self.num_conn_vars + self.num_conn_mass_flow_vars
                self.sorted_variables_module_container.append({'object': property, 'type': 'mass flow'})
                self.num_conn_mass_flow_vars += property.dimension
            elif property.is_set:
                property.is_var = False
        self.num_conn_vars += self.num_conn_mass_flow_vars
        # pressure
        for property in self.node_pressure_objective_container:
            if not property.is_set:
                property.is_var = True
                property.J_col = self.num_conn_vars + self.num_conn_pressure_vars
                self.sorted_variables_module_container.append({'object': property, 'type': 'pressure'})
                self.num_conn_pressure_vars += property.dimension
            elif property.is_set:
                property.is_var = False
        self.num_conn_vars += self.num_conn_pressure_vars
        # enthalpy
        for property in self.node_enthalpy_objective_container:
            if not property.is_set:
                property.is_var = True
                property.J_col = self.num_conn_vars + self.num_conn_enthalpy_vars
                self.sorted_variables_module_container.append({'object': property, 'type': 'enthalpy'})
                self.num_conn_enthalpy_vars += property.dimension
            elif property.is_set:
                property.is_var = False
        self.num_conn_vars += self.num_conn_enthalpy_vars
        # electric connections properties
        for property in self.node_voltage_objective_container:
            pass
        for property in self.node_electricity_objective_container:
            pass
        for property in self.node_frequency_objective_container:
            pass
        # component variables properties
        for cp in self.comps['object']:
            cp.summarize_variables()
            self.num_comp_vars += cp.num_vars
        ###### specify parameters/properties set in all connections ######
        for c in self.conns['object']:
            # variables 0 to 9: fluid properties
            local_vars = self.connections_properties_data[c.connection_type()]
            row = [c.get_attr(var).is_set for var in local_vars]
            # write information to specifaction dataframe
            self.specifications[c.__class__.__name__].loc[c.label, local_vars] = row
            row = [c.get_attr(var).is_set for var in self.specifications[f'{c.__class__.__name__}_Ref'].columns]
            # write refrenced value information to specifaction dataframe
            self.specifications[f'{c.__class__.__name__}_Ref'].loc[c.label] = row
            # variables 9 to last but one: fluid mass fractions
            if c.connection_type() == 'fluid':
                fluids = list(self.all_fluids)
                row = [True if f in c.fluid.is_set else False for f in fluids]
                self.specifications[c.__class__.__name__].loc[c.label, fluids] = row

    def solve(self, mode, init_path=None, design_path=None,
              max_iter=50, min_iter=4, init_only=False, init_previous=True,
              use_cuda=False, use_tensor=False, print_results=False, colored=True, colors=None,
              print_iterations=True,
              plot_iteration=False, algo_factor=0.1, prepare_fast_lane=False):
        r"""
        Solve the network.

        - Check network consistency.
        - Initialise calculation and preprocessing.
        - Perform actual calculation.
        - Postprocessing.

        It is possible to check programatically, if a network was solved
        successfully with the `.converged` property.

        Parameters
        ----------
        mode : str
            Choose from 'design' and 'offdesign'.

        init_path : str
            Path to the folder, where your network was saved to, e.g.
            saving to :code:`nw.save('myplant/tests')` would require loading
            from :code:`init_path='myplant/tests'`.

        design_path : str
            Path to the folder, where your network's design case was saved to,
            e.g. saving to :code:`nw.save('myplant/tests')` would require
            loading from :code:`design_path='myplant/tests'`.

        max_iter : int
            Maximum number of iterations before calculation stops, default: 50.

        min_iter : int
            Minimum number of iterations before calculation stops, default: 4.

        init_only : boolean
            Perform initialisation only, default: :code:`False`.

        init_previous : boolean
            Initialise the calculation with values from the previous
            calculation, default: :code:`True`.

        use_cuda : boolean
            Use cuda instead of numpy for matrix inversion, default:
            :code:`False`.

        Note
        ----
        For more information on the solution process
        """
        ## to own function
        self.new_design = False
        if self.design_path == design_path and design_path is not None:
            for c in self.conns['object']:
                if c.new_design:
                    self.new_design = True
                    break
            if not self.new_design:
                for cp in self.comps['object']:
                    if cp.new_design:
                        self.new_design = True
                        break
        else:
            self.new_design = True
        #
        self.converged = False
        self.progress = True
        self.init_path = init_path
        self.design_path = design_path
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.init_previous = init_previous
        self.use_cuda = use_cuda
        self.use_tensor = use_tensor
        self.algo_factor = algo_factor
        #
        if self.use_cuda and cu is None:
            msg = (
                'Specifying use_cuda=True requires cupy to be installed on '
                'your machine. Numpy will be used instead.'
            )
            logger.warning(msg)
            self.use_cuda = False

        if mode not in ['offdesign', 'design']:
            msg = 'Mode must be "design" or "offdesign".'
            logger.error(msg)
            raise ValueError(msg)
        else:
            self.mode = mode
        # prevent topological construction error
        if not self.checked:
            self.check_network()  # check and create branches
        msg = (
            "Solver properties:\n"
            f" - mode: {self.mode}\n"
            f" - init_path: {self.init_path}\n"
            f" - design_path: {self.design_path}\n"
            f" - min_iter: {self.min_iter}\n"
            f" - max_iter: {self.max_iter}"
        )
        logger.debug(msg)
        msg = (
            "Network information:\n"
            f" - Number of components: {len(self.comps)}\n"
            f" - Number of fluid components: {self.fluid_components_number}\n"
            f" - Number of electric components: {self.electric_components_number}\n"
            f" - Number of connections: {len(self.conns)}\n"
            f" - Number of fluid connections: {len(self.fluid_conns)}\n"
            f" - Number of electric connections: {len(self.electric_conns)}\n"
            f" - Number of busses: {len(self.busses)}"
        )
        logger.debug(msg)
        # initialize properties value, generation equation system
        self.initialise()  # initialise connections and components, set index of all parameters
        self.log_system_information_()  #
        self.sort_properties_objective_constructure_()
        self.generate_bounds()
        self.generate_constraints()
        self.visual_properties_information_()
        self.check_setting_properties_constructure_()

        if init_only:  # save for debug only
            # self.reset_topology_reduction_specifications()
            return

        msg = 'Starting solver.'
        logger.info(msg)
        # algorithm core
        self.solve_loop(print_iterations=print_iterations, plot_iteration=plot_iteration)  # loop iteration core algorithm
        # iteration monitor
        if not self.progress:
            msg = (
                'The solver does not seem to make any progress, aborting '
                'calculation. Residual value is '
                '{:.2e}'.format(norm(self.residual)) + '. This frequently '
                'happens, if the solver pushes the fluid properties out of '
                'their feasible range.'
            )
            logger.warning(msg)
            # return
        # calculate all properties
        self.postprocess_properties_()
        msg = 'Calculation complete.'
        logger.info(msg)
        # print results
        if print_results:
            self.print_results(colored=colored, colors=colors, print_results=print_results)
        # recover the shared properties objective
        if not prepare_fast_lane:  # if the solver erupt, the topology resetting won't be processed
            self.reset_topology_reduction_specifications()
        return

    def log_system_information_(self):
        r"""Log,logger the number of variables and equations."""
        # number of user defined functions
        self.num_ude_eqs = len(self.user_defined_eq)
        for func in self.user_defined_eq.values():
            # remap connection objects
            func.conns = [
                self.conns.loc[c.label, 'object'] for c in func.conns
            ]
            # remap jacobian
            func.jacobian = {}
        # total number of variables
        self.num_eqs = self.num_comp_eqs + self.num_conn_eqs + self.num_bus_eqs + self.num_ude_eqs - self.num_presolved_eqs
        self.num_vars = self.num_conn_vars + self.num_comp_vars
        ###### equation summar ######
        logger.debug(f'------Equation Summarization------')
        msg = f"Total number of equations: {self.num_eqs}"
        logger.debug(msg)
        msg = f'Number of connection equations: {self.num_conn_eqs}.'  # sum of no_primary to primary property equations set in all connections
        logger.debug(msg)
        msg = f'Number of component equations: {self.num_comp_eqs}.'  # number of all equations including constrains and parameter equation chosen
        logger.debug(msg)
        msg = f'Number of bus equations: {self.num_bus_eqs}.'  # sum of P set
        logger.debug(msg)
        msg = f'Number of user defined equations: {self.num_ude_eqs}.'
        logger.debug(msg)
        msg = f'Number of presolved equations: {self.num_presolved_eqs}.'
        logger.debug(msg)
        ##### variable summar ######
        logger.debug(f'------Variable Summarization------')
        msg = f'Total number of variables: {self.num_vars}.'
        logger.debug(msg)
        msg = f'Number of component variables: {self.num_comp_vars}.'  # index of the parameters chosen of components
        logger.debug(msg)
        msg = f"Number of connection variables: {self.num_conn_vars}."  # index of the primary variables of connections
        logger.debug(msg)
        msg = f'Number of mass flow variables: {self.num_conn_mass_flow_vars}.'
        logger.debug(msg)
        msg = f'Number of pressure variables: {self.num_conn_pressure_vars}.'
        logger.debug(msg)
        msg = f'Number of enthalpy variables: {self.num_conn_enthalpy_vars}.'
        logger.debug(msg)
        ###### fluid composition summar ######
        logger.debug(f'------Fluid Module Summarization------')
        msg = f'Number of fluid composition equations: {self.num_fluid_eqs}.'
        logger.debug(msg)
        msg = f'Number of fluid variables: {self.num_fluid_vars}.'
        logger.debug(msg)

    def sort_properties_objective_constructure_(self):
        """
        Sort properties scale and objective of systems.

        Returns
        -------

        """
        self.sorted_equations_objective = np.empty([self.num_eqs], dtype=object)
        self.sorted_variables_objective = np.empty([self.num_vars], dtype=object)
        self.sorted_fluid_equations_objective = np.empty([self.num_fluid_eqs], dtype=object)

        self.residual_scale = np.ones([self.num_eqs])  # the factor of residual vector
        self.variables_scale = np.ones([self.num_vars])  # the factor of variables vector

        self.variables_list_visual = np.empty([self.num_vars], dtype=str)
        self.fluid_variables_list_visual = np.empty([self.num_fluid_vars], dtype=str)

        self.functions_list_visual = np.empty([self.num_eqs], dtype=str)
        self.fluid_functions_list_visual = np.empty([self.num_fluid_eqs], dtype=str)

        self.residual = np.zeros([self.num_eqs])  # residual vector
        self.increment = np.ones([self.num_vars])  # the increment array of parameters
        self.jacobian = np.zeros((self.num_eqs, self.num_vars))  # jacobin matrix
        self.tensor = np.zeros((self.num_vars, self.num_vars, self.num_eqs))
        # apply for fluid module
        self.fluid_residual = np.zeros([self.num_fluid_eqs])
        self.fluid_increment = np.ones([self.num_fluid_vars])
        self.fluid_jacobian = np.zeros((self.num_fluid_eqs, self.num_fluid_vars))
        self.fluid_tensor = np.zeros((self.num_fluid_vars, self.num_fluid_vars, self.num_fluid_eqs))

        # Sort: sorted all the equations
        equations_index = 0
        for equation_module in self.sorted_equations_module_container:
            self.sorted_equations_objective[equations_index: equations_index + equation_module.num_eq] = \
                [equation_module for _ in range(equation_module.num_eq)]
            self.residual_scale[equations_index: equations_index + equation_module.num_eq] = \
                [equation_module.scale for _ in range(equation_module.num_eq)]
            equation_module.row = equations_index
            equations_index += equation_module.num_eq
        if equations_index > self.num_eqs:
            logger.error(f'the number of equations: {equations_index} out of {self.num_eqs}.')
        # Sort: sorted fluid equations
        fluid_equations_index = 0
        for fluid_equation_module in self.fluid_equations_module_container:
            self.sorted_fluid_equations_objective[fluid_equations_index: fluid_equations_index + fluid_equation_module.num_eq] = \
                [fluid_equation_module for _ in range(fluid_equation_module.num_eq)]
            fluid_equation_module.row = fluid_equations_index
            fluid_equations_index += fluid_equation_module.num_eq
        if fluid_equations_index > self.num_fluid_eqs:
            logger.error(f'the number of fluid equations: {fluid_equations_index} out of {self.num_fluid_eqs}.')
        # Sort: sorted property variables (no fluid composition)
        variables_index = 0
        for variable_item in self.sorted_variables_module_container:
            self.sorted_variables_objective[variables_index: variables_index + variable_item['object'].dimension] = \
                [variable_item for _ in range(variable_item['object'].dimension)]
            self.variables_list_visual[variables_index: variables_index + variable_item['object'].dimension] = \
                [f"{variables_index + i}: {variable_item['object'].label}" for i in range(variable_item['object'].dimension)]
            self.variables_scale[variables_index: variables_index + variable_item['object'].dimension] = \
                [variable_item['object'].scale if not variable_item['type'] == 'component variable'
                 else variable_item['object'].var_scale
                 for _ in range(variable_item['object'].dimension)]
            variables_index += variable_item['object'].dimension

        # Visual: containing the fluid variable names
        for index, items in enumerate(self.fluid_variables_module_container):
            if items['type'] == 'fluid':
                self.fluid_variables_list_visual[index] = f"{index}: {items['object'].label}:{items['fluid']} fraction"
            else:
                logger.error(f'unknown fluid variable type: {items["type"]}')
        # Visual: containing the equation names
        self.checking_matrix = np.zeros([self.num_eqs, self.num_vars])
        function_index = 0
        while function_index < self.num_eqs:
            self.functions_list_visual[function_index: function_index + self.sorted_equations_objective[function_index].num_eq] = \
                [f"{function_index + i}: {self.sorted_equations_objective[function_index].label}"
                 for i in range(self.sorted_equations_objective[function_index].num_eq)]
            # generate checking matrix
            if hasattr(self.sorted_equations_objective[function_index], 'func_params'):
                kwargs = self.sorted_equations_objective[function_index].func_params.copy()
            else:
                kwargs = {}
            variables_columns = self.sorted_equations_objective[function_index].variables_columns(**kwargs)
            if len(variables_columns) == self.sorted_equations_objective[function_index].num_eq:
                for i in range(self.sorted_equations_objective[function_index].num_eq):
                    # variables_columns[i] = list(set(variables_columns[i])).sort()
                    if variables_columns[i] == []:
                        pass
                    else:
                        try:
                            for column in variables_columns[i]:
                                self.checking_matrix[function_index + i, column] = 1
                        except TypeError:
                            msg = f"the variables columns of {self.sorted_equations_objective[function_index].label}: {variables_columns} is None"
                            logger.error(msg)
            else:
                logger.error(f'the variables_columns index {len(variables_columns)} '
                             f'out of range {self.sorted_equations_objective[function_index].num_eq}')
            # satisfy index of equation group
            function_index += self.sorted_equations_objective[function_index].num_eq
        # Visual: containing the fluid equation names
        fluid_function_index = 0
        while fluid_function_index < self.num_fluid_eqs:
            self.fluid_functions_list_visual[fluid_function_index: fluid_function_index + self.sorted_fluid_equations_objective[fluid_function_index].num_eq] = \
                [f'{fluid_function_index + i}: {self.sorted_fluid_equations_objective[fluid_function_index].label}'
                f'----{self.sorted_fluid_equations_objective[fluid_function_index].fluid_composition_list[i]}'
                for i in range(self.sorted_fluid_equations_objective[fluid_function_index].num_eq)]
            fluid_function_index += self.sorted_fluid_equations_objective[fluid_function_index].num_eq

    @staticmethod
    def checking_matrix_rank(matrix):  # SI_unit
        def checking_matrix_rows(A):
            dependent_rows = []
            dio1_rows = [[] for _ in range(A.shape[1])]
            for i in range(A.shape[0]):
                if np.all(A[i, :] == 0):
                    dependent_rows.append(i)
                columns_set_list = []
                for j in range(A.shape[1]):
                    data = A[i, j]
                    if data == 1:
                        columns_set_list.append(j)
                if len(columns_set_list) == 1:
                    dio1_rows[columns_set_list[0]] += [i]
            for row in dio1_rows:
                if len(row) > 1:
                    dependent_rows += row
            return dependent_rows

        def checking_matrix_columns(A):
            dependent_cols = []
            dio1_cols = [[] for _ in range(A.shape[0])]
            for j in range(A.shape[1]):
                if np.all(A[:, j] == 0):
                    dependent_cols.append(j)
                rows_set_list = []
                for i in range(A.shape[0]):
                    data = A[i, j]
                    if data == 1:
                        rows_set_list.append(i)
                if len(rows_set_list) == 1:
                    dio1_cols[rows_set_list[0]] += [j]
            for col in dio1_cols:
                if len(col) > 1:
                    dependent_cols += col
            return dependent_cols

        return checking_matrix_rows(matrix), checking_matrix_columns(matrix)

    def generate_bounds(self):
        self.bounds = [(10, 1e12) for _ in range(self.num_vars)]
        self.fluid_bounds = [(0, 1 + ERR) for _ in range(self.num_fluid_vars)]
        # p、h bounds of connections
        for c in self.fluid_conns['object']:
            c.bounds_ph_generate()
        for cp in self.fluid_comps['object']:
            cp.bounds_h_generate()
        # set bounds for common variables
        for data in self.sorted_variables_module_container:
            container = data["object"]
            for i in range(container.dimension):
                self.bounds[container.J_col + i] = (container.min_val / self.variables_scale[container.J_col + i],
                                                    container.max_val / self.variables_scale[container.J_col + i])
        # set bounds for fluid variables
        for data in self.fluid_variables_module_container:
            container = data["object"]
            if data["type"] == "fluid":
                self.fluid_bounds[container.J_col[data["fluid"]]] = (0, 1 + ERR)

    def generate_constraints(self):
        # p、h ineq constraints,
        self.constraints = []
        self.jac_sparse_info = []  # contain information of constraints matrix
        self.constraints_k_list = []
        self.constraints_epsilon_list = []
        # fluid ineq constraints
        self.fluid_composition_constraints = []

        def make_composition_constraint(composition_groups, fraction, info):
            def constraint_func(x):
                total = 0
                for idx_group in composition_groups:
                    total += np.sum(x[idx_group])
                return -(total - (1 - fraction))

            def constraint_jac(x):
                jac = np.zeros_like(x)
                for idx_group in composition_groups:
                    jac[idx_group] = -1
                return jac

            return {'type': 'ineq', 'fun': constraint_func, 'jac': constraint_jac, 'jac_indices': composition_groups,
                    'k': 10, 'epsilon': -0.05, 'bounds': (-ERR, ERR+1-fraction), 'info': f'composition constraint of {info}'}  # , 'jac': constraint_jac    -fraction

        def make_enthalpy_constraints(p_index, h_index, c):
            # min enthalpy constraints：h ≥ h_min(p)
            def enthalpy_min_constraint(x):
                p = x[p_index] * self.variables_scale[p_index]
                h = x[h_index]
                return h - c.calc_min_enthalpy(p) / self.variables_scale[h_index]  # ≥0

            def enthalpy_min_constraint_jac(x):
                func = enthalpy_min_constraint
                return self.approx_jacobian(func, x, [p_index, h_index])

            # max enthalpy constraints：h ≤ h_max(p)
            def enthalpy_max_constraint(x):
                p = x[p_index] * self.variables_scale[p_index]
                h = x[h_index]
                return c.calc_max_enthalpy(p) / self.variables_scale[h_index] - h  # ≥0

            def enthalpy_max_constraint_jac(x):
                func = enthalpy_max_constraint
                return self.approx_jacobian(func, x, [p_index, h_index])

            # constraints list
            constraints = [
                {'type': 'ineq', 'fun': enthalpy_min_constraint, 'jac': enthalpy_min_constraint_jac, 'jac_indices': [p_index, h_index],
                 'k': 10, 'epsilon': 1, 'bounds': (-ERR, np.inf), 'info': f'enthalpy_min_constraint of {c.label}'},
                {'type': 'ineq', 'fun': enthalpy_max_constraint, 'jac': enthalpy_max_constraint_jac, 'jac_indices': [p_index, h_index],
                 'k': 10, 'epsilon': 1, 'bounds': (-ERR, np.inf), 'info': f'enthalpy_max_constraint of {c.label}'},
            ]
            return constraints

        # fluid fraction ineq constraints of connections
        for item in self.fluid_objective_list:
            fluid = item['obj']
            fraction_index_list = [fluid.J_col[f] for f in fluid.is_var]
            fraction = item['fraction']
            info = item['info']
            fluid_constraint = make_composition_constraint(fraction_index_list, fraction, info)
            self.fluid_composition_constraints.append(fluid_constraint)
        # p、h ineq constraints of connections
        for c in self.fluid_conns['object']:
            if c.p.is_var and c.h.is_var:
                entalpy_constraints = make_enthalpy_constraints(c.p.J_col, c.h.J_col, c)
                self.constraints.extend(entalpy_constraints)
        # h ineq constraints of heat exchanger class of components, maybe so strict ?
        # for cp in self.comps['object']:
        #     cp.constraints_h_generate()
        # generate the constraints sparse info、k_list、epsilon_list
        for i, constraint_dict in enumerate(self.constraints):
            self.jac_sparse_info.append(constraint_dict['jac_indices'])
            self.constraints_k_list.append(constraint_dict['k'])
            self.constraints_epsilon_list.append(constraint_dict['epsilon'])

    @staticmethod
    def approx_jacobian(func, x, index_list, epsilon=1e-6):
        n = len(x)
        f0 = func(x)
        jac = np.zeros(n)
        for i in index_list:
            x_pert = x.copy()
            x_pert[i] += epsilon
            f_pert = func(x_pert)
            jac[i] = (f_pert - f0) / epsilon
        return jac

    @staticmethod
    def wrap_label_(label):
        separators = ['of', 'in']
        for sep in separators:
            if sep in label:
                parts = label.split(sep)
                if len(parts) > 1:
                    # 尝试找到最佳换行位置
                    current_line = parts[0]
                    result = current_line
                    for part in parts[1:]:
                        if len(current_line + sep + part) > 20:
                            result += '\n' + sep + part
                            current_line = part
                        else:
                            result += sep + part
                            current_line += sep + part
                    return result
        return label

    def visual_properties_information_(self):
        """
        Visualization properties of system.

        :return:
        """
        properties_information = ''
        ###### contain variables information ######
        variables_info_dict = {
            'label': str,
            'min value': float,
            'max value': float,
            'unit': str,
            'scale': float,
        }
        variables_info_dataframe = pd.DataFrame(columns=list(variables_info_dict.keys())).astype(variables_info_dict)
        for col in range(self.num_vars):
            variable_container = self.sorted_variables_objective[col]['object']
            if variable_container.dimension == 1:
                min_value = convert_from_SI(variable_container.property_data, variable_container.min_val, variable_container.unit)
                max_value = convert_from_SI(variable_container.property_data, variable_container.max_val, variable_container.unit)
            else:
                min_value = convert_from_SI(variable_container.property_data, variable_container.min_val,
                                            variable_container.unit)[col - variable_container.J_col]
                max_value = convert_from_SI(variable_container.property_data, variable_container.max_val,
                                            variable_container.unit)[col - variable_container.J_col]
            variables_info_dataframe.loc[col] = [variable_container.label, min_value, max_value, variable_container.unit, self.variables_scale[col]]
        properties_information += f"\n##### VARIABLES #####\n"
        properties_information += tabulate(variables_info_dataframe, headers='keys', tablefmt='psql', floatfmt='.3e')
        ###### contain equations information ######
        equations_info_dict = {
            'label': str,
            'scale': float,
        }
        equations_info_dataframe = pd.DataFrame(columns=list(equations_info_dict.keys())).astype(equations_info_dict)
        for row in range(self.num_eqs):
            equation_container = self.sorted_equations_objective[row]
            equations_info_dataframe.loc[row] = [equation_container.label, self.residual_scale[row]]
        properties_information += f"\n##### EQUATIONS #####\n"
        properties_information += tabulate(equations_info_dataframe, headers='keys', tablefmt='psql', floatfmt='.3e')
        ###### contain fluid variables ######
        fluid_variables_info_dict = {
            'label': str,
            'fluid composition': str,
        }
        fluid_variables_info_dataframe = pd.DataFrame(columns=list(fluid_variables_info_dict.keys())).astype(fluid_variables_info_dict)
        for col in range(self.num_fluid_vars):
            fluid_variables_info_dataframe.loc[col] = [self.fluid_variables_module_container[col]['object'].label,
                                                       self.fluid_variables_module_container[col]['fluid']]
        properties_information += f"\n##### FLUID VARIABLES #####\n"
        properties_information += tabulate(fluid_variables_info_dataframe, headers='keys', tablefmt='psql', floatfmt='.3e')
        ###### contain fluid equations ######
        fluid_equations_info_dict = {
            'label': str,
            'fluid composition': str,
        }
        fluid_equations_info_dataframe = pd.DataFrame(columns=list(fluid_equations_info_dict.keys())).astype(fluid_equations_info_dict)
        for row in range(self.num_fluid_eqs):
            fluid_equation_container = self.sorted_fluid_equations_objective[row]
            fluid_composition = fluid_equation_container.fluid_composition_list[row - fluid_equation_container.row]
            fluid_equations_info_dataframe.loc[row] = [fluid_equation_container.label, fluid_composition]
        properties_information += f"\n##### FLUID EQUATIONS #####\n"
        properties_information += tabulate(fluid_equations_info_dataframe, headers='keys', tablefmt='psql', floatfmt='.3e')
        # log the variables and equations set
        logger.debug(properties_information)
        # logger temperature range
        temperature_info_dict = {
            'min value': float,
            'max value': float,
            'unit': str,
        }
        temperature_info_dataframe = pd.DataFrame(columns=list(temperature_info_dict.keys())).astype(temperature_info_dict)
        for c in self.fluid_conns['object']:
            if c.h.is_var:
                temperature_info_dataframe.loc[c.label] = [convert_from_SI(c.T.property_data, c.T.min_val, c.T.unit),
                                                           convert_from_SI(c.T.property_data, c.T.max_val, c.T.unit),
                                                           c.T.unit]
        temperature_information = f"\n##### TEMPERATURE BOUNDS #####\n"
        temperature_information += tabulate(temperature_info_dataframe, headers='keys', tablefmt='psql', floatfmt='.3e')
        logger.debug(temperature_information)

    def check_setting_properties_constructure_(self):
        """
        Check the topologic constructure of variables-equations,
        whether the number of supplied variables is sufficient.

        Returns
        -------

        """
        # checking the linear dependence rows/columns of matrix
        linear_rows_index, linear_columns_index = self.checking_matrix_rank(self.checking_matrix)
        type_dict_ = {
            'label': str,
        }
        linear_information = ''
        linear_variables_info_data = pd.DataFrame(columns=list(type_dict_.keys())).astype(type_dict_)
        linear_equations_info_data = pd.DataFrame(columns=list(type_dict_.keys())).astype(type_dict_)
        for row in linear_rows_index:
            linear_equations_info_data.loc[row] = [self.sorted_equations_objective[row].label,]
        for col in linear_columns_index:
            linear_variables_info_data.loc[col] = [self.sorted_variables_objective[col]['object'].label]
        if len(linear_rows_index) > 0:
            linear_information += f"\n##### LINEAR EQUATIONS #####\n"
            linear_information += tabulate(linear_equations_info_data, headers='keys', tablefmt='psql', floatfmt='.3e')
        if len(linear_columns_index) > 0:
            linear_information += f"\n##### LINEAR VARIABLES #####\n"
            linear_information += tabulate(linear_variables_info_data, headers='keys', tablefmt='psql', floatfmt='.3e')
        # log linear information
        if len(linear_rows_index) > 0 or len(linear_columns_index) > 0:
            logger.error(linear_information)
            raise hlp.AURORANetworkError(linear_information)
        else:
            logger.debug(f'has no linear dependence in network topology')
        # check the number of variables supplied
        if self.num_eqs > self.num_vars:
            msg = (
                f"You have not provided enough parameters: {self.num_vars} supplied, "
                f"{self.num_eqs} required. Aborting calculation!"
            )
            logger.error(msg)
            raise hlp.AURORANetworkError(msg)
        elif self.num_eqs < self.num_vars:
            msg = (
                f"You have provided too many parameters: {self.num_vars} supplied, "
                f"{self.num_eqs} required. Aborting calculation!"
            )
            logger.error(msg)
            raise hlp.AURORANetworkError(msg)
        # check the number of fluid variables
        if self.num_fluid_eqs > self.num_fluid_vars:
            msg = (f"You have provided too many parameters for fluid module: {self.num_fluid_vars} supplied, "
                   f"{self.num_fluid_eqs} required. Aborting calculation!")
            logger.error(msg)
            raise hlp.AURORANetworkError(msg)
        elif self.num_fluid_eqs < self.num_fluid_vars:
            msg = (f"You have not provided enough parameters for fluid module: {self.num_fluid_vars} supplied, "
                   f"{self.num_fluid_eqs} required. Aborting calculation!")
            logger.error(msg)
            raise hlp.AURORANetworkError(msg)

    def solve_loop(self, print_iterations=True, plot_iteration=True):
        r"""Loop of the iteration algorithm."""
        # document history
        self.residual_history = []
        self.norm_residual_history = []
        self.increment_history = []
        # iteration param definition
        self.jacobian_temp = self.jacobian.copy()  # duplicate the jacobin matrix
        self.tensor_temp = self.tensor.copy()  # duplicate the tensor
        self.fluid_jacobian_temp = self.fluid_jacobian.copy()
        # contain the variables
        self.variables_vector = np.zeros([self.num_vars])  #
        self.fluid_variables_vector = np.zeros([self.num_fluid_vars])
        # variables bounds
        self.lb = np.array([b[0] for b in self.bounds])
        self.ub = np.array([b[1] for b in self.bounds])
        # constraints bounds
        self.cl = np.array([con['bounds'][0] for con in self.constraints])
        self.cu = np.array([con['bounds'][1] for con in self.constraints])
        # generate index specification
        self.fl_index = [k for k, v in enumerate(self.fluid_variables_module_container) if v["type"] == "fluid"]
        self.m_index = [k for k, v in enumerate(self.sorted_variables_objective) if v["type"] == "mass flow"]
        self.p_index = [k for k, v in enumerate(self.sorted_variables_objective) if v["type"] == "pressure"]
        self.h_index = [k for k, v in enumerate(self.sorted_variables_objective) if v["type"] == "enthalpy"]
        self.U_index = [k for k, v in enumerate(self.sorted_variables_objective) if v["type"] == "voltage"]
        self.I_index = [k for k, v in enumerate(self.sorted_variables_objective) if v["type"] == "electricity"]
        self.f_index = [k for k, v in enumerate(self.sorted_variables_objective) if v["type"] == "frequency"]
        self.cp_index = [k for k,v in enumerate(self.sorted_variables_objective) if
                         k not in self.m_index + self.p_index + self.h_index + self.U_index + self.I_index + self.f_index]
        # get the normal variables initial values to variables_container、fluid_variables_container
        self.get_normal_variables_val0()
        # self.variables_vector += np.random.uniform(-4, 4, self.num_vars)
        self.check_feasibility()
        # correct the inequation constraints of mass flow、enthalpy variables
        for comp in self.fluid_comps['object']:
            comp.correct_massflow_enthalpy()
        # generate fluid data for fluid property calculation
        for c in self.fluid_conns['object']:
            c.build_fluid_data()
        # iteration start
        self.start_time = time()
        self.iterinfo_head(print_iterations=print_iterations)
        # loop calculation
        for self.iter in range(self.max_iter):
            ######## fluid iteration ######
            if self.num_fluid_vars > 0:
                fluid_iteration = True
            else:
                fluid_iteration = False
            fluid_iter = 0
            while fluid_iteration:
                self.calculate_fluid_modules()
                if norm(self.fluid_residual) < ERR ** 0.5:
                    logger.debug(f"the fluid calculation done in {fluid_iter}th iteration")
                    break
                # fluid iteration core
                try:
                    self.fluid_increment = -np.linalg.inv(self.fluid_jacobian) @ self.fluid_residual
                    self.fluid_variables_vector += self.fluid_increment
                except np.linalg.LinAlgError:
                    logger.debug(f"the fluid calculation failed due to infeasible fluid matrix")
                    self.fluid_variables_vector -= np.ones([self.num_fluid_vars])
                # update fluid composition
                self.set_fluid_variables()
                if fluid_iter > 40:
                    logger.debug(f"the fluid calculation failed due to touching max iterations")
                fluid_iter += 1
            for c in self.fluid_conns['object']:
                c.build_fluid_data()
            ###### properties iteration ######
            # calculate equations system
            self.calculate_combine_modules()
            # correct the matrix structure
            matrix_value = np.linalg.det(self.jacobian)
            logger.debug(f"matrix value: {matrix_value}")
            if abs(matrix_value) < 1e-3:
                self.repair_matrix_()
                repaired_matrix_value = np.linalg.det(self.jacobian)
                logger.debug(f"repaired matrix value: {repaired_matrix_value}")
            # algorithm core
            try:
                increment = np.linalg.inv(self.jacobian) @ self.residual
                alpha = min(1, (2 * self.num_vars * self.algo_factor / norm(increment)) ** 0.5)
                # hessian = self.jacobian.T @ self.jacobian + 0 * np.eye(self.num_vars)
                # increment = np.zeros([self.num_vars])
                # gradient = - self.jacobian.T @ self.residual
                # direction = - gradient
                # for _ in range(self.num_vars):
                #     alpha_k = ((self.residual.T @ self.jacobian @ direction - increment.T @ hessian @ direction) /
                #                (direction.T @ hessian @ direction))
                #     increment = increment + alpha_k * direction
                #     gradient = hessian @ increment - self.jacobian.T @ self.residual
                #     belta_k = ((gradient.T @ hessian @ direction) /
                #                (direction.T @ hessian @ direction))
                #     direction = - gradient + belta_k * direction
                #     if np.any(np.isnan(increment)):
                #         raise ValueError
                self.increment = -increment * alpha
            except (ValueError, np.linalg.LinAlgError) as e:
                #
                logger.debug(f"singular matrix constructure: {e}")
                self.increment = np.random.uniform(-5, 5, self.num_vars)
            self.increment_history.append(self.increment.copy())
            self.residual_history.append(self.residual.copy())
            self.norm_residual_history.append(norm(self.residual.copy()))
            if len(self.norm_residual_history) > 10:
                if ((max(self.norm_residual_history[-9:]) - min(self.norm_residual_history[-9:]))/ max(self.norm_residual_history[-9:])) < 0.01:
                    logger.debug(f"the iteration has not progressed, re_just")
                    self.increment = np.random.uniform(-5, 5, self.num_vars)
            self.variables_vector += self.increment
            self.check_feasibility()
            self.correct_attracting_basin_()
            # correct the inequation constraints of mass flow、enthalpy variables
            for comp in self.fluid_comps['object']:
                comp.correct_massflow_enthalpy()
            # set property values
            self.set_un_normal_variables(self.variables_vector)
            if self.iterinfo:
                self.iterinfo_body(print_iterations=print_iterations)
            if norm(self.residual) < ERR ** 0.5 and norm(self.fluid_residual) < ERR ** 0.5:
                self.converged = True
                logger.debug(f"has found the solution in {fluid_iter}th iteration")
                break
            if self.iter > 40:
                if (
                        all(
                            self.norm_residual_history[(self.iter - 3):] >= self.norm_residual_history[-3] * 0.999
                        ) and self.norm_residual_history[-1] >= self.norm_residual_history[-2] * 0.999
                ):
                    self.progress = False
                    break
            if self.iter == self.max_iter - 1:
                logger.warning(f"the max iteration: {self.max_iter} has reached, solution failed")
        # iteration end
        self.end_time = time()
        self.iterinfo_tail(print_iterations=print_iterations)
        self.set_un_normal_variables(self.variables_vector)
        # logger the un_converged function
        if not self.progress:
            mask_residual = abs(self.residual) > ERR
            indices_residual = np.where(mask_residual)[0]
            sorted_indices_residual = indices_residual[np.argsort(-self.residual[indices_residual])]
            un_converged_residual = [self.functions_list_visual[i] + f"--residual: {self.residual[i]}" for i in sorted_indices_residual[:5]]
            mask_increment = abs(self.increment) > ERR
            indices_increment = np.where(mask_increment)[0]
            sorted_indices_increment = indices_increment[np.argsort(-self.increment[indices_increment])]
            un_converged_increment = [self.variables_list_visual[j] + f"--increment: {self.increment[j]}" for j in sorted_indices_increment[:5]]
            msg = (f"the un_converged functions are:" + "[\n  " + ",\n  ".join(un_converged_residual) + "\n]" +
                   f"the un_converged variables are:" + "[\n  " + ",\n  ".join(un_converged_increment) + "\n]")
            logger.warning(msg)

    def iterinfo_head(self, print_iterations=True):
        """Print head of convergence progress."""
        # Start with defining the format here
        self.iterinfo_fmt = ' {iter:5s} | {residual:10s} | {progress:10s} '
        self.iterinfo_fmt += '| {massflow:10s} | {pressure:10s} | {enthalpy:10s} '
        self.iterinfo_fmt += '| {voltage:10s} | {electricity:10s} | {frequency:10s} '
        self.iterinfo_fmt += '| {fluid:10s} | {component:10s} '
        # Use the format to create the first logging entry
        msg = self.iterinfo_fmt.format(
            iter='iter',
            residual='residual',
            progress='progress',
            massflow='massflow',
            pressure='pressure',
            enthalpy='enthalpy',
            voltage='voltage',
            electricity='electricity',
            frequency='frequency',
            fluid='fluid',
            component='component'
        )
        logger.progress(0, msg)
        msg2 = '-' * 7 + '+------------' * 10

        logger.progress(0, msg2)
        if print_iterations:
            print('\n' + msg + '\n' + msg2)
        return

    def iterinfo_body(self, print_iterations=True):
        """Print convergence progress."""
        iter_str = str(self.iter + 1)
        residual_norm = norm(self.residual)  # /self.num_vars
        residual = 'NaN'
        progress = 'NaN'
        massflow = 'NaN'
        pressure = 'NaN'
        enthalpy = 'NaN'
        voltage = 'NaN'
        electricity = 'NaN'
        frequency = 'NaN'
        fluid = 'NaN'
        component = 'NaN'
        progress_val = -1
        if not np.isnan(residual_norm):
            residual = '{:.2e}'.format(residual_norm)
            if norm(self.increment):
                massflow = '{:.2e}'.format(norm(self.increment[self.m_index]))
                pressure = '{:.2e}'.format(norm(self.increment[self.p_index]))
                enthalpy = '{:.2e}'.format(norm(self.increment[self.h_index]))
                voltage = '{:.2e}'.format(norm(self.increment[self.U_index]))
                electricity = '{:.2e}'.format(norm(self.increment[self.I_index]))
                frequency = '{:.2e}'.format(norm(self.increment[self.f_index]))
                fluid = '{:.2e}'.format(norm(self.fluid_increment[self.fl_index]))
                component = '{:.2e}'.format(norm(self.increment[self.cp_index]))
            # This should not be hardcoded here.
            if residual_norm > np.finfo(float).eps * 100:
                progress_min = math.log(ERR)
                progress_max = math.log(ERR ** 0.5) * -1
                progress_val = math.log(max(residual_norm, ERR ** 0.5)) * -1
                # Scale to 0-1
                progres_scaled = (
                    (progress_val - progress_min)
                    / (progress_max - progress_min)
                )
                progress_val = max(0, min(1, progres_scaled))
                # Scale to 100%
                progress_val = int(progress_val * 100)
            else:
                progress_val = 100
            progress = '{:d} %'.format(progress_val)
        msg = self.iterinfo_fmt.format(
            iter=iter_str,
            residual=residual,
            progress=progress,
            massflow=massflow,
            pressure=pressure,
            enthalpy=enthalpy,
            voltage=voltage,
            electricity=electricity,
            frequency=frequency,
            fluid=fluid,
            component=component
        )
        logger.progress(progress_val, msg)
        if print_iterations:
            print(msg)
        return

    def iterinfo_tail(self, print_iterations=True):
        """Print tail of convergence progress."""
        num_iter = self.iter + 1
        clc_time = self.end_time - self.start_time
        num_ips = num_iter / clc_time if clc_time > 1e-10 else np.inf
        msg = '-' * 7 + '+------------' * 10
        logger.progress(100, msg)
        msg = (
            "Total iterations: {0:d}, Calculation time: {1:.2f} s, "
            "Iterations per second: {2:.2f}"
        ).format(num_iter, clc_time, num_ips)
        logger.debug(msg)
        if print_iterations:
            print(msg)
        return

    def get_normal_variables_val0(self):
        # converge the scale
        for col, data in enumerate(self.sorted_variables_objective):
            container = data['object']
            self.variables_vector[col] = container.extract_(col - container.J_col) / self.variables_scale[col]
        # fluid variables value
        for data in self.fluid_variables_module_container:
            container = data['object']
            if data["type"] == "fluid":
                self.fluid_variables_vector[container.J_col[data["fluid"]]] = container.val[data["fluid"]]

    def repair_matrix_(self):
        # implicit linear row
        linear_equations = []
        A = np.array(self.jacobian, dtype=float)
        # calculate null space
        null_basis = null_space(A.T)
        for i in range(null_basis.shape[1]):
            vec = null_basis[:, i]  # coefficient vector
            # ignore the noise
            idx_nonzero = np.where(~np.isclose(vec, 0, atol=1e-6))[0]
            if len(idx_nonzero) > 0:
                linear_coefficient_dict = {row_idx: vec[row_idx] for row_idx in idx_nonzero}
                linear_equations.append(linear_coefficient_dict)
        # several linear groups
        for group_idx, dict_ in enumerate(linear_equations):
            parameters_index_dict = {}
            parameters_blank_index_dict = {}
            all_parameter_set = set()
            # contain variables index
            for item in dict_.items():
                equation_container = self.sorted_equations_objective[item[0]]
                if hasattr(equation_container, 'func_params'):
                    kwargs_ = equation_container.func_params.copy()
                else:
                    kwargs_ = {}
                parameters_list = equation_container.variables_columns(**kwargs_)[item[0] - equation_container.row]
                parameters_index_dict[item[0]] = parameters_list
                all_parameter_set.update(parameters_list)
                blank_parameter_list = []
                for col in parameters_list:
                    if abs(self.jacobian[item[0], col]) < ERR:
                        blank_parameter_list.append(col)
                parameters_blank_index_dict[item[0]] = blank_parameter_list
            # generate dataframe of linear conditions
            all_parameter_set_list = list(sorted(all_parameter_set))
            data_index_groups = [self.wrap_label_(f'{col} ' + self.sorted_variables_objective[col]['object'].label) for col in all_parameter_set_list]
            linear_data = pd.DataFrame(columns=data_index_groups).astype('float64')
            for item in parameters_index_dict.items():
                value_list = []
                for col in all_parameter_set_list:
                    if col in parameters_blank_index_dict[item[0]]:
                        value_list.append(self.jacobian[item[0], col])
                    elif col in parameters_index_dict[item[0]]:
                        value_list.append(self.jacobian[item[0], col])
                    else:
                        value_list.append(np.nan)
                linear_data.loc[self.wrap_label_(f'{item[0]} ' + self.sorted_equations_objective[item[0]].label)] = value_list
            if len(linear_data) > 0:
                visual_linear = f"\n##### LINEAR EQUATIONS #####\n"
                visual_linear += tabulate(linear_data, headers='keys', tablefmt='grid', floatfmt='.3e')
                logger.debug(visual_linear)
            # repair matrix
            for item in parameters_blank_index_dict.items():
                if item[1]:
                    container = self.sorted_equations_objective[item[0]]
                    if hasattr(container, 'func_params'):
                        kwargs = container.func_params.copy()
                    else:
                        kwargs = {}
                    if container.num_eq > 1:
                        kwargs.update({'row': item[0] - container.row})
                    for col in item[1]:
                        property_ = self.sorted_variables_objective[col]['object']
                        kwargs_ = kwargs.copy()
                        if property_.dimension > 1:
                            kwargs_.update({'col': col - property_.J_col})
                        try:
                            self.jacobian[item[0], col] = (container.repair_matrix(property_, **kwargs_)
                                                           * self.variables_scale[col]) / self.residual_scale[item[0]]
                        except ValueError as e:
                            msg = f'Has something wrong in repairing matrix in {container.label}: {e}'
                            logger.error(msg)

    def check_feasibility(self):
        """Check constraints' feasibility."""
        variable_bounds_dict = {
            'label': str,
            'type': str,
            'value': float,
            'min value': float,
            'max value': float,
            'unit': str,
        }
        variables_bounds_data = pd.DataFrame(columns=list(variable_bounds_dict.keys())).astype(variable_bounds_dict)
        # variables bounds
        for i in range(self.num_vars):
            variable_objective_ = self.sorted_variables_objective[i]['object']
            if self.variables_vector[i] < self.lb[i]:
                variables_bounds_data.loc[f'bounds_{i}'] = [
                    self.wrap_label_(variable_objective_.label),
                    'below',
                    convert_from_SI(variable_objective_.property_data, self.variables_vector[i] * self.variables_scale[i], variable_objective_.unit),
                    convert_from_SI(variable_objective_.property_data, self.lb[i] * self.variables_scale[i], variable_objective_.unit),
                    np.nan,
                    variable_objective_.unit
                ]
                if self.sorted_variables_objective[i]['type'] == 'mass flow' and variable_objective_.design:
                    self.variables_vector[i] = variable_objective_.design / variable_objective_.scale + 1e-3
                else:
                    self.variables_vector[i] = self.lb[i] + 1e-3
            elif self.variables_vector[i] > self.ub[i]:
                variables_bounds_data.loc[f'bounds_{i}'] = [
                    self.wrap_label_(variable_objective_.label),
                    'above',
                    convert_from_SI(variable_objective_.property_data, self.variables_vector[i] * self.variables_scale[i], variable_objective_.unit),
                    np.nan,
                    convert_from_SI(variable_objective_.property_data, self.ub[i] * self.variables_scale[i], variable_objective_.unit),
                    variable_objective_.unit
                ]
                if self.sorted_variables_objective[i]['type'] == 'mass flow' and variable_objective_.design:
                    self.variables_vector[i] = variable_objective_.design / variable_objective_.scale - 1e-3
                else:
                    self.variables_vector[i] = self.ub[i] - 1e-3
        # variables constraints vector
        enthalpy_constraints_dict = {
            'label': str,
            'type': str,
            'enthalpy value': float,
            'min value': float,
            'max value': float,
            'enthalpy unit': str,
            'pressure': float,
            'pressure unit': str,
            'temperature min value': float,
            'temperature max value': float,
            'temperature unit': str,
        }
        enthalpy_constraints_data = pd.DataFrame(columns=list(enthalpy_constraints_dict.keys())).astype(enthalpy_constraints_dict)
        for conn in self.fluid_conns['object']:
            if conn.p.is_var and conn.h.is_var:
                min_enthalpy = (conn.calc_min_enthalpy(self.variables_vector[conn.p.J_col] *
                                                       self.variables_scale[conn.p.J_col])
                                                           / self.variables_scale[conn.h.J_col])
                max_enthalpy = (conn.calc_max_enthalpy(self.variables_vector[conn.p.J_col] *
                                                       self.variables_scale[conn.p.J_col])
                                                           / self.variables_scale[conn.h.J_col])
                if self.variables_vector[conn.h.J_col] < min_enthalpy:
                    enthalpy_constraints_data.loc[f'constraints_{conn.h.J_col}'] = [
                        self.wrap_label_(conn.h.label),
                        'below',
                        convert_from_SI(conn.h.property_data, self.variables_vector[conn.h.J_col] * self.variables_scale[conn.h.J_col], conn.h.unit),
                        convert_from_SI(conn.h.property_data, min_enthalpy * self.variables_scale[conn.h.J_col], conn.h.unit),
                        np.nan,
                        conn.h.unit,
                        convert_from_SI(conn.p.property_data, self.variables_vector[conn.p.J_col] * self.variables_scale[conn.p.J_col], conn.p.unit),
                        conn.p.unit,
                        convert_from_SI(conn.T.property_data, conn.T.min_val, conn.T.unit),
                        np.nan,
                        conn.T.unit,
                    ]
                    self.variables_vector[conn.h.J_col] = min_enthalpy + ERR
                elif self.variables_vector[conn.h.J_col] > max_enthalpy:
                    enthalpy_constraints_data.loc[f'constraints_{conn.h.J_col}'] = [
                        self.wrap_label_(conn.h.label),
                        'above',
                        convert_from_SI(conn.h.property_data, self.variables_vector[conn.h.J_col] * self.variables_scale[conn.h.J_col], conn.h.unit),
                        np.nan,
                        convert_from_SI(conn.h.property_data, max_enthalpy * self.variables_scale[conn.h.J_col], conn.h.unit),
                        conn.h.unit,
                        convert_from_SI(conn.p.property_data, self.variables_vector[conn.p.J_col] * self.variables_scale[conn.p.J_col], conn.p.unit),
                        conn.p.unit,
                        np.nan,
                        convert_from_SI(conn.T.property_data, conn.T.max_val, conn.T.unit),
                        conn.T.unit,
                    ]
                    self.variables_vector[conn.h.J_col] = max_enthalpy - ERR
        # log information
        variables_feasibility_information = ''
        if len(variables_bounds_data) > 0:
            variables_feasibility_information += f"\n##### VARIABLES BOUNDS #####\n"
            variables_feasibility_information += tabulate(variables_bounds_data, headers='keys', tablefmt='grid', floatfmt='.3e')
        if len(enthalpy_constraints_data) > 0:
            variables_feasibility_information += f"\n##### VARIABLES CONSTRAINTS #####\n"
            variables_feasibility_information += tabulate(enthalpy_constraints_data, headers='keys', tablefmt='grid', floatfmt='.3e')
        if len(variables_bounds_data) > 0 or len(enthalpy_constraints_data) > 0:
            logger.debug(variables_feasibility_information)

    def correct_attracting_basin_(self):
        """
        Check weather be attracted by another attraction basin.
        """
        for comp in self.comps['object']:
            comp.correct_attracting_basin_path_()

    def calculate_fluid_modules(self):
        """
        Calculate residual、jacobin matrix fluid modules

        Parameters
        ----------

        Returns
        -------

        """
        # set fluid composition values for fluid variables
        self.set_fluid_variables()
        self.fluid_jacobian = self.fluid_jacobian_temp.copy()
        fluid_func_index = 0
        while True:
            if fluid_func_index >= self.num_fluid_eqs:
                if fluid_func_index == self.num_fluid_eqs:
                    break
                else:
                    raise ValueError(f"Fluid function index out of range: {fluid_func_index} > {self.num_fluid_eqs}")
            # equations module objective
            fluid_equations_module = self.sorted_fluid_equations_objective[fluid_func_index]
            if hasattr(fluid_equations_module, 'func_params'):
                kwargs = fluid_equations_module.func_params.copy()
            else:
                kwargs = {}
            # filte variables for derive solvent
            increment_filter = np.absolute(np.ones([self.num_fluid_vars])) < ERR ** 2
            try:
                self.fluid_residual[fluid_func_index: fluid_func_index + fluid_equations_module.num_eq] = fluid_equations_module.func(**kwargs)
                fluid_equations_module.deriv(increment_filter, fluid_func_index, **kwargs)
            except ValueError as e:
                msg = f'the {fluid_func_index}th fluid equation {fluid_equations_module.label} raise: {e}'
                logger.error(msg)
            fluid_func_index += fluid_equations_module.num_eq

    def calculate_combine_modules(self):
        self.jacobian = self.jacobian_temp.copy()
        # self.increment_filter = np.absolute(self.increment) < ERR ** 2  # bool array to ignore tiny variables
        self.set_un_normal_variables(self.variables_vector)
        # solve
        func_index = 0
        while True:
            if func_index >= self.num_vars:
                if func_index == self.num_vars:
                    break
                else:
                    raise ValueError('Function index out of range')

            equation_container = self.sorted_equations_objective[func_index]
            if hasattr(equation_container, 'func_params'):
                kwargs = equation_container.func_params.copy()
            else:
                kwargs = {}

            self.increment_filter = np.absolute(np.ones([self.num_vars])) < ERR ** 2
            # calculate the residual、jacobian、tensor of function group
            try:
                self.residual[func_index: func_index + equation_container.num_eq] = equation_container.func(**kwargs)
                # if not (hasattr(equation_container, 'constant_deriv') and equation_container.constant_deriv):
                #     equation_container.deriv(self.increment_filter, func_index, **kwargs)
                equation_container.deriv(self.increment_filter, func_index, **kwargs)
            except ValueError as err:
                msg = f'the {func_index}th equation {equation_container.label}  raise: {err}'
                logger.error(msg)
                raise ValueError(msg)
            func_index += equation_container.num_eq

        def normalize_residual_scale(scale):
            return 1 / scale

        vectorized_normalize_residual_scale = np.vectorize(normalize_residual_scale)
        residual_scale_factor = vectorized_normalize_residual_scale(self.residual_scale)  # the scale factor of residual
        self.jacobian = np.einsum('mn,m,n->mn', self.jacobian, residual_scale_factor, self.variables_scale)
        self.residual = np.einsum('i,i->i', self.residual, residual_scale_factor)

    def set_un_normal_variables(self, variables):
        r"""

        Parameters
        ----------
        variables: list (understand normal)

        Returns
        -------

        """
        # converge the scale
        for index, data in enumerate(self.sorted_variables_objective):
            container = data['object']
            container.vest_(index - container.J_col, variables[index] * self.variables_scale[index])

    def set_fluid_variables(self):
        # fluid value set
        for data in self.fluid_variables_module_container:
            container = data['object']
            if data["type"] == "fluid":
                container.val[data["fluid"]] = self.fluid_variables_vector[container.J_col[data["fluid"]]]
                if container.val[data["fluid"]] < ERR:
                    container.val[data["fluid"]] = 0
                    self.fluid_variables_vector[container.J_col[data["fluid"]]] = 0
                elif container.val[data["fluid"]] > 1 - ERR:
                    container.val[data["fluid"]] = 1
                    self.fluid_variables_vector[container.J_col[data["fluid"]]] = 1

    def postprocess_properties_(self):
        r"""Calculate connection, bus and component properties."""
        self.postprocess_connections_()
        self.postprocess_components_()
        self.postprocess_busses_()
        self.postprocess_combines_()
        if self.converged:
            self.postprocess_nodes_()
        msg = 'Postprocessing complete.'
        logger.info(msg)

    def postprocess_connections_(self):
        """Process the Connection results."""
        for c in self.conns['object']:
            c.good_starting_values = True
            c.calc_results()
            prop_data = {}
            for pro in self.connections_properties_data[c.connection_type()]:
                prop_data[pro] = c.get_attr(pro).val
                prop_data[f'{pro}_unit'] = c.get_attr(pro).unit
            if c.connection_type() == 'fluid':
                prop_data['phase'] = c.phase.val
                for fluid in self.all_fluids:
                    if fluid in c.fluid.val:
                        prop_data[fluid] = c.fluid.val[fluid]
                    else:
                        prop_data[fluid] = np.nan
            self.results[c.__class__.__name__].loc[c.label] = prop_data

    def postprocess_components_(self):
        """Process the component results."""
        # components
        for cp in self.comps['object']:
            try:
                cp.calc_parameters()
            except (ZeroDivisionError, ValueError) as e:
                msg = f'Has something wrong in calculating parameters at {cp.__class__.__name__}: {cp.label}'
                logger.error(msg)
                raise ValueError(msg)
            cp.check_parameter_bounds()
            cp.convert_set_property_from_criterion()
            key = cp.__class__.__name__
            for param in cp.parameters.keys():
                p = cp.get_attr(param)
                if isinstance(p, dc_cp):
                    if (p.func is not None or (p.func is None and p.is_set) or
                            p.is_result):
                        self.results[key].loc[cp.label, param] = p.val
                        self.results[key].loc[cp.label, f'{param}_unit'] = p.unit
                    else:
                        self.results[key].loc[cp.label, param] = np.nan
                        self.results[key].loc[cp.label, f'{param}_unit'] = p.unit

    def postprocess_busses_(self):
        """Process the bus results."""
        # busses
        for b in self.busses.values():
            for cp in b.comps.index:
                # get components bus func value
                bus_val = cp.calc_bus_value(b)
                eff = cp.calc_bus_efficiency(b)
                cmp_val = cp.bus_func(b.comps.loc[cp])
                b.comps.loc[cp, 'char'].get_domain_errors(
                    cp.calc_bus_expr(b), cp.label)
                # save as reference value
                if self.mode == 'design':
                    if b.comps.loc[cp, 'base'] == 'component':
                        design_value = cmp_val
                    else:
                        design_value = bus_val
                    b.comps.loc[cp, 'P_ref'] = design_value
                else:
                    design_value = b.comps.loc[cp, 'P_ref']
                result = [cmp_val, bus_val, eff, design_value, b.P.SI_unit]
                self.results[b.label].loc[cp.label] = result
            b.P.val_SI = float(self.results[b.label]['bus value'].sum())
            b.P.val = hlp.convert_from_SI(b.P.property_data, b.P.val_SI, b.P.unit)

    def postprocess_combines_(self):
        for comb in self.combs['object']:
            pass

    def postprocess_nodes_(self):
        for comp in self.differ_comps['object']:
            comp.calc_nodes_properties()
            for row_index in comp.nodes.index:
                for key in comp.nodes_properties[row_index]['node_properties']:
                    row_name = comp.nodes_properties[row_index]['side_name']
                    property_name = dict(itertools.chain(fpd.items(), epd.items(), cpd.items(), stpd.items(), mapd.items()))[key]['text']
                    units = set()
                    value_list = []
                    for col_index in range(comp.nodes_num):
                        node = comp.nodes.loc[row_index, col_index]
                        value_list.append(node.get_attr(key).val)
                        if hasattr(node.get_attr(key), 'unit') and node.get_attr(key).unit:
                            units.add(node.get_attr(key).unit)
                    if len(units) > 1:
                        msg = f'Multiple units:{units} have been set at nodes of {row_name} in {comp.__class__.__name__}: {comp.label}.'
                        logger.error(msg)
                        raise AttributeError(msg)
                    if units:
                        unit = list(units)[0]
                    else:
                        unit = ''
                    self.nodes_results[f'{comp.__class__.__name__}: {comp.label}'].loc[f'{row_name}---{property_name} ({unit})'] = value_list

    def print_results(self, colored=True, colors=None, print_results=True):
        r"""Print the calculations results to prompt."""
        # Define colors for highlighting values in result table
        if colors is None:
            colors = {}
        result = ""
        result_log = ""
        coloring = {
            'end': '\033[0m',
            'set': '\033[94m',
            'err': '\033[31m',
            'var': '\033[32m'
        }
        coloring.update(colors)
        if not hasattr(self, 'results'):
            msg = (
                'It is not possible to print the results of a network, that '
                'has never been solved successfully. Results DataFrames are '
                'only available after a full simulation run is performed.')
            raise hlp.AURORANetworkError(msg)
        # component properties
        for cpcl in self.comps['comp_type'].unique():
            df = self.results[cpcl].copy()
            df_log = df.copy()
            # are there any parameters to print?
            if df.size > 0 and len(df.columns) > 0:
                for comp_lb in df.index:
                    comp = self.get_comp(comp_lb)
                    if not comp.printout:
                        df.drop([comp_lb], axis=0, inplace=True)
                    elif colored:
                        for prop in [prop for prop, data in comp.parameters.items() if isinstance(data, dc_cp)]:
                            if (comp.get_attr(prop).val_SI < comp.get_attr(prop).min_val - ERR or
                                comp.get_attr(prop).val_SI > comp.get_attr(prop).max_val + ERR):
                                df.loc[comp_lb, prop] = f"{coloring['err']} {comp.get_attr(prop).val} {coloring['end']}"
                            else:
                                if self.specifications[cpcl]['variables'].loc[comp_lb, prop]:
                                    df.loc[comp_lb, prop] = f"{coloring['var']} {comp.get_attr(prop).val} {coloring['end']}"
                                elif self.specifications[cpcl]['properties'].loc[comp_lb, prop]:
                                    df.loc[comp_lb, prop] = f"{coloring['set']} {comp.get_attr(prop).val} {coloring['end']}"
                # df.dropna(how='all', inplace=True)
                if len(df) > 0:
                    # printout with tabulate
                    result += f"\n##### RESULTS ({cpcl}) #####\n"
                    result += tabulate(df, headers='keys', tablefmt='psql',floatfmt='.2e')
                    result_log += f"\n##### RESULTS ({cpcl}) #####\n"
                    result_log += tabulate(df_log, headers='keys', tablefmt='psql',floatfmt='.2e')
        # connection properties
        for conl in self.conns['conn_type'].unique():
            df = self.results[conl].copy()
            df = df.astype(str)
            df_log = df.copy()
            for c_lb in df.index:
                conn = self.get_conn(c_lb)
                if not conn.printout:
                    df.drop([c_lb], axis=0, inplace=True)
                elif colored:
                    for col in self.connections_properties_data[conn.connection_type()]:
                        if conn.get_attr(col).is_set:
                            df.loc[c_lb, col] = (
                                    coloring['set'] + str(conn.get_attr(col).val) +
                                    coloring['end'])
            if len(df) > 0:
                result += f'\n##### RESULTS ({conl}) #####\n'
                result += tabulate(df, headers='keys', tablefmt='psql', floatfmt='.3e')
                result_log += f'\n##### RESULTS ({conl}) #####\n'
                result_log += tabulate(df_log, headers='keys', tablefmt='psql', floatfmt='.3e')
        # bus properties
        for b in self.busses.values():
            if b.printout:
                df = self.results[b.label].loc[
                    :, ['component value', 'bus value', 'efficiency']
                ].copy()
                df.loc['total'] = df.sum()
                df.loc['total', 'efficiency'] = np.nan
                df_log = df.copy()
                if colored:
                    df["bus value"] = df["bus value"].astype(str)
                    if b.P.is_set:
                        df.loc['total', 'bus value'] = (
                            coloring['set'] + str(df.loc['total', 'bus value']) +
                            coloring['end']
                        )
                result += f"\n##### RESULTS (Bus: {b.label}) #####\n"
                result += tabulate(df, headers='keys', tablefmt='psql', floatfmt='.3e')
                result_log += f"\n##### RESULTS (Bus: {b.label}) #####\n"
                result_log += tabulate(df_log, headers='keys', tablefmt='psql', floatfmt='.3e')
        # nodes information of components
        for comp in self.differ_comps['object']:
            info_label = f'{comp.__class__.__name__}: {comp.label}'
            if comp.print_nodes and self.converged:
                df = self.nodes_results[info_label].copy()
                df_log = df.copy()
                if colored:
                    pass
                result += f"\n##### RESULTS (Nodes: {info_label}) #####\n"
                result += tabulate(df, headers='keys', tablefmt='psql', floatfmt='.3e')
                result_log += f"\n##### RESULTS (Nodes: {info_label}) #####\n"
                result_log += tabulate(df_log, headers='keys', tablefmt='psql', floatfmt='.3e')
        # log and print out
        if len(str(result)) > 0:
            logger.result(result_log)
            if print_results:
                print(result)
        return

    def save(self, json_file_path):
        r"""
        Dump the results to a json style output.

        Parameters
        ----------
        json_file_path : str
            Filename to dump results into.

        Note
        ----
        Results will be saved to specified file path
        """
        dump = {}
        # save relevant state information only
        dump["Connection"] = self._save_connections()
        dump["Component"] = self._save_components()
        dump["Bus"] = self._save_busses()
        dump = self._nested_dict_of_dataframes_to_dict(dump)
        with open(json_file_path, "w") as f:
            json.dump(dump, f)

    def save_csv(self, folder_path):
        """Export the results in multiple csv files in a folder structure

        - Connection.csv
        - Component/
          - Compressor.csv
          - ....
        - Bus/
          - power input bus.csv
          - ...

        Parameters
        ----------
        folder_path : str
            Path to dump results to
        """
        dump = {}
        # save relevant state information only
        dump["Connection"] = self._save_connections()
        dump["Component"] = self._save_components()
        dump["Bus"] = self._save_busses()
        dump['Node'] = self._save_nodes_()
        self._nested_dict_of_dataframes_to_csv(dump, folder_path)

    def _nested_dict_of_dataframes_to_csv(self, dictionary, basepath):
        """Dump a nested dict with dataframes into a folder structrue

        The upper level keys with subdictionaries are folder names, the lower
        level keys (where a dataframe is the value) will be the names of the
        csv files.

        Parameters
        ----------
        dictionary : dict
            Nested dictionary to write to filesystem.
        basepath : str
            path to dump data to
        """
        os.makedirs(basepath, exist_ok=True)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # create sub path
                subpath = os.path.join(basepath, key)
                self._nested_dict_of_dataframes_to_csv(value, subpath)
            else:
                # save to basepath
                value.to_csv(os.path.join(basepath, f"{key}.csv"))

    def _nested_dict_of_dataframes_to_dict(self, dictionary):
        """Transpose a nested dict with dataframes in a json style dict

        Parameters
        ----------
        dictionary : dict
            Dictionary of dataframes

        Returns
        -------
        dict
            json style dictionary containing all data from the dataframes
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                dictionary[key] = self._nested_dict_of_dataframes_to_dict(value)
            else:
                # Series to csv does not have orient
                kwargs = {}
                if isinstance(value, pd.DataFrame):
                    kwargs = {"orient": "index"}
                dictionary[key] = value.to_dict(**kwargs)
        return dictionary

    def _save_connections(self):
        """Save the connection properties.

        Returns
        -------
        pandas.DataFrame
            pandas.Dataframe of the connection results
        """
        dump = {}
        for con_cl in self.conns['conn_type'].unique():
            dump[con_cl] = self.results[con_cl].replace(np.nan, None)
        return dump

    def _save_components(self):
        r"""
        Save the component properties.

        Returns
        -------
        dump : dict
            Dump of the component information.
        """
        dump = {}
        for comp_cl in self.comps['comp_type'].unique():
            dump[comp_cl] = self.results[comp_cl].replace(np.nan, None)
        return dump

    def _save_busses(self):
        r"""
        Save the bus properties.

        Returns
        -------
        dump : dict
            Dump of the component information.
        """
        dump = {}
        for label in self.busses:
            dump[label] = self.results[label]["design value"].replace(np.nan, None)
        return dump

    def _save_nodes_(self):
        """
        Save the node properties in component.

        Returns
        -------
        dump : dict
            Dump of the component information.
        """
        dump = {}
        for comp in self.differ_comps['object']:
            info_label = f'{comp.__class__.__name__}: {comp.label}'
            if self.converged:
                dump[f'{comp.__class__.__name__}-{comp.label}'] = self.nodes_results[info_label].replace(np.nan, None)
        return dump

    @classmethod
    def from_json(cls, json_file_path):
        r"""
        Load a network from a base path.

        Parameters
        ----------
        path : str
            The path to the network data.

        Returns
        -------
        nw : tespy.networks.network.Network
            TESPy networks object.

        Note
        ----
        If you export the network structure of an existing TESPy network, it will be
        saved to the path you specified. The structure of the saved data in that
        path is the structure you need to provide in the path for loading the
        network.

        The structure of the path must be as follows:

        - Folder: path (e.g. 'mynetwork')
        - Component.json
        - Connection.json
        - Bus.json
        - Network.json

        Example
        -------
        Create a network and export it. This is followed by loading the network
        with the network_reader module. All network information stored will be
        passed to a new network object. Components, connections and busses will
        be accessible by label. The following example setup is simple gas turbine
        setup with compressor, combustion chamber and turbine. The fuel is fed
        from a pipeline and throttled to the required pressure while keeping the
        temperature at a constant value.

        >>> from Aurora.components import (Sink, Source, CombustionChamber,
        ... Compressor, Turbine, SimpleHeatExchanger)
        >>> from Aurora.connections import Connection, Ref, Bus
        >>> from Aurora.networks import Network
        >>> import shutil
        >>> nw = Network(p_unit='bar', T_unit='C', h_unit='kJ / kg', iterinfo=False)
        >>> air = Source('air')
        >>> f = Source('fuel')
        >>> c = Compressor('compressor')
        >>> comb = CombustionChamber('combustion')
        >>> t = Turbine('turbine')
        >>> p = SimpleHeatExchanger('fuel preheater')
        >>> si = Sink('sink')
        >>> inc = Connection(air, 'out1', c, 'in1', label='ambient air')
        >>> cc = Connection(c, 'out1', comb, 'in1')
        >>> fp = Connection(f, 'out1', p, 'in1')
        >>> pc = Connection(p, 'out1', comb, 'in2')
        >>> ct = Connection(comb, 'out1', t, 'in1')
        >>> outg = Connection(t, 'out1', si, 'in1')
        >>> nw.add_conns(inc, cc, fp, pc, ct, outg)

        Specify component and connection properties. The intlet pressure at the
        compressor and the outlet pressure after the turbine are identical. For the
        compressor, the pressure ratio and isentropic efficiency are design
        parameters. A compressor map (efficiency vs. mass flow and pressure rise
        vs. mass flow) is selected for the compressor. Fuel is Methane.

        >>> c.set_attr(pr=10, eta_s=0.88, design=['eta_s', 'pr'],
        ... offdesign=['char_map_eta_s', 'char_map_pr'])
        >>> t.set_attr(eta_s=0.9, design=['eta_s'],
        ... offdesign=['eta_s_char', 'cone'])
        >>> comb.set_attr(lamb=2)
        >>> inc.set_attr(fluid={'N2': 0.7556, 'O2': 0.2315, 'Ar': 0.0129}, T=25, p=1)
        >>> fp.set_attr(fluid={'CH4': 0.96, 'CO2': 0.04}, T=25, p=40)
        >>> pc.set_attr(T=25)
        >>> outg.set_attr(p=Ref(inc, 1, 0))
        >>> power = Bus('total power output')
        >>> power.add_comps({"comp": c, "base": "bus"}, {"comp": t})
        >>> nw.add_busses(power)

        For a stable start, we specify the fresh air mass flow.

        >>> inc.set_attr(m=3)
        >>> nw.solve('design')

        The total power output is set to 1 MW, electrical or mechanical
        efficiencies are not considered in this example. The documentation
        example in class :py:class:`tespy.connections.bus.Bus` provides more
        information on efficiencies of generators, for instance.

        >>> comb.set_attr(lamb=None)
        >>> ct.set_attr(T=1100)
        >>> inc.set_attr(m=None)
        >>> power.set_attr(P=-1e6)
        >>> nw.solve('design')
        >>> nw.lin_dep
        False
        >>> nw.save('design_state.json')
        >>> _ = nw.export('exported_nwk.json')
        >>> mass_flow = round(nw.get_conn('ambient air').m.val_SI, 1)
        >>> c.set_attr(igva='var')
        >>> nw.solve('offdesign', design_path='design_state.json')
        >>> round(t.eta_s.val, 1)
        0.9
        >>> power.set_attr(P=-0.75e6)
        >>> nw.solve('offdesign', design_path='design_state.json')
        >>> nw.lin_dep
        False
        >>> eta_s_t = round(t.eta_s.val, 3)
        >>> igva = round(c.igva.val, 3)
        >>> eta_s_t
        0.898
        >>> igva
        20.138

        The designed network is exported to the path 'exported_nwk'. Now import the
        network and recalculate. Check if the results match with the previous
        calculation in design and offdesign case.

        >>> imported_nwk = Network.from_json('exported_nwk.json')
        >>> imported_nwk.set_attr(iterinfo=False)
        >>> imported_nwk.solve('design')
        >>> imported_nwk.lin_dep
        False
        >>> round(imported_nwk.get_conn('ambient air').m.val_SI, 1) == mass_flow
        True
        >>> round(imported_nwk.get_comp('turbine').eta_s.val, 3)
        0.9
        >>> imported_nwk.get_comp('compressor').set_attr(igva='var')
        >>> imported_nwk.solve('offdesign', design_path='design_state.json')
        >>> round(imported_nwk.get_comp('turbine').eta_s.val, 3)
        0.9
        >>> imported_nwk.busses['total power output'].set_attr(P=-0.75e6)
        >>> imported_nwk.solve('offdesign', design_path='design_state.json')
        >>> round(imported_nwk.get_comp('turbine').eta_s.val, 3) == eta_s_t
        True
        >>> round(imported_nwk.get_comp('compressor').igva.val, 3) == igva
        True
        >>> shutil.rmtree('./exported_nwk', ignore_errors=True)
        >>> shutil.rmtree('./design_state', ignore_errors=True)
        """
        msg = f'Reading network data from base path {json_file_path}.'
        logger.info(msg)

        # load components
        comps = {}

        module_name = "Aurora.components"
        _ = importlib.import_module(module_name)

        with open(json_file_path, "r") as f:
            network_data = json.load(f)

        for component, data in network_data["Component"].items():
            if component not in component_registry.items:
                msg = (
                    f"A class {component} is not available through the "
                    "Aurora.components.component.component_registry decorator. "
                    "If you are using a custom component make sure to "
                    "decorate the class."
                )
                logger.error(msg)
                raise hlp.AURORANetworkError(msg)

            target_class = component_registry.items[component]
            comps.update(_construct_components(target_class, data))

        msg = 'Created network components.'
        logger.info(msg)

        # create network
        nw = cls(**network_data["Network"])

        # load connections
        conns = _construct_connections(network_data["Connection"], comps)

        # add connections to network
        for c in conns.values():
            nw.add_conns(c)

        msg = 'Created connections.'
        logger.info(msg)

        # load busses
        data = network_data["Bus"]
        if len(data) > 0:
            busses = _construct_busses(data, comps)
            # add busses to network
            for b in busses.values():
                nw.add_busses(b)

            msg = 'Created busses.'
            logger.info(msg)

        else:
            msg = 'No bus data found!'
            logger.debug(msg)

        msg = 'Created network.'
        logger.info(msg)

        nw.check_network()

        return nw

    def export(self, json_file_path=None):
        """Export the parametrization and structure of the Network instance

        Parameters
        ----------
        json_file_path : str, optional
            Path for exporting to filesystem. If path is None, the data are
            only returned and not written to the filesystem, by default None.

        Returns
        -------
        dict
            Parametrization and structure of the Network instance.
        """
        export = {}
        export["Network"] = self._export_network()
        export["Connection"] = self._export_connections()
        export["Component"] = self._export_components()
        export["Bus"] = self._export_busses()

        if json_file_path:
            with open(json_file_path, "w") as f:
                json.dump(export, f, indent=2)

            logger.debug(f'Model information saved to {json_file_path}.')

        return export

    def to_exerpy(self, Tamb, pamb, exerpy_mappings):
        """Export the network to exerpy

        Parameters
        ----------
        Tamb : float
            Ambient temperature.
        pamb : float
            Ambient pressure.
        exerpy_mappings : dict
            Mappings for Aurora components to exerpy components

        Returns
        -------
        dict
            exerpy compatible input dictionary
        """
        component_json = {}
        for comp_type in self.comps["comp_type"].unique():
            if comp_type not in exerpy_mappings.keys():
                msg = f"Component class {comp_type} not available in exerpy."
                logger.warning(msg)
                continue

            key = exerpy_mappings[comp_type]
            if key not in component_json:
                component_json[key] = {}

            for c in self.comps.loc[self.comps["comp_type"] == comp_type, "object"]:
                component_json[key][c.label] = {
                    "name": c.label,
                    "type": comp_type
                }

        connection_json = {}
        for c in self.conns["object"]:
            c.get_physical_exergy(pamb, Tamb)

            connection_json[c.label] = {
                "source_component": c.source.label,
                "source_connector": int(c.source_id.removeprefix("out")) - 1,
                "target_component": c.target.label,
                "target_connector": int(c.target_id.removeprefix("in")) - 1
            }
            connection_json[c.label].update({f"mass_composition": c.fluid.val})
            connection_json[c.label].update({"kind": "material"})
            for param in ["m", "T", "p", "h", "s"]:
                connection_json[c.label].update({
                    param: c.get_attr(param).val_SI,
                    f"{param}_unit": c.get_attr(param).unit
                })
            connection_json[c.label].update(
                {"e_T": c.ex_therm, "e_M": c.ex_mech, "e_PH": c.ex_physical}
            )

        from Aurora.components.fluid_components.turbomachinery.base import Turbomachine
        for label, bus in self.busses.items():

            if "Motor" not in component_json:
                component_json["Motor"] = {}
            if "Generator" not in component_json:
                component_json["Generator"] = {}

            for i, (idx, row) in enumerate(bus.comps.iterrows()):
                if isinstance(idx, Turbomachine):
                    kind = "power"
                else:
                    kind = "heat"

                if row["base"] == "component":
                    component_label = f"generator_of_{idx.label}"
                    connection_label = f"{idx.label}__{component_label}"
                    connection_json[connection_label] = {
                        "source_component": idx.label,
                        "source_connector": 999,
                        "target_component": component_label,
                        "target_connector": 0,
                        "mass_composition": None,
                        "kind": kind,
                        "energy_flow": abs(idx.bus_func(bus))
                    }
                    connection_label = f"{component_label}__{label}"
                    connection_json[connection_label] = {
                        "source_component": component_label,
                        "source_connector": 0,
                        "target_component": label,
                        "target_connector": i,
                        "mass_composition": None,
                        "kind": kind,
                        "energy_flow": abs(idx.calc_bus_value(bus))
                    }
                    component_json["Generator"][component_label] = {
                        "name": component_label,
                        "type": "Generator",
                        "type_index": None,
                    }

                else:
                    component_label = f"motor_of_{idx.label}"
                    connection_label = f"{label}__{component_label}"
                    connection_json[connection_label] = {
                        "source_component": label,
                        "source_connector": i,
                        "target_component": component_label,
                        "target_connector": 0,
                        "mass_composition": None,
                        "kind": kind,
                        "energy_flow": idx.calc_bus_value(bus)
                    }
                    connection_label = f"{component_label}__{idx.label}"
                    connection_json[connection_label] = {
                        "source_component": component_label,
                        "source_connector": 0,
                        "target_component": idx.label,
                        "target_connector": 999,
                        "mass_composition": None,
                        "kind": kind,
                        "energy_flow": idx.bus_func(bus)
                    }
                    component_json["Motor"][component_label] = {
                        "name": component_label,
                        "type": "Motor",
                        "type_index": None,
                    }

        return {
            "components": component_json,
            "connections": connection_json,
            "ambient_conditions": {
                "Tamb": Tamb,
                "Tamb_unit": "K",
                "pamb": pamb,
                "pamb_unit": "Pa"
            }
        }

    def _export_network(self):
        r"""Export network information

        Returns
        -------
        dict
            Serialization of network object.
        """
        return self._serialize()

    def _export_connections(self):
        """Export connection information

        Returns
        -------
        dict
            Serialization of connection objects.
        """
        connections = {}
        for c in self.conns["object"]:
            connections.update(c._serialize())
        return connections

    def _export_components(self):
        """Export component information

        Returns
        -------
        dict
            Dict of dicts with per class serialization of component objects.
        """
        components = {}
        for c in self.comps["comp_type"].unique():
            components[c] = {}
            for cp in self.comps.loc[self.comps["comp_type"] == c, "object"]:
                components[c].update(cp._serialize())

        return components

    def _export_busses(self):
        """Export bus information

        Returns
        -------
        dict
            Serialization of bus objects.
        """
        busses = {}
        for bus in self.busses.values():
            busses.update(bus._serialize())

        return busses


def _construct_components(target_class, data):
    r"""
    Create TESPy component from class name and set parameters.

    Parameters
    ----------
    component : str
        Name of the component class to be constructed.

    data : dict
        Dictionary with component information.

    Returns
    -------
    dict
        Dictionary of all components of the specified type.
    """
    instances = {}
    for cp, cp_data in data.items():
        instances[cp] = target_class(cp)
        for param, param_data in cp_data.items():
            container = instances[cp].get_attr(param)
            if isinstance(container, dc):
                if "char_func" in param_data:
                    if isinstance(container, dc_cc):
                        param_data["char_func"] = CharLine(**param_data["char_func"])
                    elif isinstance(container, dc_cm):
                        param_data["char_func"] = CharMap(**param_data["char_func"])
                if isinstance(container, dc_prop):
                    param_data["val0"] = param_data["val"]
                container.set_attr(**param_data)
            else:
                instances[cp].set_attr(**{param: param_data})

    return instances


def _construct_connections(data, comps):
    r"""
    Create TESPy connection from data in the .json-file and its parameters.

    Parameters
    ----------
    data : dict
        Dictionary with connection data.

    comps : dict
        Dictionary of constructed components.

    Returns
    -------
    dict
        Dictionary of TESPy connection objects.
    """
    conns = {}

    arglist = [
        _ for _ in data[list(data.keys())[0]]
        if _ not in ["source", "source_id", "target", "target_id", "label", "fluid"]
        and "ref" not in _
    ]
    arglist_ref = [_ for _ in data[list(data.keys())[0]] if "ref" in _]

    module_name = "Aurora.tools.fluid_properties.wrappers"
    _ = importlib.import_module(module_name)

    for label, conn in data.items():
        conns[label] = Connection(
            comps[conn["source"]], conn["source_id"],
            comps[conn["target"]], conn["target_id"],
            label=label
        )
        for arg in arglist:
            container = conns[label].get_attr(arg)
            if isinstance(container, dc):
                container.set_attr(**conn[arg])
            else:
                conns[label].set_attr(**{arg: conn[arg]})

        for f, engine in conn["fluid"]["engine"].items():
            conn["fluid"]["engine"][f] = wrapper_registry.items[engine]

        conns[label].fluid.set_attr(**conn["fluid"])
        conns[label]._create_fluid_wrapper()

    for label, conn in data.items():
        for arg in arglist_ref:
            if len(conn[arg]) > 0:
                param = arg.replace("_ref", "")
                ref = Ref(
                    conns[conn[arg]["conn"]],
                    conn[arg]["factor"],
                    conn[arg]["delta"]
                )
                conns[label].set_attr(**{param: ref})

    return conns


def _construct_busses(data, comps):
    r"""
    Create busses of the network.

    Parameters
    ----------
    data : dict
        Bus information from .json file.

    comps : dict
        TESPy components dictionary.

    Returns
    -------
    dict
        Dict with TESPy bus objects.
    """
    busses = {}

    for label, bus_data in data.items():
        busses[label] = Bus(label)
        busses[label].P.set_attr(**bus_data["P"])

        components = [_ for _ in bus_data if _ != "P"]
        for cp in components:
            char = CharLine(**bus_data[cp]["char"])
            component_data = {
                "comp": comps[cp], "param": bus_data[cp]["param"],
                "base": bus_data[cp]["base"], "char": char
            }
            busses[label].add_comps(component_data)

    return busses
