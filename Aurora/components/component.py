# -*- coding: utf-8

"""Module class component.
"""

import math

import numpy as np
import pandas as pd

from Aurora.tools import logger
from Aurora.tools.characteristics import CharLine
from Aurora.tools.characteristics import CharMap
from Aurora.tools.characteristics import load_default_char as ldc
from Aurora.tools.data_containers import ComponentCharacteristicMaps as dc_cm
from Aurora.tools.data_containers import ComponentCharacteristics as dc_cc
from Aurora.tools.data_containers import ComponentProperties as dc_cp
from Aurora.tools.data_containers import GroupedComponentCharacteristics as dc_gcc
from Aurora.tools.data_containers import GroupedComponentProperties as dc_gcp
from Aurora.tools.data_containers import SimpleDataContainer as dc_simple
from Aurora.tools.data_containers import FitCoefficient as dc_fit
from Aurora.tools.data_containers import Constraints as dc_cons
from Aurora.tools.document_models import generate_latex_eq

from Aurora.tools.global_vars import ERR
from Aurora.tools.helpers import bus_char_derivative
from Aurora.tools.helpers import bus_char_evaluation
from Aurora.tools.helpers import newton_with_kwargs
from Aurora.tools.helpers import convert_to_SI
from Aurora.tools.helpers import convert_from_SI


def component_registry(type):
    component_registry.items[type.__name__] = type
    return type


component_registry.items = {}


@component_registry
class Component:
    r"""
    Class Component is the base class of all AURORA components.

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

    **kwargs :
        See the class documentation of desired component for available
        keywords.
    """

    def __init__(self, label, nodes_num=0, **kwargs):

        # check if components label is of type str and for prohibited chars
        _forbidden = [';', ',', '.']
        if not isinstance(label, str):
            msg = 'Component label must be of type str!'
            logger.error(msg)
            raise ValueError(msg)

        elif any([True for x in _forbidden if x in label]):
            msg = (
                f"You cannot use any of " + ", ".join(_forbidden) + " in a "
                f"component label ({self.component()}"
            )
            logger.error(msg)
            raise ValueError(msg)

        else:
            self.label = label

        # defaults
        self.new_design = True  # load another design mode of single connection as reference
        self.design_path = None
        self.design = []
        self.offdesign = []
        self.local_design = False
        self.local_offdesign = False
        self.char_warnings = True
        self.printout = True
        self.print_nodes = True
        self.bypass = False
        self.boundary_rectify = False
        self.fkt_group = self.label  #
        self.nodes_num = self.correct_nodes_num(nodes_num)
        self.nodes_properties = {}
        self.nodes = pd.DataFrame(columns=list(range(self.nodes_num))).astype(object)

        # add container for components attributes
        self.parameters = self.get_parameters().copy()
        self.__dict__.update(self.parameters)  # add property of comp instance
        self.set_attr(**kwargs)  # update the value of property

    @classmethod
    def correct_nodes_num(cls, num):
        if num % 2 != 0:
            return num + 1
        return num

    def set_attr(self, **kwargs):
        r"""
        Set, reset or unset attributes of a component for provided arguments.

        Parameters
        ----------
        design : list
            List containing design parameters (stated as String).

        offdesign : list
            List containing offdesign parameters (stated as String).

        design_path: str
            Path to the components design case.

        **kwargs :
            See the class documentation of desired component for available
            keywords.

        Note
        ----
        Allowed keywords in kwargs are obtained from class documentation as all
        components share the
        :py:meth:`Aurora.components.component.Component.set_attr` method.
        """
        # set specified values
        for key in kwargs:
            if key in self.parameters:  # determine whether the key is condition or parameters
                data = self.get_attr(key)
                if kwargs[key] is None:  # unset the value of param
                    data.set_attr(is_set=False)
                    try:
                        data.set_attr(is_var=False)  # is_var may not exist
                    except KeyError:
                        pass
                    continue
                #
                try:
                    float(kwargs[key])  # judge the type of value, dict or not
                    is_numeric = True
                except (TypeError, ValueError):
                    is_numeric = False
                # dict specification
                if (isinstance(kwargs[key], dict) and
                        not isinstance(data, dc_fit) and
                        not isinstance(data, dc_simple)):  # the dict type of param,just like fluids,charline
                    data.set_attr(**kwargs[key])  # may be char line or char map
                    data.is_set = True
                # fit factor
                elif isinstance(data, dc_fit):
                    if isinstance(kwargs[key], str):
                        if hasattr(data, kwargs[key]):
                            data.rule = kwargs[key]
                        else:
                            msg = f'Has no attribute {kwargs[key]} of {key} for component {self.label}'
                            logger.error(msg)
                            raise TypeError(msg)
                    elif isinstance(kwargs[key], dict):
                        if 'rule' in kwargs[key]:
                            if isinstance(kwargs[key]['rule'], str) and hasattr(data, kwargs[key]['rule']):
                                data.rule = kwargs[key]['rule']
                            else:
                                msg = f'Has no attribute {kwargs[key]["rule"]} of {key} for component {self.label}'
                                logger.error(msg)
                                raise TypeError(msg)
                        if 'func_params' in kwargs[key]:
                            if isinstance(kwargs[key]['func_params'], dict):
                                for func_key in kwargs[key]['func_params'].keys():
                                    if func_key in data.func_params:
                                        data.func_params[func_key].update(kwargs[key]['func_params'][func_key])
                                    else:
                                        msg = f'Has no attribute {func_key} in func params of {key} for component {self.label}'
                                        logger.error(msg)
                                        raise AttributeError(msg)
                            else:
                                msg = f'The func params of {key} for component {self.label} is not a dict.'
                                logger.error(msg)
                                raise TypeError(msg)
                # value specification for component properties
                elif isinstance(data, dc_cp) or isinstance(data, dc_simple):
                    if is_numeric:  # int/float type
                        data.set_attr(val=kwargs[key], is_set=True)
                        if isinstance(data, dc_cp):
                            data.set_attr(is_var=False)  # the known param
                    elif kwargs[key] == 'var' and isinstance(data, dc_cp):  # var value unknown(properties set)
                        data.set_attr(is_set=True, is_var=True)
                    elif isinstance(data, dc_simple):
                        data.set_attr(val=kwargs[key], is_set=True)
                    # invalid datatype for keyword
                    else:
                        msg = (
                            f"Bad datatype for keyword argument {key} for "
                            f"component {self.label}."
                        )
                        logger.error(msg)
                        raise TypeError(msg)
                elif isinstance(data, dc_cc) or isinstance(data, dc_cm):  # the fitted curve property
                    # value specification for characteristics
                    if (isinstance(kwargs[key], CharLine) or
                            isinstance(kwargs[key], CharMap)):
                        data.char_func = kwargs[key]   # just char function no char parameter
                        data.is_set = True
                    # invalid datatype for keyword
                    else:
                        msg = (
                            f"Bad datatype for keyword argument {key} for "
                            f"component {self.label}."
                        )
                        logger.error(msg)
                        raise TypeError(msg)
            elif 'unit' in key:
                prop = key.split('_unit')[0]
                if not prop in self.parameters:
                    msg = f"Component {self.__class__.__name__}: {self.label} has no attribute {key}."
                    logger.error(msg)
                    raise KeyError(msg)
                data = self.get_attr(prop)
                if isinstance(data, dc_cp):
                    if not hasattr(data, 'property_data'):
                        msg = f"Has wrong in code of parameters defined of component {self.__class__.__name__}: {self.label}."
                        logger.error(msg)
                        raise NotImplementedError(msg)
                    if kwargs[key] in data.property_data['units']:
                        data.unit = kwargs[key]
                    else:
                        msg = f"The parameter {prop} for component {self.__class__.__name__}: {self.label} has no unit: {kwargs[key]}."
                        logger.error(msg)
                        raise NotImplementedError(msg)
                else:
                    msg = f"The parameter {prop} for component {self.__class__.__name__}: {self.label} has no unit attribute: {key}."
                    logger.error(msg)
                    raise KeyError(msg)
            elif key in ['design', 'offdesign']:  # mode condition
                if not isinstance(kwargs[key], list):
                    msg = (
                        f"Please provide the {key} parameters as list for "
                        f"component {self.label}."
                    )
                    logger.error(msg)
                    raise TypeError(msg)
                if set(kwargs[key]).issubset(list(self.parameters.keys())):  # the design/offdesign must be parameters
                    self.__dict__.update({key: kwargs[key]})
                else:
                    keys = ", ".join(self.parameters.keys())
                    msg = (
                        "Available parameters for (off-)design specification "
                        f"of component {self.label} are: {keys}."
                    )
                    logger.error(msg)
                    raise ValueError(msg)
            elif key in ['local_design', 'local_offdesign', 'print_nodes',
                         'printout', 'char_warnings', 'bypass']:
                if not isinstance(kwargs[key], bool):
                    msg = (
                        f"Please provide the {key} parameters as bool for "
                        f"component {self.label}."
                    )
                    logger.error(msg)
                    raise TypeError(msg)
                else:
                    self.__dict__.update({key: kwargs[key]})
            elif key == 'design_path' or key == 'fkt_group':
                self.__dict__.update({key: kwargs[key]})
                self.new_design = True  # set another design path, reference another mode
            # invalid keyword
            else:
                msg = f"Component {self.__class__.__name__}: {self.label} has no attribute {key}."
                logger.error(msg)
                raise KeyError(msg)

    def get_attr(self, key):
        r"""
        Get the value of a component's attribute.

        Parameters
        ----------
        key : str
            The attribute you want to retrieve.

        Returns
        -------
        out :
            Value of specified attribute.
        """
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            msg = f"Component {self.label} has no attribute {key}."
            logger.error(msg)
            raise KeyError(msg)

    def _serialize(self):  # dict of setting
        export = {}
        for k in self._serializable():   # condition and others setting
            export.update({k: self.get_attr(k)})
        for k in self.parameters:   # param setting
            data = self.get_attr(k)
            export.update({k: data._serialize()})  # data dict of parameters (data container type)
        return {self.label: export}

    @staticmethod
    def _serializable():
        return [
            "design", "offdesign", "local_design", "local_offdesign",
            "design_path", "printout", "fkt_group", "char_warnings", "bypass"
        ]

    @staticmethod
    def is_branch_source():  # judge whether the component is the source comp of a branch
        return False

    def start_branch(self):
        msg = f'The component {self.__class__.__name__}: {self.label} has no branch start attribute.'
        logger.error(msg)
        raise NotImplementedError(msg)

    def propagate_to_target(self, branch):  # iteration to identify comps on the same branch
        inconn = branch["connections"][-1]  # inconn: object of inlet connection.
        conn_idx = self.inl.index(inconn)  # the index of branch
        outconn = self.outl[conn_idx]  # outconn: object of outlet connection.
        branch["connections"] += [outconn]  # add next connection belong to the same branch
        branch["components"] += [outconn.target]  # add next component belong to the same branch
        outconn.target.propagate_to_target(branch)  # the iteration of identifying comps/conns on the same branch

    @staticmethod
    def is_simplify_topology_start():
        return False

    @staticmethod
    def is_differential_component():
        return False

    def generate_nodes_constructure(self):
        """
        Generate the nodes' construction.

        properties generated:
        ---------------------
        self.nodes: pd.DataFrame
        self.nodes_properties: dict
        """
        msg = f'The code of component {self.__class__.__name__}: {self.label} has not defined nodes constructure.'
        logger.error(msg)
        raise NotImplementedError(msg)

    def simplify_nodes_topology(self):
        """
        Simplify the nodes' topology.
        Share objective and unit of properties.
        """
        msg = f'The code of component {self.__class__.__name__}: {self.label} has not defined simplify nodes topology.'
        logger.error(msg)
        raise NotImplementedError(msg)

    def init_nodes_properties_units_(self):
        """
        Initialize nodes' properties units.
        """
        msg = f'The code of component {self.__class__.__name__}: {self.label} has not defined initialize nodes properties units.'
        logger.error(msg)
        raise NotImplementedError(msg)

    @staticmethod
    def component(self):
        msg = f'The code of component {self.__class__.__name__}: {self.label} is incorrect.'
        logger.error(msg)
        raise NotImplementedError(msg)

    def component_type(self):
        msg = f'The code of component {self.__class__.__name__}: {self.label} has not defined type of component.'
        logger.error(msg)
        raise NotImplementedError(msg)

    def interface_type(self, port_id):
        msg = f'The code of component {self.__class__.__name__}: {self.label} has not defined port type of {port_id} in component.'
        logger.error(msg)
        raise NotImplementedError(msg)

    @staticmethod
    def inlets():
        return []

    @staticmethod
    def outlets():
        return []

    def initialise_source(self, c, key):
        r"""
        Return a starting value for pressure and enthalpy at outlet.

        Parameters
        ----------
        """
        return 0

    def initialise_target(self, c, key):
        r"""
        Return a starting value for pressure and enthalpy at inlet.

        Parameters
        ----------
        """
        return 0

    def get_parameters(self):  #
        return {}

    def get_mandatory_constraints(self):  #
        return {}

    def get_bypass_constraints(self):
        msg = (
            f"The component {self.label} of type {self.__class__.__name__} "
            "does not have bypassing functionality yet."
        )
        logger.exception(msg)
        raise NotImplementedError(msg)

    def get_liner_constraints(self):
        msg = (
            f"The component {self.label} of type {self.__class__.__name__} "
            f"does not have liner functionality yet."
        )
        logger.exception(msg)
        raise NotImplementedError(msg)

    def convert_set_property_to_criterion(self):
        """
        Convert property has been set to criterion.
        """
        properties_converted_list = []
        for key, val in self.parameters.items():  # component property initialization
            data = self.get_attr(key)
            if isinstance(data, dc_cp) and data.is_set and hasattr(data, 'property_data'):
                properties_converted_list.append(data)
        for container in properties_converted_list:
            container.val_SI = convert_to_SI(container.property_data, container.val, container.unit)

    def convert_set_property_from_criterion(self):
        """
        Convert all properties documented from criterion to user defined format.
        """
        properties_converted_list = []
        for key, val in self.parameters.items():  # component property initialization
            data = self.get_attr(key)
            if isinstance(data, dc_cp) and hasattr(data, 'property_data'):
                properties_converted_list.append(data)
        for container in properties_converted_list:
            container.val = convert_from_SI(container.property_data, container.val_SI, container.unit)

    def summarize_equations(self):
        """
        Summarize the equations of the component.

        Returns
        -------

        """
        self.it = 0  #
        self.num_eq = 0  # the number of all equations of constraints and parameters
        if self.bypass:
            self.constraints = self.get_bypass_constraints().copy()
        else:
            self.constraints = self.get_mandatory_constraints().copy()
        # add constraints to components
        self.__dict__.update(self.constraints)
        # configure the constraint equations first
        for key, constraint in self.constraints.items():
            num_eq = constraint.num_eq
            if num_eq > 0 and constraint.take_effect():
                constraint.label = f"<component constraint>: {key} of {self.__class__.__name__}: {self.label}"
                self.network.sorted_equations_module_container.append(constraint)
                self.num_eq += num_eq
        # bypass: no parameters
        if self.bypass:
            return
        # configure the parameters equations
        for key, val in self.parameters.items():  # component property initialization
            data = self.get_attr(key)
            if isinstance(val, dc_cp):  # component properties
                pass
            # component characteristics
            elif isinstance(val, dc_cc):  # the fitted curve of component
                if data.is_set and data.char_func is None:  # set default fitted curve of component
                    try:
                        data.char_func = ldc(self.component(), key, 'DEFAULT', CharLine)
                    except KeyError:
                        data.char_func = CharLine(x=[0, 1], y=[1, 1])
            # component characteristics
            elif isinstance(val, dc_cm):  # the fitted map of component
                if data.is_set and data.char_func is None:  # set default fitted map
                    try:
                        data.char_func = ldc(self.component(), key, 'DEFAULT', CharMap)
                    except KeyError:
                        data.char_func = CharLine(x=[0, 1], y=[1, 1])  #
            # grouped component properties
            elif isinstance(val, dc_gcp):
                is_set = True
                for e in data.elements:  # make sure all elements of grouped component properties be set
                    if not self.get_attr(e).is_set:  # make sure all elements used has been set
                        is_set = False
                if is_set:
                    data.set_attr(is_set=True)
                elif data.is_set:
                    start = (
                            'All parameters of the component group have to be '
                            'specified! This component group uses the following '
                            'parameters: '
                        )
                    end = f" at {self.label}. Group will be set to False."
                    logger.warning(start + ', '.join(val.elements) + end)
                    data.set_attr(is_set=False)
                else:
                    data.set_attr(is_set=False)
            # component properties
            # add the number of no constraints equations
            # only parameters set are contained in iteration
            if data.is_set and data.func is not None:
                if data.take_effect():
                    data.label = f"<component parameter>: {key} of {self.__class__.__name__}: {self.label}"
                    self.network.sorted_equations_module_container.append(data)
                    self.num_eq += data.num_eq
        # logger for debug
        msg = f"The component: {self.label} of type {self.__class__.__name__} has {self.num_eq} equations."
        logger.debug(msg)

    def summarize_variables(self):
        """
        Summarize the variables of the component, delivering to container of network.

        Returns
        -------

        """
        self.num_vars = 0  # the number of variables belong to comp
        for key, val in self.parameters.items():  # component property initialization
            data = self.get_attr(key)
            if isinstance(val, dc_cp):  # component properties
                if data.is_var:
                    data.label = f"<component property>: {key} of {self.__class__.__name__}: {self.label}"
                    data.J_col = self.network.num_conn_vars + self.network.num_comp_vars + self.num_vars  # the column in Jacobin matrix
                    self.network.sorted_variables_module_container.append({'object': data, 'type': 'component variable'})
                    self.num_vars += data.dimension
        # done
        msg = f"The component: {self.label} of type {self.__class__.__name__} has {self.num_vars} variables."
        logger.debug(msg)

    def preprocess(self):  # used in function: init_design and init_offdesign in network
        r"""
        Perform component initialization in network preprocessing.

        Parameters
        ----------

        """
        # component properties
        self.prop_specifications = {}  # param setting dict
        self.var_specifications = {}  # var setting dict
        # component characteristics / component characteristic maps
        self.char_specifications = {}  # component characteristics: char fitted curve/map setting dict
        # grouped component properties / grouped component characteristics
        self.group_specifications = {}  # group properties/char setting dict
        if not self.bypass:
            for key, val in self.parameters.items():  # component property initialization
                data = self.get_attr(key)
                if isinstance(val, dc_cp):  # component properties
                    self.prop_specifications[key] = data.is_set  # value set dict
                    self.var_specifications[key] = data.is_var  # var set dict
                # component characteristics
                elif isinstance(val, dc_cc):  # the fitted curve of component
                    if data.func is not None:
                        self.char_specifications[key] = data.is_set  # char set dict
                # component characteristics
                elif isinstance(val, dc_cm):  # the fitted map of component
                    if data.func is not None:
                        self.char_specifications[key] = data.is_set  # char set dict
                # grouped component properties
                elif isinstance(val, dc_gcp):
                    self.group_specifications[key] = data.is_set  # group properties data set dict
                # grouped component characteristics
                elif isinstance(val, dc_gcc):
                    self.group_specifications[key] = data.is_set  # group characteristics data set dict

    @staticmethod
    def is_variable(var, increment_filter=None):  # judge whether the key is variable or not
        if var.is_var:  # var.J_col: the column in Jacobin
            if increment_filter is None or not increment_filter[var.J_col]:  # increment_filter: Matrix for filtering non-changing variables
                return True
        return False

    def correct_attracting_basin_path_(self):
        pass

    def numeric_deriv(self, func, dx, conn=None, **kwargs):  # calculate discrete derivatives
        r"""
        Calculate partial derivative of the function func to dx.
        For details see :py:func:`Aurora.tools.helpers._numeric_deriv`
        """

        def _numeric_deriv(obj, func, dx, conn=None, **kwargs):
            if conn is None:
                d = obj.get_attr(dx).d  # d: the value of parameter delta in data container
                exp = 0
                obj.get_attr(dx).val_SI += d
                exp += func(**kwargs)

                obj.get_attr(dx).val_SI -= 2 * d
                exp -= func(**kwargs)
                deriv = exp / (2 * d)  # discrete derivative (slope coefficient)

                obj.get_attr(dx).val_SI += d  # restore the value of data container

            elif dx in self.inl[0].connections_properties_data[self.inl[0].connection_type()]:
                if dx == 'm':
                    d = 1e-4
                else:
                    d = 1e-1
                conn.get_attr(dx).val_SI += d
                exp = func(**kwargs)

                conn.get_attr(dx).val_SI -= 2 * d
                exp -= func(**kwargs)
                deriv = exp / (2 * d)

                conn.get_attr(dx).val_SI += d

            else:
                msg = (
                    "Your variable specification for the numerical derivative "
                    "calculation seems to be wrong. It has to be a fluid name, m, "
                    "p, h or the name of a component variable."
                )
                logger.exception(msg)
                raise ValueError(msg)
            return deriv

        return _numeric_deriv(self, func, dx, conn, **kwargs)

    def numeric_tensor(self, func, dx1, dx2, conn1=None, conn2=None, **kwargs):
        if dx1 == dx2 and conn1 == conn2:
            if conn1 is None and hasattr(self, dx1):
                d = self.get_attr(dx1).d
                original_val = self.get_attr(dx1).val
                exp = 0
                exp -= 2 * func(**kwargs)
                self.get_attr(dx1).val = original_val + 2*d
                exp += func(**kwargs)
                self.get_attr(dx1).val = original_val - 2*d
                exp += func(**kwargs)
                self.get_attr(dx1).val = original_val
                tensor = exp / (4 * d**2)

            elif dx1 in conn1.fluid.is_var:  # calculate the derivative of variable of fluid composition
                d = 1e-5
                original_val = conn1.fluid.val[dx1]
                exp = 0
                exp -= 2 * func(**kwargs)

                if original_val + 2*d <= 1:
                    conn1.fluid.val[dx1] = original_val + 2*d
                else:
                    conn1.fluid.val[dx1] = 1
                conn1.build_fluid_data()
                exp += func(**kwargs)

                if original_val - 2 * d >= 0:
                    conn1.fluid.val[dx1] = original_val - 2*d
                else:
                    conn1.fluid.val[dx1] = 0
                conn1.build_fluid_data()
                exp += func(**kwargs)

                conn1.fluid.val[dx1] = original_val
                conn1.build_fluid_data()
                tensor = exp / (4 * d**2)

            elif dx1 in ['m', 'p', 'h']:
                if dx1 == 'm':
                    d = 1e-4
                else:
                    d = 1e-1
                original_val = conn1.get_attr(dx1).val_SI
                exp = 0
                exp -= 2 * func(**kwargs)
                conn1.get_attr(dx1).val_SI = original_val + 2*d
                exp += func(**kwargs)
                conn1.get_attr(dx1).val_SI = original_val - 2*d
                exp += func(**kwargs)
                conn1.get_attr(dx1).val_SI = original_val
                tensor = exp / (4 * d**2)
            else:
                msg = (
                    "Your variable specification for the numerical derivative "
                    "calculation seems to be wrong. It has to be a fluid name, m, "
                    "p, h or the name of a component variable."
                )
                logger.exception(msg)
                raise ValueError(msg)
        else:
            if dx1 in ['m', 'p', 'h']:
                if dx1 == 'm':
                    d1 = 1e-4
                else:
                    d1 = 1e-1
                original_val1 = conn1.get_attr(dx1).val_SI
                exp = 0
                if dx2 in ['m', 'p', 'h']:
                    if dx2 == 'm':
                        d2 = 1e-4
                    else:
                        d2 = 1e-1
                    original_val2 = conn2.get_attr(dx2).val_SI

                    conn1.get_attr(dx1).val_SI = original_val1 + d1
                    conn2.get_attr(dx2).val_SI = original_val2 + d2
                    exp += func(**kwargs)
                    conn1.get_attr(dx1).val_SI = original_val1 - d1
                    conn2.get_attr(dx2).val_SI = original_val2 - d2
                    exp += func(**kwargs)
                    conn1.get_attr(dx1).val_SI = original_val1 + d1
                    conn2.get_attr(dx2).val_SI = original_val2 - d2
                    exp -= func(**kwargs)
                    conn1.get_attr(dx1).val_SI = original_val1 - d1
                    conn2.get_attr(dx2).val_SI = original_val2 + d2
                    exp -= func(**kwargs)

                    conn2.get_attr(dx2).val_SI = original_val2
                    tensor = exp / (4 * d1 * d2)
                elif dx2 in conn2.fluid.is_var:
                    d2 = 1e-5
                    original_val2 = conn2.fluid.val[dx2]

                    conn1.get_attr(dx1).val_SI = original_val1 + d1
                    if original_val2 + d2 <= 1:
                        conn2.fluid.val[dx2] = original_val2 + d2
                    else:
                        conn2.fluid.val[dx2] = 1
                    conn2.build_fluid_data()
                    exp += func(**kwargs)

                    conn1.get_attr(dx1).val_SI = original_val1 - d1
                    if original_val2 - d2 >= 0:
                        conn2.fluid.val[dx2] = original_val2 - d2
                    else:
                        conn2.fluid.val[dx2] = 0
                    conn2.build_fluid_data()
                    exp += func(**kwargs)

                    conn1.get_attr(dx1).val_SI = original_val1 - d1
                    if original_val2 + d2 <= 1:
                        conn2.fluid.val[dx2] = original_val2 + d2
                    else:
                        conn2.fluid.val[dx2] = 1
                    conn2.build_fluid_data()
                    exp -= func(**kwargs)

                    conn1.get_attr(dx1).val_SI = original_val1 + d1
                    if original_val2 - d2 >= 0:
                        conn2.fluid.val[dx2] = original_val2 - d2
                    else:
                        conn2.fluid.val[dx2] = 0
                    conn2.build_fluid_data()
                    exp -= func(**kwargs)

                    conn2.fluid.val[dx2] = original_val2
                    conn2.build_fluid_data()
                    tensor = exp / (4 * d1 * d2)
                elif conn2 is None and hasattr(self, dx2):
                    d2 = self.get_attr(dx2).d
                    original_val2 = self.get_attr(dx2).val
                    conn1.get_attr(dx1).val_SI = original_val1 + d1

                    self.get_attr(dx2).val = original_val2 + d2
                    exp += func(**kwargs)
                    conn1.get_attr(dx1).val_SI = original_val1 - d1
                    self.get_attr(dx2).val = original_val2 - d2
                    exp += func(**kwargs)
                    conn1.get_attr(dx1).val_SI = original_val1 + d1
                    self.get_attr(dx2).val = original_val2 - d2
                    exp -= func(**kwargs)
                    conn1.get_attr(dx1).val_SI = original_val1 - d1
                    self.get_attr(dx2).val = original_val2 + d2
                    exp -= func(**kwargs)

                    self.get_attr(dx2).val = original_val2
                    tensor = exp / (4 * d1 * d2)
                else:
                    msg = (
                        "Your variable specification for the numerical derivative "
                        "calculation seems to be wrong. It has to be a fluid name, m, "
                        "p, h or the name of a component variable."
                    )
                    logger.exception(msg)
                    raise ValueError(msg)
                conn1.get_attr(dx1).val_SI = original_val1
            elif dx1 in conn1.fluid.is_var:
                d1 = 1e-5
                original_val1 = conn1.fluid.val[dx1]
                exp = 0
                if dx2 in ['m', 'p', 'h']:
                    if dx2 == 'm':
                        d2 = 1e-4
                    else:
                        d2 = 1e-1
                    original_val2 = conn2.get_attr(dx2).val_SI

                    if original_val1 + d1 <= 1:
                        conn1.fluid.val[dx1] = original_val1 + d1
                    else:
                        conn1.fluid.val[dx1] = 1
                    conn2.get_attr(dx2).val_SI = original_val2 + d2
                    conn1.build_fluid_data()
                    exp += func(**kwargs)

                    if original_val1 - d1 >= 0:
                        conn1.fluid.val[dx1] = original_val1 - d1
                    else:
                        conn1.fluid.val[dx1] = 0
                    conn2.get_attr(dx2).val_SI = original_val2 - d2
                    conn1.build_fluid_data()
                    exp += func(**kwargs)

                    if original_val1 + d1 <= 1:
                        conn1.fluid.val[dx1] = original_val1 + d1
                    else:
                        conn1.fluid.val[dx1] = 1
                    conn2.get_attr(dx2).val_SI = original_val2 - d2
                    conn1.build_fluid_data()
                    exp -= func(**kwargs)

                    if original_val1 - d1 >= 0:
                        conn1.fluid.val[dx1] = original_val1 - d1
                    else:
                        conn1.fluid.val[dx1] = 0
                    conn2.get_attr(dx2).val_SI = original_val2 + d2
                    conn1.build_fluid_data()
                    exp -= func(**kwargs)

                    conn2.get_attr(dx2).val_SI = original_val2
                    tensor = exp / (4 * d1 * d2)
                elif dx2 in conn2.fluid.is_var:
                    d2 = 1e-5
                    original_val2 = conn2.fluid.val[dx2]

                    if original_val1 + d1 <= 1:
                        conn1.fluid.val[dx1] = original_val1 + d1
                    else:
                        conn1.fluid.val[dx1] = 1
                    if original_val2 + d2 <= 1:
                        conn2.fluid.val[dx2] = original_val2 + d2
                    else:
                        conn2.fluid.val[dx2] = 1
                    conn1.build_fluid_data()
                    conn2.build_fluid_data()
                    exp += func(**kwargs)

                    if original_val1 - d1 >= 0:
                        conn1.fluid.val[dx1] = original_val1 - d1
                    else:
                        conn1.fluid.val[dx1] = 0
                    if original_val2 - d2 >= 0:
                        conn2.fluid.val[dx2] = original_val2 - d2
                    else:
                        conn2.fluid.val[dx2] = 0
                    conn1.build_fluid_data()
                    conn2.build_fluid_data()
                    exp += func(**kwargs)

                    if original_val1 + d1 <= 1:
                        conn1.fluid.val[dx1] = original_val1 + d1
                    else:
                        conn1.fluid.val[dx1] = 1
                    if original_val2 - d2 >= 0:
                        conn2.fluid.val[dx2] = original_val2 - d2
                    else:
                        conn2.fluid.val[dx2] = 0
                    conn1.build_fluid_data()
                    conn2.build_fluid_data()
                    exp -= func(**kwargs)

                    if original_val1 - d1 >= 0:
                        conn1.fluid.val[dx1] = original_val1 - d1
                    else:
                        conn1.fluid.val[dx1] = 0
                    if original_val2 + d2 <= 1:
                        conn2.fluid.val[dx2] = original_val2 + d2
                    else:
                        conn2.fluid.val[dx2] = 1
                    conn1.build_fluid_data()
                    conn2.build_fluid_data()
                    exp -= func(**kwargs)

                    conn2.fluid.val[dx2] = original_val2
                    conn2.build_fluid_data()
                    tensor = exp / (4 * d1 * d2)
                elif conn2 is None and hasattr(self, dx2):
                    d2 = self.get_attr(dx2).d
                    original_val2 = self.get_attr(dx2).val

                    if original_val1 + d1 <= 1:
                        conn1.fluid.val[dx1] = original_val1 + d1
                    else:
                        conn1.fluid.val[dx1] = 1
                    self.get_attr(dx2).val = original_val2 + d2
                    conn1.build_fluid_data()
                    exp += func(**kwargs)

                    if original_val1 - d1 >= 0:
                        conn1.fluid.val[dx1] = original_val1 - d1
                    else:
                        conn1.fluid.val[dx1] = 0
                    self.get_attr(dx2).val = original_val2 - d2
                    conn1.build_fluid_data()
                    exp += func(**kwargs)

                    if original_val1 + d1 <= 1:
                        conn1.fluid.val[dx1] = original_val1 + d1
                    else:
                        conn1.fluid.val[dx1] = 1
                    self.get_attr(dx2).val = original_val2 - d2
                    conn1.build_fluid_data()
                    exp -= func(**kwargs)

                    if original_val1 - d1 >= 0:
                        conn1.fluid.val[dx1] = original_val1 - d1
                    else:
                        conn1.fluid.val[dx1] = 0
                    self.get_attr(dx2).val = original_val2 + d2
                    conn1.build_fluid_data()
                    exp -= func(**kwargs)

                    self.get_attr(dx2).val = original_val2
                    tensor = exp / (4 * d1 * d2)
                else:
                    msg = (
                        "Your variable specification for the numerical derivative "
                        "calculation seems to be wrong. It has to be a fluid name, m, "
                        "p, h or the name of a component variable."
                    )
                    logger.exception(msg)
                    raise ValueError(msg)
                conn1.fluid.val[dx1] = original_val1
                conn1.build_fluid_data()
            elif conn1 is None and hasattr(self, dx1):
                d1 = self.get_attr(dx1).d
                original_val1 = self.get_attr(dx1).val
                exp = 0
                if dx2 in ['m', 'p', 'h']:
                    if dx2 == 'm':
                        d2 = 1e-4
                    else:
                        d2 = 1e-1
                    original_val2 = conn2.get_attr(dx2).val_SI

                    self.get_attr(dx1).val = original_val1 + d1
                    conn2.get_attr(dx2).val_SI = original_val2 + d2
                    exp += func(**kwargs)
                    self.get_attr(dx1).val = original_val1 - d1
                    conn2.get_attr(dx2).val_SI = original_val2 - d2
                    exp += func(**kwargs)
                    self.get_attr(dx1).val = original_val1 + d1
                    conn2.get_attr(dx2).val_SI = original_val2 - d2
                    exp -= func(**kwargs)
                    self.get_attr(dx1).val = original_val1 - d1
                    conn2.get_attr(dx2).val_SI = original_val2 + d2
                    exp -= func(**kwargs)

                    conn2.get_attr(dx2).val_SI = original_val2
                    tensor = exp / (4 * d1 * d2)
                elif dx2 in conn2.fluid.is_var:
                    d2 = 1e-5
                    original_val2 = conn2.fluid.val[dx2]

                    self.get_attr(dx1).val = original_val1 + d1
                    if original_val2 + d2 <= 1:
                        conn2.fluid.val[dx2] = original_val2 + d2
                    else:
                        conn2.fluid.val[dx2] = 1
                    conn2.build_fluid_data()
                    exp += func(**kwargs)

                    self.get_attr(dx1).val = original_val1 - d1
                    if original_val2 - d2 >= 0:
                        conn2.fluid.val[dx2] = original_val2 - d2
                    else:
                        conn2.fluid.val[dx2] = 0
                    conn2.build_fluid_data()
                    exp += func(**kwargs)

                    self.get_attr(dx1).val = original_val1 - d1
                    if original_val2 + d2 <= 1:
                        conn2.fluid.val[dx2] = original_val2 + d2
                    else:
                        conn2.fluid.val[dx2] = 1
                    conn2.build_fluid_data()
                    exp -= func(**kwargs)

                    self.get_attr(dx1).val = original_val1 + d1
                    if original_val2 - d2 >= 0:
                        conn2.fluid.val[dx2] = original_val2 - d2
                    else:
                        conn2.fluid.val[dx2] = 0
                    conn2.build_fluid_data()
                    exp -= func(**kwargs)

                    conn2.fluid.val[dx2] = original_val2
                    conn2.build_fluid_data()
                    tensor = exp / (4 * d1 * d2)
                else:
                    msg = (
                        "Your variable specification for the numerical derivative "
                        "calculation seems to be wrong. It has to be a fluid name, m, "
                        "p, h or the name of a component variable."
                    )
                    logger.exception(msg)
                    raise ValueError(msg)
                self.get_attr(dx1).val = original_val1
            else:
                msg = (
                    "Your variable specification for the numerical derivative "
                    "calculation seems to be wrong. It has to be a fluid name, m, "
                    "p, h or the name of a component variable."
                )
                logger.exception(msg)
                raise ValueError(msg)
        return tensor

    def generate_numerical_tensor(self, f, k, numeric_variables_list, **kwargs):
        for i in range(len(numeric_variables_list)):
            for j in range(i + 1):
                if i == j:
                    if numeric_variables_list[i][1]:
                        self.network.tensor[numeric_variables_list[i][3], numeric_variables_list[i][3], k] = (
                            self.numeric_tensor(f, numeric_variables_list[i][0], numeric_variables_list[i][0],
                                                numeric_variables_list[i][2], numeric_variables_list[i][2], **kwargs))
                else:
                    if numeric_variables_list[i][1] and numeric_variables_list[j][1]:
                        tensor_2 = self.numeric_tensor(f, numeric_variables_list[i][0], numeric_variables_list[j][0],
                                    numeric_variables_list[i][2], numeric_variables_list[j][2], **kwargs)
                        self.network.tensor[numeric_variables_list[i][3], numeric_variables_list[j][3], k] = tensor_2
                        self.network.tensor[numeric_variables_list[j][3], numeric_variables_list[i][3], k] = tensor_2

    def generate_numerical_bus_tensor(self, f, k, numeric_variables_list, **kwargs):
        for i in range(len(numeric_variables_list)):
            for j in range(i + 1):
                if i == j:
                    if numeric_variables_list[i][1]:
                        self.network.tensor[numeric_variables_list[i][3], numeric_variables_list[i][3], k] += (
                            self.numeric_tensor(f, numeric_variables_list[i][0], numeric_variables_list[i][0],
                                                numeric_variables_list[i][2], numeric_variables_list[i][2], **kwargs))
                else:
                    if numeric_variables_list[i][1] and numeric_variables_list[j][1]:
                        tensor_2 = self.numeric_tensor(f, numeric_variables_list[i][0], numeric_variables_list[j][0],
                                                       numeric_variables_list[i][2], numeric_variables_list[j][2],
                                                       **kwargs)
                        self.network.tensor[numeric_variables_list[i][3], numeric_variables_list[j][3], k] += tensor_2
                        self.network.tensor[numeric_variables_list[j][3], numeric_variables_list[i][3], k] += tensor_2

    def get_char_expr(self, param, type='rel', inconn=0, outconn=0):  # need to be rewritten !!!
        r"""
        Generic method to access characteristic function parameters (ratio).

        Parameters
        ----------
        param : str
            Parameter for characteristic function evaluation.

        type : str
            Type of expression:

            - :code:`rel`: relative to design value
            - :code:`abs`: absolute value

        inconn : int
            Index of inlet connection.

        outconn : int
            Index of outlet connection.

        Returns
        -------
        expr : float
            Value of expression
        """
        pass

    def bus_func(self, bus):  # will be rewritten by the children class
        r"""
        Base method for calculation of the value of the bus function.

        The method is different due to the type of component.

        Parameters
        ----------
        bus : tespy.connections.bus.Bus
            TESPy bus object.

        Returns
        -------
        residual : float
            Residual value of bus equation.   ??????
        """
        return 0

    def calc_bus_expr(self, bus):
        r"""
        Return the busses' characteristic line input expression (x).

        The iteration of offdesign bus value is independent of the network iteration,
        since bus value is determined by the network iteration in one direction.

        Parameters
        ----------
        bus : tespy.connections.bus.Bus
            Bus to calculate the characteristic function expression for.

        Returns
        -------
        expr : float
            Ratio of power to power design depending on the bus base
            specification.
        """
        b = bus.comps.loc[self]  #
        if np.isnan(b['P_ref']) or b['P_ref'] == 0:
            return 1
        else:
            comp_val = self.bus_func(b)  # value of component bus equation
            if b['base'] == 'component':
                return abs(comp_val / b['P_ref'])
            else:
                kwargs = {
                    "function": bus_char_evaluation,  # function to calculate the residual value of a bus.
                    "parameter": "bus_value",  # bus_value: _ declare, no works
                    "component_value": comp_val,  # Value of the energy transfer at the component.
                    "reference_value": b["P_ref"],  # the reference design value of component bus
                    "char_func": b["char"]  # the interpolation function of charline(base bus)
                }
                bus_value = newton_with_kwargs(
                    derivative=bus_char_derivative,  # function to calculate discrete derivative for bus char evaluation
                    target_value=0,
                    val0=b['P_ref'],  # the design value of bus.P is set as the initialized value in bus iteration
                    valmin=-1e15,
                    valmax=1e15,
                    **kwargs
                )  # bus_value: the revisionary component bus value depend on charline base bus
                return bus_value / b['P_ref']

    def calc_bus_efficiency(self, bus):  # calculate the char efficiency of component bus value compared with bus
        r"""
        Return the busses' efficiency (y).

        Parameters
        ----------
        bus : tespy.connections.bus.Bus
            Bus to calculate the efficiency value on.

        Returns
        -------
        efficiency : float
            Efficiency value of the bus.

            .. math::

                \eta_\mathrm{bus} = \begin{cases}
                \eta\left(
                \frac{\dot{E}_\mathrm{bus}}{\dot{E}_\mathrm{bus,ref}}\right) &
                \text{bus base = 'bus'}\\
                \eta\left(
                \frac{\dot{E}_\mathrm{component}}
                {\dot{E}_\mathrm{component,ref}}\right) &
                \text{bus base = 'component'}
                \end{cases}

        Note
        ----
        If the base value of the bus is the bus value itself, a newton
        iteration is used to find the bus value satisfying the corresponding
        equation (case 1).
        """
        return bus.comps.loc[self, 'char'].evaluate(self.calc_bus_expr(bus))

    def calc_bus_value(self, bus):
        r"""
        Return the buses' value of the component's energy transfer.

        Parameters
        ----------
        bus : tespy.connections.bus.Bus
            Bus to calculate energy transfer on.

        Returns
        -------
        bus_value : float
            Value of the energy transfer on the specified bus.

            .. math::

                \dot{E}_\mathrm{bus} = \begin{cases}
                \frac{\dot{E}_\mathrm{component}}{f\left(
                \frac{\dot{E}_\mathrm{bus}}{\dot{E}_\mathrm{bus,ref}}\right)} &
                \text{bus base = 'bus'}\\
                \dot{E}_\mathrm{component} \cdot f\left(
                \frac{\dot{E}_\mathrm{component}}
                {\dot{E}_\mathrm{component,ref}}\right) &
                \text{bus base = 'component'}
                \end{cases}

        Note
        ----
        If the base value of the bus objective is the bus value itself, a newton
        iteration is used to find the bus value satisfying the corresponding
        equation (case 1).
        """
        b = bus.comps.loc[self]
        comp_val = self.bus_func(b)
        expr = self.calc_bus_expr(bus)  # the x ratio due to the changing of condition
        # calculate the revisionary component bus value due to the char efficiency of component bus value compared with bus
        if b['base'] == 'component':
            return comp_val * b['char'].evaluate(expr)
        else:
            return comp_val / b['char'].evaluate(expr)

    def calc_parameters(self):  #
        r"""Postprocessing parameter calculation."""
        return

    def calc_nodes_properties(self):
        r"""Postprocessing node properties calculation."""
        for side in self.nodes.index:
            for col in range(self.nodes_num):
                node = self.nodes.loc[side, col]
                node.calc_properties()
                for key in node.properties.keys():
                    data = node.get_attr(key)
                    if (hasattr(data, 'is_result') and data.is_result) and (hasattr(data, 'property_data') and data.unit):
                        data.val = convert_from_SI(data.property_data, data.val_SI, data.unit)

    def check_parameter_bounds(self):  # should contain all properties, need to be rewritten !!!
        r"""Check parameter value limits."""
        for p in self.parameters.keys():
            data = self.get_attr(p)
            if isinstance(data, dc_cp):
                valid_value = True
                try:
                    float(data.val_SI)
                except (TypeError, ValueError, AttributeError) as e:
                    valid_value = False
                if not valid_value:
                    msg = f'Invalid parameter value {p}: {data.val_SI} at component {self.__class__.__name__}: {self.label}'
                    logger.warning(msg)
                elif data.val_SI > data.max_val + ERR:
                    msg = (
                        f"Invalid value for {p}: {p} = {data.val_SI} above "
                        f"maximum value ({data.max_val}) ({data.unit}) at component {self.__class__.__name__} "
                        f"{self.label}."
                    )
                    logger.warning(msg)
                elif data.val_SI < data.min_val - ERR:
                    msg = (
                        f"Invalid value for {p}: {p} = {data.val_SI} below "
                        f"minimum value ({data.min_val}) ({data.unit}) at component {self.__class__.__name__} "
                        f"{self.label}."
                    )
                    logger.warning(msg)
            elif isinstance(data, dc_cc) and data.is_set:
                expr = self.get_char_expr(data.param, **data.char_params)
                data.char_func.get_domain_errors(expr, self.label)
            elif isinstance(data, dc_gcc) and data.is_set:
                for char in data.elements:
                    char_data = self.get_attr(char)
                    expr = self.get_char_expr(char_data.param, **char_data.char_params)
                    char_data.char_func.get_domain_errors(expr, self.label)

    def convergence_check(self):
        return

    def boundary_check(self):
        return

    def entropy_balance(self):  #
        r"""Entropy balance calculation method."""
        return

    def exergy_balance(self, T0):  #
        r"""
        Exergy balance calculation method.

        Parameters
        ----------
        T0 : float
            Ambient temperature T0 / K.
        """
        self.E_P = np.nan  # Product(outlet) Exergy
        self.E_F = np.nan  # Fuel(inlet) Exergy
        self.E_bus = {"chemical": np.nan, "physical": np.nan, "massless": np.nan}  # generate bus exergy
        self.E_D = np.nan  # Destruction Exergy due to The increase in entropy brought by irreversible thermodynamic processes.
        self.epsilon = self._calc_epsilon()  # Efficiency

    def _calc_epsilon(self):  # calculate Second Law Efficiency
        if self.E_F == 0:
            return np.nan
        else:
            return self.E_P / self.E_F

    def get_plotting_data(self):
        return

    @staticmethod
    def generate_interpolated_array_(start_val, end_val, nodes_num):
        """
        Generate interpolated array from start and end values.

        :parameter:
            start_val: start value
            end_val: end value
            nodes_num: number of nodes

        :return:
            np.ndarray: array of interpolated values
        """
        mid_idx = nodes_num // 2
        temp_nodes = nodes_num - 1
        temp_array = np.linspace(start_val, end_val, temp_nodes)
        result = np.empty(nodes_num, dtype=float)
        result[:mid_idx] = temp_array[:mid_idx]
        result[mid_idx:] = temp_array[mid_idx - 1:]
        return result

